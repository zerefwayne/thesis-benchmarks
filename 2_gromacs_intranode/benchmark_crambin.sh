#!/bin/bash
#SBATCH --job-name=gromacs_crambin
#SBATCH --account=project_462000226
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --exclusive
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --time=01:00:00
#SBATCH --output=results/%x_%j.out
#SBATCH --constraint=eessi
#
# Crambin (~20k atoms). Small system: launch/runtime-overhead bound.
# Mirrors flags from 2_gbs_eessi/benchmark_crambin.sh — only the launcher
# differs between stacks.
# Per-run wall ~30 s; 7 runs ~3-4 min — 1 h gives wide headroom.
#
# One SLURM job runs ONE stack only. Submit:
#   sbatch benchmark_crambin.sh eessi
#   sbatch benchmark_crambin.sh native

STACK="${1:?usage: sbatch $0 <eessi|native>}"
if [[ "$STACK" != "eessi" && "$STACK" != "native" ]]; then
    echo "ERROR: stack must be 'eessi' or 'native', got '$STACK'" >&2
    exit 1
fi

source common.sh

BENCHMARK="crambin"
FILE_BASE="gromacs_${BENCHMARK}_${STACK}"
CSV_FILE="${RESULT_DIR}/${FILE_BASE}_${SLURM_JOB_ID}.csv"
LOG_FILE="${RESULT_DIR}/${FILE_BASE}_${SLURM_JOB_ID}.log"
META_FILE="${RESULT_DIR}/${FILE_BASE}_${SLURM_JOB_ID}.meta"

NUM_RUNS=7         # 1 warm-up (discarded) + 6 recorded — matches the
                   # AMD ROCm Blog LUMI guide ("averaged each configuration
                   # over 6 benchmark runs") with an extra discarded warmup
                   # for cold-cache / GPU JIT / MPI connection setup.
WARMUP_RUN=1
INPUT_TPR="$(pwd)/GROMACS_Benchmark_Suite/HECBioSim/Crambin/benchmark.tpr"
WORKDIR="$(pwd)/${RESULT_DIR}/${FILE_BASE}_${SLURM_JOB_ID}_work"
SELECT_GPU="$(pwd)/select_gpu"
mkdir -p "$WORKDIR"

MDRUN_FLAGS=(-nb gpu -pme gpu -update gpu -bonded gpu
             -nsteps 100000 -resethway -noconfout -npme 1)

echo "benchmark,stack,jobid,run,perf_ns_per_day,wall_s,core_s,ntmpi,toolchain" > "$CSV_FILE"

# Native: srun + L3CC-aware CPU mask + per-rank GPU wrapper (AMD ROCm Blog recipe).
run_mdrun_native() {
    local r=$1
    local deffnm="$WORKDIR/run${r}"
    srun --cpu-bind="$CPU_BIND" "$SELECT_GPU" \
        gmx_mpi mdrun -s "$INPUT_TPR" -deffnm "$deffnm" "${MDRUN_FLAGS[@]}" \
        2>&1 | tee -a "$LOG_FILE"
}

# EESSI: mpirun (OpenMPI 5/PRRTE) — NO select_gpu wrapper. Under EESSI
# 2025.06's OpenMPI the per-rank local-id env vars (SLURM_LOCALID,
# OMPI_COMM_WORLD_LOCAL_RANK, PMIX_RANK, ...) are not visible to wrapped
# processes, so the wrapper collapsed every rank to LOCAL_ID=0 ("PP:0,...,
# PME:0" in the md.log). Letting GROMACS auto-distribute across the 8
# GCDs is what worked pre-wrapper (PP:0,PP:1,...,PME:7).
#
# CPU pinning IS done here, via OpenMPI 5/PRRTE's mapper. This is the
# cross-launcher analogue of native's `srun --cpu-bind=$CPU_BIND`:
#   --map-by ppr:1:l3cache:PE=7  : one rank per L3 cache complex, with 7
#                                  PEs (cores) allocated per rank.
# PRRTE rejects an explicit `--bind-to l3cache` alongside `PE=7` ("the
# PE=<list> mapping directive cannot be combined with a binding directive
# other than 'core' or 'hwt'"), so we omit it — PE=7 already implies the
# rank is bound to its 7 assigned cores. LUMI's Trento CPU has 8 L3CCs
# (8 cores each) so 8 ranks fit perfectly.
# Without this we observed 5x higher run-to-run variance vs native
# (CV ~10% vs ~2% on Crambin) — threads were floating across NUMA nodes.
run_mdrun_eessi() {
    local r=$1
    local deffnm="$WORKDIR/run${r}"
    mpirun -np 8 --map-by ppr:1:l3cache:PE=7 --oversubscribe \
        gmx_mpi mdrun -s "$INPUT_TPR" -deffnm "$deffnm" "${MDRUN_FLAGS[@]}" \
        2>&1 | tee -a "$LOG_FILE"
}

append_csv_row() {
    local r=$1
    local mdlog="$WORKDIR/run${r}.log"
    local perf wall core
    if [[ -f "$mdlog" ]]; then
        perf=$(awk '/^Performance:/ { print $2; exit }' "$mdlog")
        read -r core wall < <(awk '/^[[:space:]]*Time:/ { print $2" "$3; exit }' "$mdlog")
    fi
    printf "%s,%s,%s,%d,%s,%s,%s,%d,%s\n" \
        "$BENCHMARK" "$STACK" "$SLURM_JOB_ID" "$r" \
        "${perf:-NA}" "${wall:-NA}" "${core:-NA}" 8 "${STACK_TOOLCHAIN:-NA}" \
        >> "$CSV_FILE"
}

# ============================================================================
if [[ "$STACK" == "native" ]]; then
    echo "################# NATIVE STACK #################" | tee -a "$LOG_FILE"
    setup_native || { echo "ERROR: setup_native failed" >&2; exit 1; }
else
    echo "################# EESSI STACK ##################" | tee -a "$LOG_FILE"
    setup_eessi  || { echo "ERROR: setup_eessi failed"  >&2; exit 1; }
fi

node_metadata_dump "$META_FILE"

for r in $(seq 1 $NUM_RUNS); do
    if (( r == WARMUP_RUN )); then
        tag="WARM-UP (discarded)"
    else
        tag="recorded $((r - WARMUP_RUN))/$((NUM_RUNS - 1))"
    fi
    {
        echo
        echo "################################################################"
        echo "# [$STACK] $BENCHMARK  run $r/$NUM_RUNS  $tag  @ $(date -Iseconds)"
        echo "################################################################"
    } | tee -a "$LOG_FILE"

    t0=$SECONDS
    if [[ "$STACK" == "native" ]]; then
        run_mdrun_native "$r"
    else
        run_mdrun_eessi  "$r"
    fi
    echo "  [run $r] elapsed $((SECONDS - t0))s" | tee -a "$LOG_FILE"

    if (( r != WARMUP_RUN )); then
        append_csv_row "$r"
    fi
done

echo
echo "================================================================" | tee -a "$LOG_FILE"
echo "ALL RUNS COMPLETE"                                                 | tee -a "$LOG_FILE"
echo "  CSV : $CSV_FILE"                                                 | tee -a "$LOG_FILE"
echo "  Log : $LOG_FILE"                                                 | tee -a "$LOG_FILE"
echo "  Meta: $META_FILE"                                                | tee -a "$LOG_FILE"
echo "  Work: $WORKDIR"                                                  | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"

# Rename SLURM's .out (written under --job-name) to include the stack.
# SLURM has the fd open; an inode-level rename is transparent.
OLD_OUT="${RESULT_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
NEW_OUT="${RESULT_DIR}/${FILE_BASE}_${SLURM_JOB_ID}.out"
[[ -f "$OLD_OUT" && "$OLD_OUT" != "$NEW_OUT" ]] && mv "$OLD_OUT" "$NEW_OUT" 2>/dev/null || true
