#!/bin/bash
#SBATCH --job-name=gromacs_stmv
#SBATCH --account=project_462000226
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --exclusive
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --time=02:30:00
#SBATCH --output=results/%x_%j.out
#SBATCH --constraint=eessi
#
# STMV (~1M atoms). Canonical MI250X-GROMACS comparator
# (Páll et al. CUG'24; AMD ROCm Blog LUMI guide).
# Requires: bash fetch_benchmarks.sh — run once on a login node before this.
# Per-run wall ~12-18 min; 7 runs ~1.5-2 h.
#
# Submit:
#   sbatch benchmark_stmv.sh eessi
#   sbatch benchmark_stmv.sh native

STACK="${1:?usage: sbatch $0 <eessi|native>}"
if [[ "$STACK" != "eessi" && "$STACK" != "native" ]]; then
    echo "ERROR: stack must be 'eessi' or 'native', got '$STACK'" >&2
    exit 1
fi

source common.sh

BENCHMARK="stmv"
FILE_BASE="gromacs_${BENCHMARK}_${STACK}"
CSV_FILE="${RESULT_DIR}/${FILE_BASE}_${SLURM_JOB_ID}.csv"
LOG_FILE="${RESULT_DIR}/${FILE_BASE}_${SLURM_JOB_ID}.log"
META_FILE="${RESULT_DIR}/${FILE_BASE}_${SLURM_JOB_ID}.meta"

NUM_RUNS=7         # 1 warm-up + 6 recorded (matches AMD ROCm Blog LUMI guide).
WARMUP_RUN=1
INPUT_TPR="$(pwd)/GROMACS_Benchmark_Suite/STMV/benchmark.tpr"
WORKDIR="$(pwd)/${RESULT_DIR}/${FILE_BASE}_${SLURM_JOB_ID}_work"
SELECT_GPU="$(pwd)/select_gpu"

if [[ ! -f "$INPUT_TPR" ]]; then
    echo "ERROR: STMV TPR not found at $INPUT_TPR" >&2
    echo "Run: bash fetch_benchmarks.sh   (on a login node)" >&2
    exit 1
fi
mkdir -p "$WORKDIR"

MDRUN_FLAGS=(-nb gpu -pme gpu -bonded gpu
             -nsteps 100000 -resetstep 20000 -noconfout -npme 1)

echo "benchmark,stack,jobid,run,perf_ns_per_day,wall_s,core_s,ntmpi,toolchain" > "$CSV_FILE"

run_mdrun_native() {
    local r=$1
    local deffnm="$WORKDIR/run${r}"
    srun --cpu-bind="$CPU_BIND" "$SELECT_GPU" \
        gmx_mpi mdrun -s "$INPUT_TPR" -deffnm "$deffnm" "${MDRUN_FLAGS[@]}" \
        2>&1 | tee -a "$LOG_FILE"
}

# EESSI: no select_gpu wrapper; CPU pinning via OpenMPI's mapper (one rank
# per L3 cache complex, 7 cores each). See benchmark_crambin.sh for details.
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

OLD_OUT="${RESULT_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
NEW_OUT="${RESULT_DIR}/${FILE_BASE}_${SLURM_JOB_ID}.out"
[[ -f "$OLD_OUT" && "$OLD_OUT" != "$NEW_OUT" ]] && mv "$OLD_OUT" "$NEW_OUT" 2>/dev/null || true
