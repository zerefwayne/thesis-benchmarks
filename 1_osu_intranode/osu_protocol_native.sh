#!/bin/bash
#SBATCH --job-name=osu_protocol_native
#SBATCH --account=project_462000226
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --exclusive
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=28
#SBATCH --time=00:15:00
#SBATCH --output=results/%x_%j.out
#
# osu_protocol_native.sh — A5: Cray MPICH IPC threshold sweep (native)
# Single pair (0,1), SDMA=0, 6 thresholds × 4 runs (1 warmup + 3 recorded).
#
# Finer sweep across the IPC transition zone — mirrors osu_protocol_eessi.sh.
# Previous sweep {1, DEFAULT, 16777216} showed:
#   - IPC=1 (always IPC) hurts small msgs (~half DEFAULT at 1B-512B)
#   - DEFAULT is clean (no cliff)
#   - IPC=16M was ~2× FASTER than DEFAULT at 1K-4K (host-staged wins there)
# So the sweet spot for native sits between DEFAULT (~8K?) and 16M.
# This run samples 1024..16384 to find where Cray IPC starts helping/hurting
# the 1K-16K transition zone. IPC=1 (known-bad) and 16M (already tested)
# are omitted.
#
# Native-only — no --constraint=eessi needed.

source common.sh
source topology.sh

CSV_FILE="${RESULT_DIR}/${SLURM_JOB_NAME:-osu_protocol_native}_${SLURM_JOB_ID}.csv"
LOG_FILE="${RESULT_DIR}/${SLURM_JOB_NAME:-osu_protocol_native}_${SLURM_JOB_ID}.log"
META_FILE="${RESULT_DIR}/${SLURM_JOB_NAME:-osu_protocol_native}_${SLURM_JOB_ID}.meta"

node_metadata_dump "$META_FILE"

echo "stack,sdma_enabled,mpich_gpu_ipc_threshold,pair_label,gcd_a,gcd_b,tier,num_links,run,size_bytes,bandwidth_MBps" \
    > "$CSV_FILE"

NUM_RUNS=4
WARMUP_RUN=1
RUN_TIMEOUT=300
OSU_FLAGS="-m 1:67108864 -i 100 -d rocm D D"

G0=0; G1=1; LABEL="intra_pkg_OAM0"
TOPO=$(get_topology $G0 $G1)
TIER="${TOPO%,*}"; NLINKS="${TOPO#*,}"

setup_native || { echo "ERROR: setup_native failed" >&2; exit 1; }
export HSA_ENABLE_SDMA=0

# Sweep matches the EESSI side (osu_protocol_eessi.sh):
#   DEFAULT       — Cray built-in (baseline)
#   1024..16384   — powers of 2 to find the IPC transition sweet spot
for ipc in DEFAULT 1024 2048 4096 8192 16384; do
    if [[ "$ipc" == "DEFAULT" ]]; then
        unset MPICH_GPU_IPC_THRESHOLD
    else
        export MPICH_GPU_IPC_THRESHOLD=$ipc
    fi
    {
        echo
        echo "# [native sdma=0 MPICH_GPU_IPC_THRESHOLD=$ipc] $LABEL  GCD($G0,$G1)"
    } | tee -a "$LOG_FILE"

    for run in $(seq 1 $NUM_RUNS); do
        t0=$SECONDS
        if (( run == WARMUP_RUN )); then
            echo "  [run $run/$NUM_RUNS] warm-up — $(date '+%H:%M:%S')" | tee -a "$LOG_FILE"
            timeout $RUN_TIMEOUT srun --ntasks=2 --ntasks-per-node=2 \
                bash -c "export ROCR_VISIBLE_DEVICES=$G0,$G1; exec ${OSU_NATIVE_PT2PT}/osu_bw $OSU_FLAGS" \
                > /dev/null 2>&1
            echo "  [run $run/$NUM_RUNS] warm-up done in $((SECONDS - t0))s" | tee -a "$LOG_FILE"
            continue
        fi
        echo "  [run $run/$NUM_RUNS] recording — $(date '+%H:%M:%S')" | tee -a "$LOG_FILE"
        raw=$(timeout $RUN_TIMEOUT srun --ntasks=2 --ntasks-per-node=2 \
            bash -c "export ROCR_VISIBLE_DEVICES=$G0,$G1; exec ${OSU_NATIVE_PT2PT}/osu_bw $OSU_FLAGS" \
            2>>"$LOG_FILE")
        echo "  [run $run/$NUM_RUNS] done in $((SECONDS - t0))s" | tee -a "$LOG_FILE"
        echo "$raw" >> "$LOG_FILE"
        echo "$raw" | awk \
            -v st="native" -v sdma=0 -v ipc="$ipc" -v lab="$LABEL" \
            -v g0="$G0" -v g1="$G1" -v tier="$TIER" -v nl="$NLINKS" -v run="$run" '
            /^#/ { next }
            NF == 2 && $1 ~ /^[0-9]+$/ {
                printf "%s,%d,%s,%s,%d,%d,%s,%d,%d,%d,%.2f\n",
                       st, sdma, ipc, lab, g0, g1, tier, nl, run, $1, $2
            }' >> "$CSV_FILE"
    done
done

echo
echo "ALL BENCHMARKS COMPLETE  CSV=$CSV_FILE" | tee -a "$LOG_FILE"
