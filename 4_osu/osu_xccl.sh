#!/bin/bash
#SBATCH --job-name=osu_xccl_eessi
#SBATCH --account=project_462000226
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --exclusive
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --time=00:15:00
#SBATCH --output=results/%x_%j.out
#SBATCH --constraint=eessi
#
# osu_xccl.sh — A6: RCCL direct via OSU XCCL (EESSI only, N=8)
# Compares the RCCL path to the GPU-Aware MPI path measured in
# osu_collectives.sh. Paper arXiv:2408.14090v2 Sec IV-B shows *CCL beats
# GPU-Aware MPI on large transfers but loses on small ones on LUMI.

source common.sh

CSV_FILE="${RESULT_DIR}/${SLURM_JOB_NAME:-osu_xccl_eessi}_${SLURM_JOB_ID}.csv"
LOG_FILE="${RESULT_DIR}/${SLURM_JOB_NAME:-osu_xccl_eessi}_${SLURM_JOB_ID}.log"
META_FILE="${RESULT_DIR}/${SLURM_JOB_NAME:-osu_xccl_eessi}_${SLURM_JOB_ID}.meta"

node_metadata_dump "$META_FILE"

echo "stack,sdma_enabled,benchmark,num_gcds,run,size_bytes,latency_us" > "$CSV_FILE"

NUM_RUNS=6
WARMUP_RUN=1
RUN_TIMEOUT=600

N_GCDS=8
DEVICES="0,1,2,3,4,5,6,7"

parse_coll() {
    local sdma=$1 bench=$2 run=$3
    awk -v sdma="$sdma" -v bench="$bench" -v ng="$N_GCDS" -v run="$run" '
        /^#/ { next }
        NF >= 2 && $1 ~ /^[0-9]+$/ {
            printf "eessi_xccl,%d,%s,%d,%d,%d,%.2f\n", sdma, bench, ng, run, $1, $2
        }'
}

run_xccl() {
    local sdma=$1 benchmark=$2 bin=$3
    { echo; echo "# [eessi_xccl sdma=$sdma] $benchmark  N=$N_GCDS"; } | tee -a "$LOG_FILE"
    for run in $(seq 1 $NUM_RUNS); do
        local t0=$SECONDS
        if (( run == WARMUP_RUN )); then
            echo "  [run $run/$NUM_RUNS] warm-up — $(date '+%H:%M:%S')" | tee -a "$LOG_FILE"
            timeout $RUN_TIMEOUT bash -c \
                "ROCR_VISIBLE_DEVICES=$DEVICES mpirun -n $N_GCDS $bin" \
                > /dev/null 2>&1
            echo "  [run $run/$NUM_RUNS] warm-up done in $((SECONDS - t0))s" | tee -a "$LOG_FILE"
            continue
        fi
        echo "  [run $run/$NUM_RUNS] recording — $(date '+%H:%M:%S')" | tee -a "$LOG_FILE"
        local raw
        raw=$(timeout $RUN_TIMEOUT bash -c \
            "ROCR_VISIBLE_DEVICES=$DEVICES mpirun -n $N_GCDS $bin" \
            2>>"$LOG_FILE")
        echo "  [run $run/$NUM_RUNS] done in $((SECONDS - t0))s" | tee -a "$LOG_FILE"
        echo "$raw" >> "$LOG_FILE"
        echo "$raw" | parse_coll "$sdma" "$benchmark" "$run" >> "$CSV_FILE"
    done
}

XCCL_BINS=(osu_xccl_allreduce osu_xccl_alltoall osu_xccl_broadcast osu_xccl_allgather)

setup_eessi || { echo "ERROR: setup_eessi failed" >&2; exit 1; }

# XCCL bypasses MPI collectives; call RCCL directly. Don't route through UCC.
# Also: XCCL binaries accept -d for accelerator type implicitly via build flags;
# do not pass it (caused (null) errors with --full in earlier attempts).

export HSA_ENABLE_SDMA=0
echo "--- EESSI_XCCL / HSA_ENABLE_SDMA=0 ---" | tee -a "$LOG_FILE"
for binary in "${XCCL_BINS[@]}"; do
    if [[ -x "${OSU_XCCL_DIR}/${binary}" ]]; then
        run_xccl 0 "$binary" "${OSU_XCCL_DIR}/${binary}"
    else
        echo "  [skip] $binary not in $OSU_XCCL_DIR" | tee -a "$LOG_FILE"
    fi
done

echo
echo "ALL BENCHMARKS COMPLETE  CSV=$CSV_FILE" | tee -a "$LOG_FILE"
