#!/bin/bash
#SBATCH --job-name=osu_protocol_nccl
#SBATCH --account=project_462000226
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --exclusive
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --time=00:30:00
#SBATCH --output=results/%x_%j.out
#SBATCH --constraint=eessi
#
# osu_protocol_nccl.sh — A9: NCCL/RCCL tuning sweep (EESSI XCCL path)
# Reproduces the paper's 3 RCCL levers on LUMI (arXiv:2408.14090v2 Sec III-B):
#   NCCL_NCHANNELS_PER_PEER=32  (paper: 3.5x intra-node pt2pt)
#   NCCL_IGNORE_CPU_AFFINITY=1  (paper: 1.6x alltoall, 6x allreduce)
#   NCCL_NET_GDR_LEVEL=3        (paper: 2x alltoall, 3x allreduce)
#
# 5 configs × 2 binaries (xccl_allreduce, xccl_alltoall) × 6 runs.
# SDMA=0 fixed (paper-recommended baseline). EESSI only.

source common.sh

CSV_FILE="${RESULT_DIR}/${SLURM_JOB_NAME:-osu_protocol_nccl}_${SLURM_JOB_ID}.csv"
LOG_FILE="${RESULT_DIR}/${SLURM_JOB_NAME:-osu_protocol_nccl}_${SLURM_JOB_ID}.log"
META_FILE="${RESULT_DIR}/${SLURM_JOB_NAME:-osu_protocol_nccl}_${SLURM_JOB_ID}.meta"

node_metadata_dump "$META_FILE"

echo "stack,sdma_enabled,config_label,nccl_nchannels,nccl_ignore_aff,nccl_net_gdr,benchmark,num_gcds,run,size_bytes,latency_us" \
    > "$CSV_FILE"

NUM_RUNS=6
WARMUP_RUN=1
RUN_TIMEOUT=600

N_GCDS=8
DEVICES="0,1,2,3,4,5,6,7"

# Config matrix: label, NCCL_NCHANNELS_PER_PEER, NCCL_IGNORE_CPU_AFFINITY, NCCL_NET_GDR_LEVEL
CFG_LABELS=(baseline channels32 affinity gdr3 paper_full)
CFG_NCH=(    ""        32          ""        ""    32      )
CFG_AFF=(    ""        ""          1         ""    1       )
CFG_GDR=(    ""        ""          ""        3     3       )

apply_config() {
    local i=$1
    if [[ -n "${CFG_NCH[$i]}" ]]; then export NCCL_NCHANNELS_PER_PEER="${CFG_NCH[$i]}"; else unset NCCL_NCHANNELS_PER_PEER; fi
    if [[ -n "${CFG_AFF[$i]}" ]]; then export NCCL_IGNORE_CPU_AFFINITY="${CFG_AFF[$i]}"; else unset NCCL_IGNORE_CPU_AFFINITY; fi
    if [[ -n "${CFG_GDR[$i]}" ]]; then export NCCL_NET_GDR_LEVEL="${CFG_GDR[$i]}";       else unset NCCL_NET_GDR_LEVEL;       fi
}

run_one() {
    local cfg_lab=$1 nch=$2 aff=$3 gdr=$4 benchmark=$5 bin=$6
    { echo; echo "# [eessi_xccl cfg=$cfg_lab] $benchmark"; } | tee -a "$LOG_FILE"
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
        echo "$raw" | awk \
            -v lab="$cfg_lab" -v nch="${nch:-NA}" -v aff="${aff:-NA}" -v gdr="${gdr:-NA}" \
            -v bench="$benchmark" -v ng="$N_GCDS" -v run="$run" '
            /^#/ { next }
            NF >= 2 && $1 ~ /^[0-9]+$/ {
                printf "eessi_xccl,0,%s,%s,%s,%s,%s,%d,%d,%d,%.2f\n",
                       lab, nch, aff, gdr, bench, ng, run, $1, $2
            }' >> "$CSV_FILE"
    done
}

XCCL_BINS=(osu_xccl_allreduce osu_xccl_alltoall)

setup_eessi || { echo "ERROR: setup_eessi failed" >&2; exit 1; }
export HSA_ENABLE_SDMA=0

for i in 0 1 2 3 4; do
    apply_config $i
    echo "=== CFG ${CFG_LABELS[$i]}: NCH=${CFG_NCH[$i]:-unset} AFF=${CFG_AFF[$i]:-unset} GDR=${CFG_GDR[$i]:-unset} ===" \
        | tee -a "$LOG_FILE"
    for binary in "${XCCL_BINS[@]}"; do
        if [[ -x "${OSU_XCCL_DIR}/${binary}" ]]; then
            run_one "${CFG_LABELS[$i]}" "${CFG_NCH[$i]}" "${CFG_AFF[$i]}" "${CFG_GDR[$i]}" \
                "$binary" "${OSU_XCCL_DIR}/${binary}"
        else
            echo "  [skip] $binary not in $OSU_XCCL_DIR" | tee -a "$LOG_FILE"
        fi
    done
done

echo
echo "ALL BENCHMARKS COMPLETE  CSV=$CSV_FILE" | tee -a "$LOG_FILE"
