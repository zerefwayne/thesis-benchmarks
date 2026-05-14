#!/bin/bash
#SBATCH --job-name=osu_put_bw
#SBATCH --account=project_462000226
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --exclusive
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=28
#SBATCH --time=02:30:00
#SBATCH --output=results/%x_%j.out
#SBATCH --constraint=eessi
#
# osu_put_bw.sh — A8a: one-sided RMA put bandwidth (12 pairs)
# MPI_Put path — different transport selection from send/recv.
# One stack per job. Submit:
#   sbatch osu_put_bw.sh eessi
#   sbatch osu_put_bw.sh native

STACK="${1:?usage: sbatch $0 <eessi|native>}"
if [[ "$STACK" != "eessi" && "$STACK" != "native" ]]; then
    echo "ERROR: stack must be 'eessi' or 'native', got '$STACK'" >&2
    exit 1
fi

source common.sh
source topology.sh

FILE_BASE="osu_put_bw_${STACK}"
CSV_FILE="${RESULT_DIR}/${FILE_BASE}_${SLURM_JOB_ID}.csv"
LOG_FILE="${RESULT_DIR}/${FILE_BASE}_${SLURM_JOB_ID}.log"
META_FILE="${RESULT_DIR}/${FILE_BASE}_${SLURM_JOB_ID}.meta"

node_metadata_dump "$META_FILE"

echo "stack,sdma_enabled,pair_label,gcd_a,gcd_b,tier,num_links,run,size_bytes,bandwidth_MBps" > "$CSV_FILE"

NUM_RUNS=6
WARMUP_RUN=1
RUN_TIMEOUT=300
# One-sided OSU usage is "[options]" (no SRC DST). -d rocm is enough to
# place the window on device memory. The pt2pt "D D" positionals here
# would either be parsed as filenames or trigger usage error, hanging mpirun.
OSU_FLAGS="-m 1:67108864 -i 100 -d rocm"

parse_pt2pt() {
    local stack=$1 sdma=$2 lab=$3 g0=$4 g1=$5 tier=$6 nl=$7 run=$8
    awk -v st="$stack" -v sdma="$sdma" -v lab="$lab" \
        -v g0="$g0" -v g1="$g1" -v tier="$tier" -v nl="$nl" -v run="$run" '
        /^#/ { next }
        NF == 2 && $1 ~ /^[0-9]+$/ {
            printf "%s,%d,%s,%d,%d,%s,%d,%d,%d,%.2f\n",
                   st, sdma, lab, g0, g1, tier, nl, run, $1, $2
        }'
}

run_pair_native() {
    local sdma=$1 g0=$2 g1=$3 label=$4 bin=$5
    local topo tier nlinks
    topo=$(get_topology "$g0" "$g1"); tier="${topo%,*}"; nlinks="${topo#*,}"
    { echo; echo "# [native sdma=$sdma] $label  GCD($g0,$g1) tier=$tier"; } | tee -a "$LOG_FILE"
    for run in $(seq 1 $NUM_RUNS); do
        local t0=$SECONDS
        if (( run == WARMUP_RUN )); then
            echo "  [run $run/$NUM_RUNS] warm-up — $(date '+%H:%M:%S')" | tee -a "$LOG_FILE"
            timeout $RUN_TIMEOUT srun --ntasks=2 --ntasks-per-node=2 \
                bash -c "export ROCR_VISIBLE_DEVICES=${g0},${g1}; exec $bin $OSU_FLAGS" \
                > /dev/null 2>&1
            echo "  [run $run/$NUM_RUNS] warm-up done in $((SECONDS - t0))s" | tee -a "$LOG_FILE"
            continue
        fi
        echo "  [run $run/$NUM_RUNS] recording — $(date '+%H:%M:%S')" | tee -a "$LOG_FILE"
        local raw
        raw=$(timeout $RUN_TIMEOUT srun --ntasks=2 --ntasks-per-node=2 \
            bash -c "export ROCR_VISIBLE_DEVICES=${g0},${g1}; exec $bin $OSU_FLAGS" \
            2>>"$LOG_FILE")
        echo "  [run $run/$NUM_RUNS] done in $((SECONDS - t0))s" | tee -a "$LOG_FILE"
        echo "$raw" >> "$LOG_FILE"
        echo "$raw" | parse_pt2pt native "$sdma" "$label" "$g0" "$g1" "$tier" "$nlinks" "$run" >> "$CSV_FILE"
    done
}

run_pair_eessi() {
    local sdma=$1 g0=$2 g1=$3 label=$4 bin=$5
    local topo tier nlinks
    topo=$(get_topology "$g0" "$g1"); tier="${topo%,*}"; nlinks="${topo#*,}"
    { echo; echo "# [eessi sdma=$sdma] $label  GCD($g0,$g1) tier=$tier"; } | tee -a "$LOG_FILE"
    for run in $(seq 1 $NUM_RUNS); do
        local t0=$SECONDS
        if (( run == WARMUP_RUN )); then
            echo "  [run $run/$NUM_RUNS] warm-up — $(date '+%H:%M:%S')" | tee -a "$LOG_FILE"
            timeout $RUN_TIMEOUT bash -c \
                "ROCR_VISIBLE_DEVICES=${g0},${g1} mpirun -n 2 $bin $OSU_FLAGS" \
                > /dev/null 2>&1
            echo "  [run $run/$NUM_RUNS] warm-up done in $((SECONDS - t0))s" | tee -a "$LOG_FILE"
            continue
        fi
        echo "  [run $run/$NUM_RUNS] recording — $(date '+%H:%M:%S')" | tee -a "$LOG_FILE"
        local raw
        raw=$(timeout $RUN_TIMEOUT bash -c \
            "ROCR_VISIBLE_DEVICES=${g0},${g1} mpirun -n 2 $bin $OSU_FLAGS" \
            2>>"$LOG_FILE")
        echo "  [run $run/$NUM_RUNS] done in $((SECONDS - t0))s" | tee -a "$LOG_FILE"
        echo "$raw" >> "$LOG_FILE"
        echo "$raw" | parse_pt2pt eessi "$sdma" "$label" "$g0" "$g1" "$tier" "$nlinks" "$run" >> "$CSV_FILE"
    done
}

# ============================================================================
export HSA_ENABLE_SDMA=0
if [[ "$STACK" == "native" ]]; then
    echo "################# NATIVE STACK (HSA_ENABLE_SDMA=0) #################" | tee -a "$LOG_FILE"
    setup_native || { echo "ERROR: setup_native failed" >&2; exit 1; }
    for entry in "${A1_PAIRS[@]}"; do
        read -r g0 g1 label <<< "$entry"
        run_pair_native 0 "$g0" "$g1" "$label" "${OSU_NATIVE_ONESIDED}/osu_put_bw"
    done
else
    echo "################# EESSI STACK (HSA_ENABLE_SDMA=0) ##################" | tee -a "$LOG_FILE"
    setup_eessi || { echo "ERROR: setup_eessi failed" >&2; exit 1; }
    for entry in "${A1_PAIRS[@]}"; do
        read -r g0 g1 label <<< "$entry"
        run_pair_eessi 0 "$g0" "$g1" "$label" "${OSU_ONESIDED}/osu_put_bw"
    done
fi

echo
echo "ALL BENCHMARKS COMPLETE  CSV=$CSV_FILE" | tee -a "$LOG_FILE"

OLD_OUT="${RESULT_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
NEW_OUT="${RESULT_DIR}/${FILE_BASE}_${SLURM_JOB_ID}.out"
[[ -f "$OLD_OUT" && "$OLD_OUT" != "$NEW_OUT" ]] && mv "$OLD_OUT" "$NEW_OUT" 2>/dev/null || true
