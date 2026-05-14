#!/bin/bash
#SBATCH --job-name=osu_mbw_mr
#SBATCH --account=project_462000226
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --exclusive
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --time=00:15:00
#SBATCH --output=results/%x_%j.out
#SBATCH --constraint=eessi
#
# osu_mbw_mr.sh — A7: multi-pair concurrent bandwidth
# Reproduces paper's edge-forwarding-index intuition (Sec IV-A).
# One stack per job. Submit:
#   sbatch osu_mbw_mr.sh eessi
#   sbatch osu_mbw_mr.sh native

STACK="${1:?usage: sbatch $0 <eessi|native>}"
if [[ "$STACK" != "eessi" && "$STACK" != "native" ]]; then
    echo "ERROR: stack must be 'eessi' or 'native', got '$STACK'" >&2
    exit 1
fi

source common.sh

FILE_BASE="osu_mbw_mr_${STACK}"
CSV_FILE="${RESULT_DIR}/${FILE_BASE}_${SLURM_JOB_ID}.csv"
LOG_FILE="${RESULT_DIR}/${FILE_BASE}_${SLURM_JOB_ID}.log"
META_FILE="${RESULT_DIR}/${FILE_BASE}_${SLURM_JOB_ID}.meta"

node_metadata_dump "$META_FILE"

echo "stack,sdma_enabled,config_label,pairing_desc,num_pairs,run,size_bytes,bandwidth_MBps,msg_rate_Mps" \
    > "$CSV_FILE"

NUM_RUNS=6
WARMUP_RUN=1
RUN_TIMEOUT=300
NUM_PAIRS=4

OSU_FLAGS="-d rocm D D"

# Three ROCR_VISIBLE_DEVICES orderings — see 3_osu_eessi_thesis/osu_mbw_mr.sh:107-122
CFG_NAMES=(cfg_routed_split cfg_intra_pkg cfg_mixed_1link)
CFG_DESCS=(
    "(0,4)(1,5)(2,6)(3,7)_all_routed"
    "(0,1)(2,3)(4,5)(6,7)_all_intra_pkg_4link"
    "(0,2)(1,3)(4,6)(5,7)_all_inter_pkg_1link"
)
CFG_ROCR=(
    "0,1,2,3,4,5,6,7"
    "0,2,4,6,1,3,5,7"
    "0,1,4,5,2,3,6,7"
)

# Wrapper for native 8-element ROCR list
WRAPPER_SCRIPT="${RESULT_DIR}/wrap_gpu_${SLURM_JOB_ID}.sh"
cat > "$WRAPPER_SCRIPT" <<'EOF'
#!/bin/bash
export ROCR_VISIBLE_DEVICES=$1
shift
exec "$@"
EOF
chmod +x "$WRAPPER_SCRIPT"

parse_mbw() {
    local stack=$1 sdma=$2 lab=$3 pd=$4 run=$5
    awk -v st="$stack" -v sdma="$sdma" -v lab="$lab" -v pd="$pd" \
        -v np="$NUM_PAIRS" -v run="$run" '
        /^#/ { next }
        NF == 3 && $1 ~ /^[0-9]+$/ {
            printf "%s,%d,%s,%s,%d,%d,%d,%.2f,%.2f\n",
                   st, sdma, lab, pd, np, run, $1, $2, $3
        }'
}

run_native() {
    local sdma=$1 cfg_lab=$2 pd=$3 rocr=$4 bin=$5
    { echo; echo "# [native sdma=$sdma] $cfg_lab  ROCR=$rocr"; } | tee -a "$LOG_FILE"
    for run in $(seq 1 $NUM_RUNS); do
        local t0=$SECONDS
        if (( run == WARMUP_RUN )); then
            echo "  [run $run/$NUM_RUNS] warm-up — $(date '+%H:%M:%S')" | tee -a "$LOG_FILE"
            timeout $RUN_TIMEOUT srun --ntasks=8 \
                "$WRAPPER_SCRIPT" "$rocr" "$bin" $OSU_FLAGS > /dev/null 2>&1
            echo "  [run $run/$NUM_RUNS] warm-up done in $((SECONDS - t0))s" | tee -a "$LOG_FILE"
            continue
        fi
        echo "  [run $run/$NUM_RUNS] recording — $(date '+%H:%M:%S')" | tee -a "$LOG_FILE"
        local raw
        raw=$(timeout $RUN_TIMEOUT srun --ntasks=8 \
            "$WRAPPER_SCRIPT" "$rocr" "$bin" $OSU_FLAGS 2>>"$LOG_FILE")
        echo "  [run $run/$NUM_RUNS] done in $((SECONDS - t0))s" | tee -a "$LOG_FILE"
        echo "$raw" >> "$LOG_FILE"
        echo "$raw" | parse_mbw native "$sdma" "$cfg_lab" "$pd" "$run" >> "$CSV_FILE"
    done
}

run_eessi() {
    local sdma=$1 cfg_lab=$2 pd=$3 rocr=$4 bin=$5
    { echo; echo "# [eessi sdma=$sdma] $cfg_lab  ROCR=$rocr"; } | tee -a "$LOG_FILE"
    for run in $(seq 1 $NUM_RUNS); do
        local t0=$SECONDS
        if (( run == WARMUP_RUN )); then
            echo "  [run $run/$NUM_RUNS] warm-up — $(date '+%H:%M:%S')" | tee -a "$LOG_FILE"
            timeout $RUN_TIMEOUT bash -c \
                "ROCR_VISIBLE_DEVICES=$rocr mpirun -n 8 $bin $OSU_FLAGS" \
                > /dev/null 2>&1
            echo "  [run $run/$NUM_RUNS] warm-up done in $((SECONDS - t0))s" | tee -a "$LOG_FILE"
            continue
        fi
        echo "  [run $run/$NUM_RUNS] recording — $(date '+%H:%M:%S')" | tee -a "$LOG_FILE"
        local raw
        raw=$(timeout $RUN_TIMEOUT bash -c \
            "ROCR_VISIBLE_DEVICES=$rocr mpirun -n 8 $bin $OSU_FLAGS" \
            2>>"$LOG_FILE")
        echo "  [run $run/$NUM_RUNS] done in $((SECONDS - t0))s" | tee -a "$LOG_FILE"
        echo "$raw" >> "$LOG_FILE"
        echo "$raw" | parse_mbw eessi "$sdma" "$cfg_lab" "$pd" "$run" >> "$CSV_FILE"
    done
}

# ============================================================================
export HSA_ENABLE_SDMA=0
if [[ "$STACK" == "native" ]]; then
    echo "################# NATIVE STACK (HSA_ENABLE_SDMA=0) #################" | tee -a "$LOG_FILE"
    setup_native || { echo "ERROR: setup_native failed" >&2; exit 1; }
    for i in 0 1 2; do
        run_native 0 \
            "${CFG_NAMES[$i]}" "${CFG_DESCS[$i]}" "${CFG_ROCR[$i]}" \
            "${OSU_NATIVE_PT2PT}/osu_mbw_mr"
    done
else
    echo "################# EESSI STACK (HSA_ENABLE_SDMA=0) ##################" | tee -a "$LOG_FILE"
    setup_eessi || { echo "ERROR: setup_eessi failed" >&2; exit 1; }
    for i in 0 1 2; do
        run_eessi 0 \
            "${CFG_NAMES[$i]}" "${CFG_DESCS[$i]}" "${CFG_ROCR[$i]}" \
            "${OSU_PT2PT}/osu_mbw_mr"
    done
fi

rm -f "$WRAPPER_SCRIPT"
echo
echo "ALL BENCHMARKS COMPLETE  CSV=$CSV_FILE" | tee -a "$LOG_FILE"

OLD_OUT="${RESULT_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
NEW_OUT="${RESULT_DIR}/${FILE_BASE}_${SLURM_JOB_ID}.out"
[[ -f "$OLD_OUT" && "$OLD_OUT" != "$NEW_OUT" ]] && mv "$OLD_OUT" "$NEW_OUT" 2>/dev/null || true
