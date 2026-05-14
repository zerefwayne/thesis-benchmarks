#!/bin/bash
#SBATCH --job-name=osu_saturation_internode
#SBATCH --account=project_462000226
#SBATCH --partition=dev-g
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --cpus-per-task=7
#SBATCH --time=01:00:00
#SBATCH --output=results/%x_%j.out
#SBATCH --constraint=eessi
#
# osu_saturation_internode.sh — A3: 1-flow vs 2-flow concurrent inter-node BW,
# ONE stack per job.
#
# Submit twice:
#   sbatch osu_saturation_internode.sh native
#   sbatch osu_saturation_internode.sh eessi
#
# Configs:
#   1flow_GCD0: (NODE_A:GCD0, NODE_B:GCD0) via osu_bw, -n 2
#   2flow_GCD01: 2 inter-node pairs simultaneously via osu_mbw_mr, -n 4
#                ranks 0,1 on NODE_A (GCDs 0,1); ranks 2,3 on NODE_B (GCDs 0,1)
#                osu_mbw_mr pairs (i, i+N/2) -> (0,2) and (1,3), both inter-node

STACK="${1:-}"
case "$STACK" in
    native|eessi) ;;
    *) echo "ERROR: usage: sbatch $0 <native|eessi> [SDMA_LIST]  (got: '$STACK')" >&2; exit 2 ;;
esac
SDMA_LIST="${2:-0 1}"

source common.sh
source topology.sh

BASE_NAME="${SLURM_JOB_NAME:-osu_saturation_internode}"
TAGGED_NAME="${BASE_NAME}_${STACK}"
scontrol update job=$SLURM_JOB_ID JobName="$TAGGED_NAME" 2>/dev/null || true

CSV_FILE="${RESULT_DIR}/${TAGGED_NAME}_${SLURM_JOB_ID}.csv"
LOG_FILE="${RESULT_DIR}/${TAGGED_NAME}_${SLURM_JOB_ID}.log"
META_FILE="${RESULT_DIR}/${TAGGED_NAME}_${SLURM_JOB_ID}.meta"
ORIG_OUT="${RESULT_DIR}/${BASE_NAME}_${SLURM_JOB_ID}.out"
TAGGED_OUT="${RESULT_DIR}/${TAGGED_NAME}_${SLURM_JOB_ID}.out"
trap "cp -f '$ORIG_OUT' '$TAGGED_OUT' 2>/dev/null || true" EXIT

node_metadata_dump "$META_FILE"

NODES=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
if [[ ${#NODES[@]} -lt 2 ]]; then
    echo "ERROR: need 2 nodes, got ${#NODES[@]} (${NODES[*]})" | tee -a "$LOG_FILE"
    exit 1
fi
NODE_A=${NODES[0]}; NODE_B=${NODES[1]}
echo "STACK=$STACK  NODE_A=$NODE_A  NODE_B=$NODE_B  SDMA_LIST='$SDMA_LIST'" \
    | tee -a "$LOG_FILE"

echo "stack,sdma_enabled,config_label,num_pairs,run,size_bytes,bandwidth_MBps,msg_rate_Mps" \
    > "$CSV_FILE"

NUM_RUNS=6
WARMUP_RUN=1
RUN_TIMEOUT=300
OSU_FLAGS_BW="-m 8:268435456 -i 100 -d rocm D D"
OSU_FLAGS_MBW="-d rocm D D"

WRAPPER_1FLOW="${RESULT_DIR}/wrap_1flow_${SLURM_JOB_ID}.sh"
cat > "$WRAPPER_1FLOW" <<'EOF'
#!/bin/bash
rank=${SLURM_PROCID:-${OMPI_COMM_WORLD_RANK:-${PMIX_RANK:-0}}}
export ROCR_VISIBLE_DEVICES=0
exec "$@"
EOF
chmod +x "$WRAPPER_1FLOW"

WRAPPER_2FLOW="${RESULT_DIR}/wrap_2flow_${SLURM_JOB_ID}.sh"
cat > "$WRAPPER_2FLOW" <<'EOF'
#!/bin/bash
rank=${SLURM_PROCID:-${OMPI_COMM_WORLD_RANK:-${PMIX_RANK:-0}}}
export ROCR_VISIBLE_DEVICES=$((rank % 2))
exec "$@"
EOF
chmod +x "$WRAPPER_2FLOW"

parse_bw_1flow() {
    local stack=$1 sdma=$2 lab=$3 np=$4 run=$5
    awk -v st="$stack" -v sdma="$sdma" -v lab="$lab" -v np="$np" -v run="$run" '
        /^#/ { next }
        NF == 2 && $1 ~ /^[0-9]+$/ {
            printf "%s,%d,%s,%d,%d,%d,%.2f,NA\n", st, sdma, lab, np, run, $1, $2
        }'
}

parse_mbw_2flow() {
    local stack=$1 sdma=$2 lab=$3 np=$4 run=$5
    awk -v st="$stack" -v sdma="$sdma" -v lab="$lab" -v np="$np" -v run="$run" '
        /^#/ { next }
        NF == 3 && $1 ~ /^[0-9]+$/ {
            printf "%s,%d,%s,%d,%d,%d,%.2f,%.2f\n", st, sdma, lab, np, run, $1, $2, $3
        }'
}

run_1flow() {
    local sdma=$1 bin=$2
    {
        echo
        echo "################################################################"
        echo "# [$STACK sdma=$sdma] 1flow_GCD0  ($NODE_A:GCD0, $NODE_B:GCD0) osu_bw"
        echo "################################################################"
    } | tee -a "$LOG_FILE"

    for run in $(seq 1 $NUM_RUNS); do
        local t0=$SECONDS raw
        local tag=$( (( run == WARMUP_RUN )) && echo "warm-up" || echo "recording" )
        echo "  [run $run/$NUM_RUNS] $tag - $(date '+%H:%M:%S')" | tee -a "$LOG_FILE"

        if [[ "$STACK" == "native" ]]; then
            raw=$(timeout $RUN_TIMEOUT srun --nodes=2 --ntasks=2 --ntasks-per-node=1 \
                "$WRAPPER_1FLOW" "$bin" $OSU_FLAGS_BW 2>>"$LOG_FILE")
        else
            raw=$(timeout $RUN_TIMEOUT mpirun -n 2 --host "$NODE_A:1,$NODE_B:1" \
                --map-by ppr:1:node "$WRAPPER_1FLOW" "$bin" $OSU_FLAGS_BW \
                2>>"$LOG_FILE")
        fi

        echo "  [run $run/$NUM_RUNS] done in $((SECONDS - t0))s" | tee -a "$LOG_FILE"
        echo "$raw" >> "$LOG_FILE"
        if (( run != WARMUP_RUN )); then
            echo "$raw" | parse_bw_1flow "$STACK" "$sdma" "1flow_GCD0" 1 "$run" \
                >> "$CSV_FILE"
        fi
    done
}

run_2flow() {
    local sdma=$1 bin=$2
    {
        echo
        echo "################################################################"
        echo "# [$STACK sdma=$sdma] $A3_SAT_MULTI_LABEL  osu_mbw_mr 2 flows"
        echo "################################################################"
    } | tee -a "$LOG_FILE"

    for run in $(seq 1 $NUM_RUNS); do
        local t0=$SECONDS raw
        local tag=$( (( run == WARMUP_RUN )) && echo "warm-up" || echo "recording" )
        echo "  [run $run/$NUM_RUNS] $tag - $(date '+%H:%M:%S')" | tee -a "$LOG_FILE"

        if [[ "$STACK" == "native" ]]; then
            raw=$(timeout $RUN_TIMEOUT srun --nodes=2 --ntasks=4 --ntasks-per-node=2 \
                "$WRAPPER_2FLOW" "$bin" $OSU_FLAGS_MBW 2>>"$LOG_FILE")
        else
            raw=$(timeout $RUN_TIMEOUT mpirun -n 4 --host "$NODE_A:2,$NODE_B:2" \
                --map-by ppr:2:node "$WRAPPER_2FLOW" "$bin" $OSU_FLAGS_MBW \
                2>>"$LOG_FILE")
        fi

        echo "  [run $run/$NUM_RUNS] done in $((SECONDS - t0))s" | tee -a "$LOG_FILE"
        echo "$raw" >> "$LOG_FILE"
        if (( run != WARMUP_RUN )); then
            echo "$raw" | parse_mbw_2flow "$STACK" "$sdma" "$A3_SAT_MULTI_LABEL" 2 "$run" \
                >> "$CSV_FILE"
        fi
    done
}

echo "################# $STACK SECTION #################" | tee -a "$LOG_FILE"
if [[ "$STACK" == "native" ]]; then
    setup_native || { echo "ERROR: setup_native failed" | tee -a "$LOG_FILE"; exit 1; }
    BW_BIN="${OSU_NATIVE_PT2PT}/osu_bw"
    MBW_BIN="${OSU_NATIVE_PT2PT}/osu_mbw_mr"
else
    setup_eessi || { echo "ERROR: setup_eessi failed" | tee -a "$LOG_FILE"; exit 1; }
    BW_BIN="${OSU_PT2PT}/osu_bw"
    MBW_BIN="${OSU_PT2PT}/osu_mbw_mr"
fi

for sdma in $SDMA_LIST; do
    export HSA_ENABLE_SDMA=$sdma
    echo "--- $STACK / HSA_ENABLE_SDMA=$sdma ---" | tee -a "$LOG_FILE"
    run_1flow "$sdma" "$BW_BIN"
    run_2flow "$sdma" "$MBW_BIN"
done

rm -f "$WRAPPER_1FLOW" "$WRAPPER_2FLOW"
echo
echo "ALL BENCHMARKS COMPLETE  stack=$STACK  CSV=$CSV_FILE" | tee -a "$LOG_FILE"
