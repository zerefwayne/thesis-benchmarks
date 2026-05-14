#!/bin/bash
#SBATCH --job-name=osu_bw_internode
#SBATCH --account=project_462000226
#SBATCH --partition=dev-g
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --cpus-per-task=7
#SBATCH --time=01:30:00
#SBATCH --output=results/%x_%j.out
#SBATCH --constraint=eessi
#
# osu_bw_internode.sh — A1: inter-node bandwidth, ONE stack per job.
#
# Submit twice — once per stack — to get stack-tagged outputs:
#   sbatch osu_bw_internode.sh native        # both SDMA values
#   sbatch osu_bw_internode.sh eessi
#   sbatch osu_bw_internode.sh native 0      # only SDMA=0
#   sbatch osu_bw_internode.sh eessi  1      # only SDMA=1
#
# Pairs:
#   intra_pkg_OAM0_ref     (NODE_A:GCD0, NODE_A:GCD1) — 4-link xGMI ceiling
#   inter_node_GCD0_GCD0   (NODE_A:GCD0, NODE_B:GCD0) — naive default
#   inter_node_GCD7_GCD7   (NODE_A:GCD7, NODE_B:GCD7) — NIC-adjacent best case
#
# Each invocation bounded by `timeout` so EESSI inter-node hangs (likely on
# TCP fallback at large sizes) cannot eat the whole job's walltime.

STACK="${1:-}"
case "$STACK" in
    native|eessi) ;;
    *) echo "ERROR: usage: sbatch $0 <native|eessi> [SDMA_LIST]  (got: '$STACK')" >&2; exit 2 ;;
esac
SDMA_LIST="${2:-0 1}"

source common.sh
source topology.sh

BASE_NAME="${SLURM_JOB_NAME:-osu_bw_internode}"
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

echo "stack,sdma_enabled,pair_label,node_a,node_b,gcd_a,gcd_b,hop_class,num_links,run,size_bytes,bandwidth_MBps" \
    > "$CSV_FILE"

NUM_RUNS=6                # 1 warm-up + 5 recorded
WARMUP_RUN=1
RUN_TIMEOUT=300
OSU_FLAGS="-m 8:268435456 -i 100 -d rocm D D"

WRAPPER="${RESULT_DIR}/wrap_internode_${SLURM_JOB_ID}.sh"
cat > "$WRAPPER" <<'EOF'
#!/bin/bash
GCD_A=$1; GCD_B=$2; shift 2
rank=${SLURM_PROCID:-${OMPI_COMM_WORLD_RANK:-${PMIX_RANK:-0}}}
case "$rank" in
    0) export ROCR_VISIBLE_DEVICES=$GCD_A ;;
    *) export ROCR_VISIBLE_DEVICES=$GCD_B ;;
esac
exec "$@"
EOF
chmod +x "$WRAPPER"

parse_pt2pt() {
    local stack=$1 sdma=$2 lab=$3 na=$4 nb=$5 g0=$6 g1=$7 hop=$8 nl=$9 run=${10}
    awk -v st="$stack" -v sdma="$sdma" -v lab="$lab" -v na="$na" -v nb="$nb" \
        -v g0="$g0" -v g1="$g1" -v hop="$hop" -v nl="$nl" -v run="$run" '
        /^#/ { next }
        NF == 2 && $1 ~ /^[0-9]+$/ {
            printf "%s,%d,%s,%s,%s,%d,%d,%s,%s,%d,%d,%.2f\n",
                   st, sdma, lab, na, nb, g0, g1, hop, nl, run, $1, $2
        }'
}

run_intra() {
    local sdma=$1 g0=$2 g1=$3 label=$4 nlinks=$5 bin=$6
    {
        echo
        echo "################################################################"
        echo "# [$STACK sdma=$sdma] $label  intra_node GCD($g0,$g1) on $NODE_A"
        echo "################################################################"
    } | tee -a "$LOG_FILE"

    for run in $(seq 1 $NUM_RUNS); do
        local t0=$SECONDS raw
        local tag=$( (( run == WARMUP_RUN )) && echo "warm-up" || echo "recording" )
        echo "  [run $run/$NUM_RUNS] $tag - $(date '+%H:%M:%S')" | tee -a "$LOG_FILE"

        if [[ "$STACK" == "native" ]]; then
            raw=$(timeout $RUN_TIMEOUT srun --nodes=1 --nodelist=$NODE_A \
                --ntasks=2 --ntasks-per-node=2 \
                bash -c "export ROCR_VISIBLE_DEVICES=$g0,$g1; exec $bin $OSU_FLAGS" \
                2>>"$LOG_FILE")
        else
            raw=$(timeout $RUN_TIMEOUT bash -c \
                "ROCR_VISIBLE_DEVICES=$g0,$g1 mpirun -n 2 --host $NODE_A:2 $bin $OSU_FLAGS" \
                2>>"$LOG_FILE")
        fi

        echo "  [run $run/$NUM_RUNS] done in $((SECONDS - t0))s" | tee -a "$LOG_FILE"
        echo "$raw" >> "$LOG_FILE"
        if (( run != WARMUP_RUN )); then
            echo "$raw" | parse_pt2pt "$STACK" "$sdma" "$label" "$NODE_A" "$NODE_A" \
                "$g0" "$g1" "intra_node" "$nlinks" "$run" >> "$CSV_FILE"
        fi
    done
}

run_inter() {
    local sdma=$1 ga=$2 gb=$3 label=$4 bin=$5
    {
        echo
        echo "################################################################"
        echo "# [$STACK sdma=$sdma] $label  inter_node ($NODE_A:GCD$ga, $NODE_B:GCD$gb)"
        echo "################################################################"
    } | tee -a "$LOG_FILE"

    for run in $(seq 1 $NUM_RUNS); do
        local t0=$SECONDS raw
        local tag=$( (( run == WARMUP_RUN )) && echo "warm-up" || echo "recording" )
        echo "  [run $run/$NUM_RUNS] $tag - $(date '+%H:%M:%S')" | tee -a "$LOG_FILE"

        if [[ "$STACK" == "native" ]]; then
            raw=$(timeout $RUN_TIMEOUT srun --nodes=2 --ntasks=2 --ntasks-per-node=1 \
                "$WRAPPER" "$ga" "$gb" "$bin" $OSU_FLAGS 2>>"$LOG_FILE")
        else
            raw=$(timeout $RUN_TIMEOUT mpirun -n 2 --host "$NODE_A:1,$NODE_B:1" \
                --map-by ppr:1:node "$WRAPPER" "$ga" "$gb" "$bin" $OSU_FLAGS \
                2>>"$LOG_FILE")
        fi

        echo "  [run $run/$NUM_RUNS] done in $((SECONDS - t0))s" | tee -a "$LOG_FILE"
        echo "$raw" >> "$LOG_FILE"
        if (( run != WARMUP_RUN )); then
            echo "$raw" | parse_pt2pt "$STACK" "$sdma" "$label" "$NODE_A" "$NODE_B" \
                "$ga" "$gb" "inter_node" "NA" "$run" >> "$CSV_FILE"
        fi
    done
}

sweep_pairs() {
    local sdma=$1 bin=$2
    for entry in "${A1_INTERNODE_PAIRS[@]}"; do
        read -r ga gb hop label nlinks <<< "$entry"
        if [[ "$hop" == "intra_node" ]]; then
            run_intra "$sdma" "$ga" "$gb" "$label" "$nlinks" "$bin"
        else
            run_inter "$sdma" "$ga" "$gb" "$label" "$bin"
        fi
    done
}

echo "################# $STACK SECTION #################" | tee -a "$LOG_FILE"
if [[ "$STACK" == "native" ]]; then
    setup_native || { echo "ERROR: setup_native failed" | tee -a "$LOG_FILE"; exit 1; }
    BIN="${OSU_NATIVE_PT2PT}/osu_bw"
else
    setup_eessi || { echo "ERROR: setup_eessi failed" | tee -a "$LOG_FILE"; exit 1; }
    BIN="${OSU_PT2PT}/osu_bw"
fi

for sdma in $SDMA_LIST; do
    export HSA_ENABLE_SDMA=$sdma
    echo "--- $STACK / HSA_ENABLE_SDMA=$sdma ---" | tee -a "$LOG_FILE"
    sweep_pairs "$sdma" "$BIN"
done

rm -f "$WRAPPER"
echo
echo "ALL BENCHMARKS COMPLETE  stack=$STACK  CSV=$CSV_FILE" | tee -a "$LOG_FILE"
