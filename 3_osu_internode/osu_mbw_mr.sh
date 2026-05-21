#!/bin/bash
#SBATCH --job-name=osu_mbw_mr
#SBATCH --account=project_462000226
#SBATCH --partition=dev-g
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --cpus-per-task=7
#SBATCH --time=00:30:00
#SBATCH --output=results/%x_%j.out
#SBATCH --constraint=eessi
#
# osu_mbw_mr.sh — A7 multi-pair concurrent bandwidth across 2 nodes.
#
# osu_mbw_mr launches N ranks and pairs them (i, i+N/2). With N=8 and
# 4 ranks per node, the 4 pairs (0,4) (1,5) (2,6) (3,7) all become
# cross-node flows. Three ROCR orderings (defined in topology.sh) select
# how the 4 ranks per node map to GCDs:
#   cfg_nic_local_per_node      — both nodes use NIC-adjacent GCDs (1,3,5,7)
#   cfg_nic_via_xgmi_per_node   — both nodes use xGMI-hop GCDs (0,2,4,6)
#   cfg_mixed                   — alternating local/xgmi per node
#
# Submit twice:
#   sbatch osu_mbw_mr.sh eessi
#   sbatch osu_mbw_mr.sh native

STACK="${1:?usage: sbatch $0 <eessi|native>}"
[[ "$STACK" == "eessi" || "$STACK" == "native" ]] \
    || { echo "ERROR: stack must be 'eessi' or 'native'" >&2; exit 1; }

source common.sh
source topology.sh

FILE_BASE="osu_mbw_mr_${STACK}"
CSV_FILE="${RESULT_DIR}/${FILE_BASE}_${SLURM_JOB_ID}.csv"
LOG_FILE="${RESULT_DIR}/${FILE_BASE}_${SLURM_JOB_ID}.log"
META_FILE="${RESULT_DIR}/${FILE_BASE}_${SLURM_JOB_ID}.meta"

node_metadata_dump "$META_FILE"

NODES=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
[[ ${#NODES[@]} -ge 2 ]] || { echo "ERROR: need 2 nodes" | tee -a "$LOG_FILE"; exit 1; }
NODE_A=${NODES[0]}; NODE_B=${NODES[1]}
echo "STACK=$STACK  NODE_A=$NODE_A  NODE_B=$NODE_B" | tee -a "$LOG_FILE"

echo "stack,sdma_enabled,config_label,pairing_desc,num_pairs,run,size_bytes,bandwidth_MBps,msg_rate_Mps" \
    > "$CSV_FILE"

NUM_RUNS=6; WARMUP_RUN=1; RUN_TIMEOUT=300
NUM_PAIRS=4
OSU_FLAGS="-d rocm D D"

# Wrapper: per-rank ROCR_VISIBLE_DEVICES taken from the comma-separated lists
# for node A and node B. SLURM_LOCALID / OMPI_COMM_WORLD_LOCAL_RANK gives the
# per-node rank index (0..3). The node-name suffix on the wrapper picks the
# correct GCD list.
WRAPPER="${RESULT_DIR}/wrap_mbw_${SLURM_JOB_ID}.sh"
cat > "$WRAPPER" <<'EOF'
#!/bin/bash
LIST_A=$1; LIST_B=$2; shift 2
host=$(hostname -s)
local_rank=${SLURM_LOCALID:-${OMPI_COMM_WORLD_LOCAL_RANK:-0}}
# Pick the right per-node list. The driver script passes NODE_A's hostname
# via the env var WRAP_NODE_A so we can compare against $host.
if [[ "$host" == "$WRAP_NODE_A" ]]; then
    IFS=, read -ra arr <<< "$LIST_A"
else
    IFS=, read -ra arr <<< "$LIST_B"
fi
export ROCR_VISIBLE_DEVICES=${arr[$local_rank]}
exec "$@"
EOF
chmod +x "$WRAPPER"

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

run_cfg() {
    local sdma=$1 cfg_lab=$2 pd=$3 list_a=$4 list_b=$5 bin=$6
    {
        echo
        echo "################################################################"
        echo "# [$STACK sdma=$sdma] $cfg_lab"
        echo "#   $pd"
        echo "#   nodeA($list_a) nodeB($list_b)"
        echo "################################################################"
    } | tee -a "$LOG_FILE"

    for run in $(seq 1 $NUM_RUNS); do
        local t0=$SECONDS tag raw
        tag=$( (( run == WARMUP_RUN )) && echo "warm-up" || echo "recording" )
        echo "  [run $run/$NUM_RUNS] $tag - $(date '+%H:%M:%S')" | tee -a "$LOG_FILE"

        if [[ "$STACK" == "native" ]]; then
            raw=$(timeout $RUN_TIMEOUT \
                env WRAP_NODE_A="$NODE_A" \
                srun --nodes=2 --ntasks=8 --ntasks-per-node=4 \
                "$WRAPPER" "$list_a" "$list_b" "$bin" $OSU_FLAGS 2>>"$LOG_FILE")
        else
            raw=$(timeout $RUN_TIMEOUT \
                env WRAP_NODE_A="$NODE_A" \
                mpirun -n 8 --host "$NODE_A:4,$NODE_B:4" --map-by ppr:4:node \
                -x WRAP_NODE_A \
                "$WRAPPER" "$list_a" "$list_b" "$bin" $OSU_FLAGS 2>>"$LOG_FILE")
        fi
        echo "  [run $run/$NUM_RUNS] done in $((SECONDS - t0))s" | tee -a "$LOG_FILE"
        echo "$raw" >> "$LOG_FILE"
        if (( run != WARMUP_RUN )); then
            echo "$raw" | parse_mbw "$STACK" "$sdma" "$cfg_lab" "$pd" "$run" \
                >> "$CSV_FILE"
        fi
    done
}

# ============================================================================
export HSA_ENABLE_SDMA=0
if [[ "$STACK" == "native" ]]; then
    echo "################# NATIVE STACK (HSA_ENABLE_SDMA=0) #################" | tee -a "$LOG_FILE"
    setup_native || { echo "ERROR: setup_native failed" >&2; exit 1; }
    BIN="${OSU_NATIVE_PT2PT}/osu_mbw_mr"
else
    echo "################# EESSI STACK (HSA_ENABLE_SDMA=0) ##################" | tee -a "$LOG_FILE"
    setup_eessi || { echo "ERROR: setup_eessi failed" >&2; exit 1; }
    BIN="${OSU_PT2PT}/osu_mbw_mr"
fi

for i in 0 1 2; do
    run_cfg 0 \
        "${MBW_CFG_NAMES[$i]}" "${MBW_CFG_DESCS[$i]}" \
        "${MBW_CFG_ROCR_PER_NODE[$i]}" "${MBW_CFG_ROCR_PER_NODE_B[$i]}" \
        "$BIN"
done

rm -f "$WRAPPER"
echo
echo "ALL BENCHMARKS COMPLETE  CSV=$CSV_FILE" | tee -a "$LOG_FILE"

OLD_OUT="${RESULT_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
NEW_OUT="${RESULT_DIR}/${FILE_BASE}_${SLURM_JOB_ID}.out"
[[ -f "$OLD_OUT" && "$OLD_OUT" != "$NEW_OUT" ]] && mv "$OLD_OUT" "$NEW_OUT" 2>/dev/null || true
