#!/bin/bash
#SBATCH --job-name=osu_collectives_internode
#SBATCH --account=project_462000226
#SBATCH --partition=dev-g
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --cpus-per-task=7
#SBATCH --time=02:30:00
#SBATCH --output=results/%x_%j.out
#SBATCH --constraint=eessi
#
# osu_collectives_internode.sh — A2: collectives at N=8 (intra) and N=16 (inter),
# ONE stack per job.
#
# Submit twice:
#   sbatch osu_collectives_internode.sh native
#   sbatch osu_collectives_internode.sh eessi
#
# Binaries: osu_allreduce, osu_alltoall.

STACK="${1:-}"
case "$STACK" in
    native|eessi) ;;
    *) echo "ERROR: usage: sbatch $0 <native|eessi> [SDMA_LIST]  (got: '$STACK')" >&2; exit 2 ;;
esac
SDMA_LIST="${2:-0 1}"

source common.sh
source topology.sh

BASE_NAME="${SLURM_JOB_NAME:-osu_collectives_internode}"
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

echo "stack,sdma_enabled,benchmark,num_nodes,num_gcds,run,size_bytes,latency_us" > "$CSV_FILE"

NUM_RUNS=6
WARMUP_RUN=1
RUN_TIMEOUT=600

WRAPPER_INTRA="${RESULT_DIR}/wrap_intra_${SLURM_JOB_ID}.sh"
cat > "$WRAPPER_INTRA" <<'EOF'
#!/bin/bash
export ROCR_VISIBLE_DEVICES=$1
shift
exec "$@"
EOF
chmod +x "$WRAPPER_INTRA"

WRAPPER_INTER="${RESULT_DIR}/wrap_inter_${SLURM_JOB_ID}.sh"
cat > "$WRAPPER_INTER" <<'EOF'
#!/bin/bash
rank=${SLURM_PROCID:-${OMPI_COMM_WORLD_RANK:-${PMIX_RANK:-0}}}
export ROCR_VISIBLE_DEVICES=$((rank % 8))
exec "$@"
EOF
chmod +x "$WRAPPER_INTER"

parse_coll() {
    local stack=$1 sdma=$2 bench=$3 nn=$4 ng=$5 run=$6
    awk -v st="$stack" -v sdma="$sdma" -v bench="$bench" \
        -v nn="$nn" -v ng="$ng" -v run="$run" '
        /^#/ { next }
        NF >= 2 && $1 ~ /^[0-9]+$/ {
            printf "%s,%d,%s,%d,%d,%d,%d,%.2f\n", st, sdma, bench, nn, ng, run, $1, $2
        }'
}

run_collective() {
    local sdma=$1 nn=$2 ng=$3 hop=$4 bench=$5 bin=$6

    {
        echo
        echo "################################################################"
        echo "# [$STACK sdma=$sdma] $bench  nodes=$nn N=$ng $hop"
        echo "################################################################"
    } | tee -a "$LOG_FILE"

    for run in $(seq 1 $NUM_RUNS); do
        local t0=$SECONDS raw
        local tag=$( (( run == WARMUP_RUN )) && echo "warm-up" || echo "recording" )
        echo "  [run $run/$NUM_RUNS] $tag - $(date '+%H:%M:%S')" | tee -a "$LOG_FILE"

        if [[ "$STACK" == "native" ]]; then
            if [[ "$hop" == "intra_node" ]]; then
                raw=$(timeout $RUN_TIMEOUT \
                    srun --nodes=1 --nodelist=$NODE_A --ntasks=$ng --ntasks-per-node=$ng \
                        "$WRAPPER_INTRA" "0,1,2,3,4,5,6,7" "$bin" -d rocm \
                    2>>"$LOG_FILE")
            else
                raw=$(timeout $RUN_TIMEOUT \
                    srun --nodes=2 --ntasks=$ng --ntasks-per-node=8 \
                        "$WRAPPER_INTER" "$bin" -d rocm \
                    2>>"$LOG_FILE")
            fi
        else
            if [[ "$hop" == "intra_node" ]]; then
                raw=$(timeout $RUN_TIMEOUT bash -c \
                    "ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 mpirun -n $ng --host $NODE_A:$ng $bin -d rocm" \
                    2>>"$LOG_FILE")
            else
                raw=$(timeout $RUN_TIMEOUT \
                    mpirun -n $ng --host "$NODE_A:8,$NODE_B:8" --map-by ppr:8:node \
                        "$WRAPPER_INTER" "$bin" -d rocm \
                    2>>"$LOG_FILE")
            fi
        fi

        echo "  [run $run/$NUM_RUNS] done in $((SECONDS - t0))s" | tee -a "$LOG_FILE"
        echo "$raw" >> "$LOG_FILE"
        if (( run != WARMUP_RUN )); then
            echo "$raw" | parse_coll "$STACK" "$sdma" "$bench" "$nn" "$ng" "$run" \
                >> "$CSV_FILE"
        fi
    done
}

COLLECTIVES=(osu_allreduce osu_alltoall)

echo "################# $STACK SECTION #################" | tee -a "$LOG_FILE"
if [[ "$STACK" == "native" ]]; then
    setup_native || { echo "ERROR: setup_native failed" | tee -a "$LOG_FILE"; exit 1; }
    COLL_DIR="$OSU_NATIVE_COLL"
else
    setup_eessi || { echo "ERROR: setup_eessi failed" | tee -a "$LOG_FILE"; exit 1; }
    COLL_DIR="$OSU_COLL"
    # UCC tuning — kept identical to 4_osu intra-node baseline.
    export OMPI_MCA_opal_common_ucx_devices=any
    export OMPI_MCA_coll_ucc_enable=1
    export OMPI_MCA_coll_ucc_priority=100
    export OMPI_MCA_accelerator=rocm
    export UCC_TLS=ucp,self
    export UCC_CL_BASIC_TLS=ucp,self
fi

for sdma in $SDMA_LIST; do
    export HSA_ENABLE_SDMA=$sdma
    echo "--- $STACK / HSA_ENABLE_SDMA=$sdma ---" | tee -a "$LOG_FILE"
    for entry in "${A2_RANK_MODES[@]}"; do
        read -r nn ng hop <<< "$entry"
        for bench in "${COLLECTIVES[@]}"; do
            bin="${COLL_DIR}/${bench}"
            if [[ -x "$bin" ]]; then
                run_collective "$sdma" "$nn" "$ng" "$hop" "$bench" "$bin"
            else
                echo "  [skip] $bench missing at $bin" | tee -a "$LOG_FILE"
            fi
        done
    done
done

rm -f "$WRAPPER_INTRA" "$WRAPPER_INTER"
echo
echo "ALL BENCHMARKS COMPLETE  stack=$STACK  CSV=$CSV_FILE" | tee -a "$LOG_FILE"
