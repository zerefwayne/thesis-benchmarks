#!/bin/bash
#SBATCH --job-name=osu_collectives
#SBATCH --account=project_462000226
#SBATCH --partition=dev-g
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --cpus-per-task=7
#SBATCH --time=00:30:00
#SBATCH --output=results/%x_%j.out
#SBATCH --constraint=eessi
#
# osu_collectives.sh — A4 inter-node collectives at N=16 (2 nodes × 8 GCDs).
# Binaries: osu_allreduce, osu_alltoall, osu_bcast, osu_allgather.
# Per-rank ROCR_VISIBLE_DEVICES is set via OMPI_COMM_WORLD_LOCAL_RANK
# (mpirun) / SLURM_LOCALID (srun) — each node's 8 ranks map to GCDs 0..7.
#
# Submit:
#   sbatch osu_collectives.sh eessi
#   sbatch osu_collectives.sh native

STACK="${1:?usage: sbatch $0 <eessi|native>}"
[[ "$STACK" == "eessi" || "$STACK" == "native" ]] \
    || { echo "ERROR: stack must be 'eessi' or 'native'" >&2; exit 1; }

source common.sh
source topology.sh

FILE_BASE="osu_collectives_${STACK}"
CSV_FILE="${RESULT_DIR}/${FILE_BASE}_${SLURM_JOB_ID}.csv"
LOG_FILE="${RESULT_DIR}/${FILE_BASE}_${SLURM_JOB_ID}.log"
META_FILE="${RESULT_DIR}/${FILE_BASE}_${SLURM_JOB_ID}.meta"

node_metadata_dump "$META_FILE"

NODES=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
[[ ${#NODES[@]} -ge 2 ]] || { echo "ERROR: need 2 nodes" | tee -a "$LOG_FILE"; exit 1; }
NODE_A=${NODES[0]}; NODE_B=${NODES[1]}
echo "STACK=$STACK  NODE_A=$NODE_A  NODE_B=$NODE_B" | tee -a "$LOG_FILE"

echo "stack,sdma_enabled,benchmark,num_nodes,num_gcds,run,size_bytes,latency_us" \
    > "$CSV_FILE"

NUM_RUNS=6; WARMUP_RUN=1; RUN_TIMEOUT=600

# Wrapper: each rank takes ROCR_VISIBLE_DEVICES = its node-local rank index.
# That maps the 8 ranks per node to GCDs 0..7 respectively.
WRAPPER="${RESULT_DIR}/wrap_coll_${SLURM_JOB_ID}.sh"
cat > "$WRAPPER" <<'EOF'
#!/bin/bash
local_rank=${SLURM_LOCALID:-${OMPI_COMM_WORLD_LOCAL_RANK:-0}}
export ROCR_VISIBLE_DEVICES=$local_rank
exec "$@"
EOF
chmod +x "$WRAPPER"

parse_coll() {
    local stack=$1 sdma=$2 bench=$3 nn=$4 ng=$5 run=$6
    awk -v st="$stack" -v sdma="$sdma" -v bench="$bench" \
        -v nn="$nn" -v ng="$ng" -v run="$run" '
        /^#/ { next }
        NF >= 2 && $1 ~ /^[0-9]+$/ {
            printf "%s,%d,%s,%d,%d,%d,%d,%.2f\n",
                   st, sdma, bench, nn, ng, run, $1, $2
        }'
}

run_bench() {
    local sdma=$1 bench=$2 bin=$3 n=$4
    local nn=$((n / 8))   # 8 GCDs/node assumed; n=16 -> 2 nodes
    {
        echo
        echo "# [$STACK sdma=$sdma] $bench  N=$n  nodes=$nn"
    } | tee -a "$LOG_FILE"

    for run in $(seq 1 $NUM_RUNS); do
        local t0=$SECONDS tag raw
        tag=$( (( run == WARMUP_RUN )) && echo "warm-up" || echo "recording" )
        echo "  [run $run/$NUM_RUNS] $tag - $(date '+%H:%M:%S')" | tee -a "$LOG_FILE"

        if [[ "$STACK" == "native" ]]; then
            raw=$(timeout $RUN_TIMEOUT \
                srun --nodes=$nn --ntasks=$n --ntasks-per-node=$((n / nn)) \
                "$WRAPPER" "$bin" -d rocm 2>>"$LOG_FILE")
        else
            raw=$(timeout $RUN_TIMEOUT \
                mpirun -n $n --map-by ppr:$((n / nn)):node \
                "$WRAPPER" "$bin" -d rocm 2>>"$LOG_FILE")
        fi
        echo "  [run $run/$NUM_RUNS] done in $((SECONDS - t0))s" | tee -a "$LOG_FILE"
        echo "$raw" >> "$LOG_FILE"
        if (( run != WARMUP_RUN )); then
            echo "$raw" | parse_coll "$STACK" "$sdma" "$bench" "$nn" "$n" "$run" \
                >> "$CSV_FILE"
        fi
    done
}

COLLECTIVES=(osu_allreduce osu_alltoall osu_bcast osu_allgather)

# ============================================================================
export HSA_ENABLE_SDMA=0
if [[ "$STACK" == "native" ]]; then
    echo "################# NATIVE STACK (HSA_ENABLE_SDMA=0) #################" | tee -a "$LOG_FILE"
    setup_native || { echo "ERROR: setup_native failed" >&2; exit 1; }
    COLL_DIR="$OSU_NATIVE_COLL"
else
    echo "################# EESSI STACK (HSA_ENABLE_SDMA=0) ##################" | tee -a "$LOG_FILE"
    setup_eessi || { echo "ERROR: setup_eessi failed" >&2; exit 1; }
    COLL_DIR="$OSU_COLL"

    # NO UCC for inter-node. 4_osu enabled UCC's UCP TL for intra-node, where
    # UCX has working transports (sm/shm/rocm). For inter-node on our hermetic
    # CXI stack, UCX has no inter-node transport (no libibverbs in the build,
    # rocm is intra-node only) — UCC tries to construct the team, blocks on
    # UCX transport discovery, and hangs MPI_Init (verified: job 18750650 sat
    # at osu_allreduce warm-up for >7 min with no progress).
    #
    # Letting OMPI's default coll components (tuned, basic, han) run their
    # algorithms over the CM PML routes inter-node messages through OFI MTL →
    # CXI — the same fast path that delivers ~24 GB/s in osu_bw.
    export OMPI_MCA_coll_ucc_enable=0
    export OMPI_MCA_accelerator=rocm
fi

for bin in "${COLLECTIVES[@]}"; do
    if [[ ! -x "${COLL_DIR}/${bin}" ]]; then
        echo "  [skip] $bin not in $COLL_DIR" | tee -a "$LOG_FILE"
        continue
    fi
    for n in "${COLL_N_VALUES[@]}"; do
        run_bench 0 "$bin" "${COLL_DIR}/${bin}" "$n"
    done
done

rm -f "$WRAPPER"
echo
echo "ALL BENCHMARKS COMPLETE  CSV=$CSV_FILE" | tee -a "$LOG_FILE"

OLD_OUT="${RESULT_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
NEW_OUT="${RESULT_DIR}/${FILE_BASE}_${SLURM_JOB_ID}.out"
[[ -f "$OLD_OUT" && "$OLD_OUT" != "$NEW_OUT" ]] && mv "$OLD_OUT" "$NEW_OUT" 2>/dev/null || true
