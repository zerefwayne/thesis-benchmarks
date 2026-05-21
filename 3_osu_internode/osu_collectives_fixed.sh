#!/bin/bash
#SBATCH --job-name=osu_collectives_fixed
#SBATCH --account=project_462000226
#SBATCH --partition=dev-g
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --cpus-per-task=7
#SBATCH --time=01:00:00
#SBATCH --output=results/%x_%j.out
#SBATCH --constraint=eessi
#
# osu_collectives_fixed.sh — A4 inter-node collectives at N=16 with the
# EESSI-side tuning bundle described in osu_collectives_eessi_18750788
# diagnosis. Native arm runs unchanged for symmetry.
#
# EESSI tuning (applied in order after setup_eessi):
#   1. FI_CXI_RX_MATCH_MODE=hardware
#      Same override as osu_latency_fixed_rxmatch.sh — forces CXI's hardware-
#      matched receive path instead of the hybrid SW-fallback, fixing the
#      ~18.7 μs small-msg floor that drives the 27–100× small-message
#      collective gaps in the baseline run.
#
#   2. OMPI_MCA_coll_han_priority=100
#      Promotes the HAN (Hierarchical Adaptive coll component) above
#      coll/tuned. HAN was designed for multi-node MPI — it runs a fast
#      intra-node algorithm (sm/shared-memory) on each node, then a small
#      inter-node algorithm across the node-leaders. For 16 ranks over 2
#      nodes that's an 8-rank intra-node reduce/bcast plus 2 inter-node
#      ranks doing the cross-node phase — fewer total inter-node hops than
#      coll/tuned's flat ring algorithm.
#
#   3. OMPI_MCA_coll_tuned_use_dynamic_rules=1 +
#      OMPI_MCA_coll_tuned_allreduce_algorithm=4 (Rabenseifner)
#      Fallback for any rank-count where HAN bails to coll/tuned. Rabenseifner
#      allreduce uses log2(N) reduce-scatter + log2(N) allgather steps, each
#      moving size/N bytes. For 1 MiB allreduce N=16 the inter-node payload
#      drops to 64 KiB per step × 8 steps = 512 KiB total, vs ring's 30 ×
#      64 KiB = 1.9 MiB total. Ring's redundant traffic is the dominant
#      cost in the baseline.
#
# Native arm: no changes (Cray MPICH already has a tuned GPU-aware tree
# allreduce that delivers 236 μs at 1 MiB — that's the target to beat).
#
# Walltime bumped 30 min → 60 min: the baseline run only got allreduce +
# alltoall in 30 min on EESSI. With the tuning we expect EESSI to be
# faster, but keeping headroom for all 4 collectives × 6 runs × N=16 ranks.
#
# Submit:
#   sbatch osu_collectives_fixed.sh eessi
#   sbatch osu_collectives_fixed.sh native    # optional baseline re-take

STACK="${1:?usage: sbatch $0 <eessi|native>}"
[[ "$STACK" == "eessi" || "$STACK" == "native" ]] \
    || { echo "ERROR: stack must be 'eessi' or 'native'" >&2; exit 1; }

source common.sh
source topology.sh

FILE_BASE="osu_collectives_fixed_${STACK}"
CSV_FILE="${RESULT_DIR}/${FILE_BASE}_${SLURM_JOB_ID}.csv"
LOG_FILE="${RESULT_DIR}/${FILE_BASE}_${SLURM_JOB_ID}.log"
META_FILE="${RESULT_DIR}/${FILE_BASE}_${SLURM_JOB_ID}.meta"

node_metadata_dump "$META_FILE"

NODES=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
[[ ${#NODES[@]} -ge 2 ]] || { echo "ERROR: need 2 nodes" | tee -a "$LOG_FILE"; exit 1; }
NODE_A=${NODES[0]}; NODE_B=${NODES[1]}
echo "STACK=$STACK  NODE_A=$NODE_A  NODE_B=$NODE_B" | tee -a "$LOG_FILE"

# Schema mirrors osu_collectives.sh exactly so CSVs concat / diff against baseline.
echo "stack,sdma_enabled,benchmark,num_nodes,num_gcds,run,size_bytes,latency_us" \
    > "$CSV_FILE"

NUM_RUNS=6; WARMUP_RUN=1; RUN_TIMEOUT=600

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
    local nn=$((n / 8))
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

    # *** THE TUNING BUNDLE ***
    # (1) Small-msg fix: CXI hardware-matched receives instead of hybrid.
    export FI_CXI_RX_MATCH_MODE=hardware

    # (2) HAN: multi-node-aware coll component, intra-node + inter-node split.
    export OMPI_MCA_coll_han_priority=100

    # (3) Fallback: force Rabenseifner allreduce (algorithm 4) if HAN bails.
    export OMPI_MCA_coll_tuned_use_dynamic_rules=1
    export OMPI_MCA_coll_tuned_allreduce_algorithm=4

    # UCC stays disabled (it has no working inter-node TL on the hermetic
    # CXI stack — confirmed by job 18750650 hanging at allreduce warm-up).
    export OMPI_MCA_coll_ucc_enable=0
    export OMPI_MCA_accelerator=rocm

    {
        echo "[fixed] FI_CXI_RX_MATCH_MODE=$FI_CXI_RX_MATCH_MODE"
        echo "[fixed] OMPI_MCA_coll_han_priority=$OMPI_MCA_coll_han_priority"
        echo "[fixed] OMPI_MCA_coll_tuned_use_dynamic_rules=$OMPI_MCA_coll_tuned_use_dynamic_rules"
        echo "[fixed] OMPI_MCA_coll_tuned_allreduce_algorithm=$OMPI_MCA_coll_tuned_allreduce_algorithm"
        echo "[fixed] OMPI_MCA_coll_ucc_enable=$OMPI_MCA_coll_ucc_enable"
    } | tee -a "$LOG_FILE"
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
