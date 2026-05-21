#!/bin/bash
#SBATCH --job-name=osu_coll_tune
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
# osu_collectives_tune.sh — EESSI-only collective tuning sweep (N=16, 2x8 GCD).
#
# Targets the inter-node collective gap vs native Cray MPICH, in particular the
# osu_alltoall 8-256 B ~1390 us PLATEAU (70-100x native) — a CXI match-list /
# flow-control STALL under many-to-many traffic, NOT a bandwidth limit.
# See .claude/plans + collectives_slowdown_fix_analysis.md.
#
# Inherits the baseline CXI env from common.sh (incl. FI_CXI_DISABLE_NON_INJECT_MSG_IDC=1,
# RX_MATCH_MODE=hybrid). Each variant ADDS/overrides env on top.
#
# Usage: sbatch osu_collectives_tune.sh <variant>
#   diag_hostbuf  H H host buffers (DIAGNOSTIC: does the plateau need GPU buffers?)
#   diag_log      D D + FI_LOG warn/cxi on alltoall only (capture flow-control/LE events)
#   baseline      D D, no overrides (reproduce in this harness)
#   --- CXI matching / buffering (Phase 1B) ---
#   swmatch       RX_MATCH_MODE=software + big REQ/OFLOW buffers   [#1 stall candidate]
#   hybridpre     RX_MATCH_MODE=hybrid + HYBRID_PREEMPTIVE + POSTED_RECV_PREEMPTIVE
#   bigbuf        hybrid + enlarged OFLOW/REQ buffers + DEFAULT_TX_SIZE
#   --- forced algorithms (Phase 1C) ---
#   a2a_pairwise  coll_tuned_alltoall_algorithm=2
#   a2a_bruck     coll_tuned_alltoall_algorithm=3
#   a2a_linsync   coll_tuned_alltoall_algorithm=4
#   ag_bruck      coll_tuned_allgather_algorithm=1
#   ag_ring       coll_tuned_allgather_algorithm=3
#   ag_neighbor   coll_tuned_allgather_algorithm=4

VARIANT="${1:?usage: sbatch $0 <diag_hostbuf|diag_log|baseline|swmatch|hybridpre|bigbuf|a2a_pairwise|a2a_bruck|a2a_linsync|ag_bruck|ag_ring|ag_neighbor>}"

source common.sh
source topology.sh

FILE_BASE="osu_coll_${VARIANT}_eessi"
CSV_FILE="${RESULT_DIR}/${FILE_BASE}_${SLURM_JOB_ID}.csv"
LOG_FILE="${RESULT_DIR}/${FILE_BASE}_${SLURM_JOB_ID}.log"
META_FILE="${RESULT_DIR}/${FILE_BASE}_${SLURM_JOB_ID}.meta"

node_metadata_dump "$META_FILE"

NODES=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
[[ ${#NODES[@]} -ge 2 ]] || { echo "ERROR: need 2 nodes" | tee -a "$LOG_FILE"; exit 1; }
NODE_A=${NODES[0]}; NODE_B=${NODES[1]}
echo "VARIANT=$VARIANT  NODE_A=$NODE_A  NODE_B=$NODE_B" | tee -a "$LOG_FILE"

export HSA_ENABLE_SDMA=0
echo "############## EESSI coll tune  variant=$VARIANT  (SDMA=0) ##############" | tee -a "$LOG_FILE"
setup_eessi || { echo "ERROR: setup_eessi failed" >&2; exit 1; }
COLL_DIR="$OSU_COLL"

# Baseline collective env (same as osu_collectives.sh): no UCC inter-node.
export OMPI_MCA_coll_ucc_enable=0
export OMPI_MCA_accelerator=rocm

# --- per-variant overrides ---------------------------------------------------
MEMFLAG="-d rocm"   # device buffers by default; diag_hostbuf flips to host
case "$VARIANT" in
    baseline)     : ;;
    diag_hostbuf) MEMFLAG="" ;;
    diag_log)     : ;;   # FI_LOG applied at run time below
    swmatch)
        export FI_CXI_RX_MATCH_MODE=software
        export FI_CXI_REQ_BUF_SIZE=$((16*1024*1024))
        export FI_CXI_REQ_BUF_MIN_POSTED=8
        export FI_CXI_OFLOW_BUF_SIZE=$((8*1024*1024))
        export FI_CXI_OFLOW_BUF_MIN_POSTED=8 ;;
    hybridpre)
        export FI_CXI_RX_MATCH_MODE=hybrid
        export FI_CXI_HYBRID_PREEMPTIVE=1
        export FI_CXI_HYBRID_POSTED_RECV_PREEMPTIVE=1 ;;
    bigbuf)
        export FI_CXI_RX_MATCH_MODE=hybrid
        export FI_CXI_OFLOW_BUF_SIZE=$((16*1024*1024))
        export FI_CXI_OFLOW_BUF_MIN_POSTED=16
        export FI_CXI_REQ_BUF_SIZE=$((16*1024*1024))
        export FI_CXI_REQ_BUF_MIN_POSTED=16
        export FI_CXI_DEFAULT_TX_SIZE=2048 ;;
    a2a_pairwise) export OMPI_MCA_coll_tuned_use_dynamic_rules=1; export OMPI_MCA_coll_tuned_alltoall_algorithm=2 ;;
    a2a_bruck)    export OMPI_MCA_coll_tuned_use_dynamic_rules=1; export OMPI_MCA_coll_tuned_alltoall_algorithm=3 ;;
    a2a_linsync)  export OMPI_MCA_coll_tuned_use_dynamic_rules=1; export OMPI_MCA_coll_tuned_alltoall_algorithm=4 ;;
    ag_bruck)     export OMPI_MCA_coll_tuned_use_dynamic_rules=1; export OMPI_MCA_coll_tuned_allgather_algorithm=1 ;;
    ag_ring)      export OMPI_MCA_coll_tuned_use_dynamic_rules=1; export OMPI_MCA_coll_tuned_allgather_algorithm=3 ;;
    ag_neighbor)  export OMPI_MCA_coll_tuned_use_dynamic_rules=1; export OMPI_MCA_coll_tuned_allgather_algorithm=4 ;;
    # --- rendezvous threshold (Phase 1D): force the 8-256B eager-overflow band
    #     into zero-copy rendezvous (RDMA straight to/from GPU). ---
    rdzv0)        export FI_CXI_RDZV_THRESHOLD=0 ;;
    rdzv_low)     export FI_CXI_RDZV_THRESHOLD=64; export FI_CXI_RDZV_GET_MIN=64 ;;
    rdzv_eager0)  export FI_CXI_RDZV_EAGER_SIZE=0 ;;
    *) echo "ERROR: unknown variant '$VARIANT'" >&2; exit 1 ;;
esac

# Record which FI_CXI_* knobs the installed libfabric actually supports.
{
    echo "=== fi_info -e | grep -i FI_CXI_ (supported knobs) ==="
    fi_info -e 2>/dev/null | grep -iE "FI_CXI_(RX_MATCH_MODE|REQ_BUF|OFLOW_BUF|HYBRID|DEFAULT_TX_SIZE|DEFAULT_CQ_SIZE|RDZV)" | sed 's/^/  /'
    echo "=== applied env ==="
    env | grep -E "^(FI_CXI_|OMPI_MCA_coll_)" | sort | sed 's/^/  /'
} | tee -a "$LOG_FILE"

echo "stack,sdma_enabled,benchmark,num_nodes,num_gcds,run,size_bytes,latency_us" > "$CSV_FILE"

NUM_RUNS=4; WARMUP_RUN=1; RUN_TIMEOUT=600

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
    awk -v st="$stack" -v sdma="$sdma" -v bench="$bench" -v nn="$nn" -v ng="$ng" -v run="$run" '
        /^#/ { next }
        NF >= 2 && $1 ~ /^[0-9]+$/ { printf "%s,%d,%s,%d,%d,%d,%d,%.2f\n", st, sdma, bench, nn, ng, run, $1, $2 }'
}

run_bench() {
    local bench=$1 bin=$2 n=$3 extra=$4
    local nn=$((n / 8))
    { echo; echo "# [$VARIANT] $bench  N=$n  nodes=$nn  mem='${MEMFLAG:-host}'"; } | tee -a "$LOG_FILE"
    for run in $(seq 1 $NUM_RUNS); do
        local t0=$SECONDS tag raw
        tag=$( (( run == WARMUP_RUN )) && echo "warm-up" || echo "recording" )
        echo "  [run $run/$NUM_RUNS] $tag - $(date '+%H:%M:%S')" | tee -a "$LOG_FILE"
        raw=$(timeout $RUN_TIMEOUT mpirun -n $n --map-by ppr:$((n / nn)):node $extra \
            "$WRAPPER" "$bin" $MEMFLAG 2>>"$LOG_FILE")
        echo "  [run $run/$NUM_RUNS] done in $((SECONDS - t0))s" | tee -a "$LOG_FILE"
        echo "$raw" >> "$LOG_FILE"
        if (( run != WARMUP_RUN )); then
            echo "$raw" | parse_coll "eessi_$VARIANT" 0 "$bench" "$nn" "$n" "$run" >> "$CSV_FILE"
        fi
    done
}

N=16

# diag_log: only alltoall, with FI_LOG verbose, short — capture the stall mechanism.
if [[ "$VARIANT" == "diag_log" ]]; then
    echo "=== diag_log: alltoall under FI_LOG_LEVEL=warn FI_LOG_PROV=cxi ===" | tee -a "$LOG_FILE"
    timeout 180 mpirun -n $N --map-by ppr:8:node \
        --mca mtl_base_verbose 10 \
        -x FI_LOG_LEVEL=warn -x FI_LOG_PROV=cxi \
        "$WRAPPER" "${COLL_DIR}/osu_alltoall" -d rocm -m 1:256 2>&1 | tee -a "$LOG_FILE"
    rm -f "$WRAPPER"; echo "DIAG_LOG COMPLETE  LOG=$LOG_FILE" | tee -a "$LOG_FILE"; exit 0
fi

# diag_hostbuf runs only the two many-to-many collectives; tuning variants run all four.
if [[ "$VARIANT" == "diag_hostbuf" ]]; then
    COLLECTIVES=(osu_alltoall osu_allgather)
else
    COLLECTIVES=(osu_allreduce osu_alltoall osu_bcast osu_allgather)
fi

for bench in "${COLLECTIVES[@]}"; do
    [[ -x "${COLL_DIR}/${bench}" ]] || { echo "  [skip] $bench" | tee -a "$LOG_FILE"; continue; }
    run_bench "$bench" "${COLL_DIR}/${bench}" "$N" ""
done

rm -f "$WRAPPER"
echo; echo "VARIANT=$VARIANT COMPLETE  CSV=$CSV_FILE" | tee -a "$LOG_FILE"

OLD_OUT="${RESULT_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
NEW_OUT="${RESULT_DIR}/${FILE_BASE}_${SLURM_JOB_ID}.out"
[[ -f "$OLD_OUT" && "$OLD_OUT" != "$NEW_OUT" ]] && mv "$OLD_OUT" "$NEW_OUT" 2>/dev/null || true
