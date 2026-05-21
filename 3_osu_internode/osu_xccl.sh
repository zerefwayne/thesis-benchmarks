#!/bin/bash
#SBATCH --job-name=osu_xccl_eessi
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
# osu_xccl.sh — A6 RCCL via OSU XCCL across 2 nodes (N=16). EESSI only.
# Companion to osu_collectives.sh: same collective set (allreduce, alltoall,
# broadcast, allgather), same N, same wrapper, but bypasses MPI collectives
# and calls RCCL directly. Paper arXiv:2408.14090v2 Sec IV-B compares the
# two paths at scale.

source common.sh
source topology.sh

CSV_FILE="${RESULT_DIR}/${SLURM_JOB_NAME:-osu_xccl_eessi}_${SLURM_JOB_ID}.csv"
LOG_FILE="${RESULT_DIR}/${SLURM_JOB_NAME:-osu_xccl_eessi}_${SLURM_JOB_ID}.log"
META_FILE="${RESULT_DIR}/${SLURM_JOB_NAME:-osu_xccl_eessi}_${SLURM_JOB_ID}.meta"

node_metadata_dump "$META_FILE"

NODES=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
[[ ${#NODES[@]} -ge 2 ]] || { echo "ERROR: need 2 nodes" | tee -a "$LOG_FILE"; exit 1; }
NODE_A=${NODES[0]}; NODE_B=${NODES[1]}
echo "NODE_A=$NODE_A  NODE_B=$NODE_B" | tee -a "$LOG_FILE"

echo "stack,sdma_enabled,benchmark,num_nodes,num_gcds,run,size_bytes,latency_us" \
    > "$CSV_FILE"

NUM_RUNS=6; WARMUP_RUN=1; RUN_TIMEOUT=600

WRAPPER="${RESULT_DIR}/wrap_xccl_${SLURM_JOB_ID}.sh"
cat > "$WRAPPER" <<'EOF'
#!/bin/bash
local_rank=${SLURM_LOCALID:-${OMPI_COMM_WORLD_LOCAL_RANK:-0}}
export ROCR_VISIBLE_DEVICES=$local_rank
exec "$@"
EOF
chmod +x "$WRAPPER"

parse_coll() {
    local sdma=$1 bench=$2 nn=$3 ng=$4 run=$5
    awk -v sdma="$sdma" -v bench="$bench" -v nn="$nn" -v ng="$ng" -v run="$run" '
        /^#/ { next }
        NF >= 2 && $1 ~ /^[0-9]+$/ {
            printf "eessi_xccl,%d,%s,%d,%d,%d,%d,%.2f\n",
                   sdma, bench, nn, ng, run, $1, $2
        }'
}

run_xccl() {
    local sdma=$1 bench=$2 bin=$3 n=$4
    local nn=$((n / 8))
    {
        echo
        echo "# [eessi_xccl sdma=$sdma] $bench  N=$n  nodes=$nn"
    } | tee -a "$LOG_FILE"

    for run in $(seq 1 $NUM_RUNS); do
        local t0=$SECONDS tag raw
        tag=$( (( run == WARMUP_RUN )) && echo "warm-up" || echo "recording" )
        echo "  [run $run/$NUM_RUNS] $tag - $(date '+%H:%M:%S')" | tee -a "$LOG_FILE"

        raw=$(timeout $RUN_TIMEOUT \
            mpirun -n $n --map-by ppr:$((n / nn)):node \
            "$WRAPPER" "$bin" 2>>"$LOG_FILE")

        echo "  [run $run/$NUM_RUNS] done in $((SECONDS - t0))s" | tee -a "$LOG_FILE"
        echo "$raw" >> "$LOG_FILE"
        if (( run != WARMUP_RUN )); then
            echo "$raw" | parse_coll "$sdma" "$bench" "$nn" "$n" "$run" >> "$CSV_FILE"
        fi
    done
}

XCCL_BINS=(osu_xccl_allreduce osu_xccl_alltoall osu_xccl_broadcast osu_xccl_allgather)

setup_eessi || { echo "ERROR: setup_eessi failed" >&2; exit 1; }

# XCCL talks to RCCL directly; do not route via UCC. Don't pass -d (XCCL
# binaries accept accelerator type via build flags — passing -d caused (null)
# errors in earlier 4_osu attempts).

export HSA_ENABLE_SDMA=0
for binary in "${XCCL_BINS[@]}"; do
    if [[ -x "${OSU_XCCL_DIR}/${binary}" ]]; then
        for n in "${COLL_N_VALUES[@]}"; do
            run_xccl 0 "$binary" "${OSU_XCCL_DIR}/${binary}" "$n"
        done
    else
        echo "  [skip] $binary not in $OSU_XCCL_DIR" | tee -a "$LOG_FILE"
    fi
done

rm -f "$WRAPPER"
echo
echo "ALL BENCHMARKS COMPLETE  CSV=$CSV_FILE" | tee -a "$LOG_FILE"
