#!/bin/bash
#SBATCH --job-name=osu_coll_8gcd
#SBATCH --account=project_462000XXX
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --time=00:45:00
#SBATCH --output=results/coll_8gcd_%j.out
#SBATCH --constraint=eessi

# ============================================================================
# Collective benchmarks across all 8 GCDs of one MI250X node.
# Each MPI rank gets its own GCD with NUMA-local CPU cores.
# Tests how the full fabric handles all-to-all, allreduce, etc.
# ============================================================================

source utils/common.sh

mkdir -p results/coll_8gcd

# LUMI canonical CPU<->GCD mapping
# Format: "rank=GCD,core_low-core_high"
# Rank 0 -> GCD 0 -> CCD 6 cores 49-55, etc.
SELECT_CPU="49-55,57-63,17-23,25-31,1-7,9-15,33-39,41-47"

# Build a per-rank wrapper to bind each rank to its own GCD
cat > /tmp/wrap_${SLURM_JOB_ID}.sh <<'EOF'
#!/bin/bash
# rank -> GCD mapping (matches the cpuset order above)
GCDS=(0 1 2 3 4 5 6 7)
export ROCR_VISIBLE_DEVICES=${GCDS[$OMPI_COMM_WORLD_LOCAL_RANK]}
exec "$@"
EOF
chmod +x /tmp/wrap_${SLURM_JOB_ID}.sh

run_coll() {
    local bench=$1
    local bin="${OSU_COLL}/${bench}"

    if [[ ! -x "$bin" ]]; then
        echo "SKIP: $bench"
        return
    fi

    echo "######################################################################"
    echo "# $bench   (8 GCDs, full intra-node)"
    echo "######################################################################"

    # D D = all device buffers
    echo "--- placement: D D (all device) ---"
    mpirun -np 8 \
        --bind-to cpulist:ordered \
        --cpu-set $SELECT_CPU \
        /tmp/wrap_${SLURM_JOB_ID}.sh \
        "$bin" -d rocm -m 8:4194304 -i 100 -x 10
    echo

    # H H = host baseline for comparison
    echo "--- placement: H H (host baseline) ---"
    mpirun -np 8 \
        --bind-to cpulist:ordered \
        --cpu-set $SELECT_CPU \
        /tmp/wrap_${SLURM_JOB_ID}.sh \
        "$bin" -m 8:4194304 -i 100 -x 10
    echo
}

# Comprehensive collective coverage
COLLECTIVES=(
    osu_allreduce
    osu_allgather
    osu_allgatherv
    osu_alltoall
    osu_alltoallv
    osu_alltoallw
    osu_bcast
    osu_reduce
    osu_reduce_scatter
    osu_gather
    osu_gatherv
    osu_scatter
    osu_scatterv
    osu_barrier
    osu_iallreduce
    osu_iallgather
    osu_ialltoall
    osu_ibcast
)

for c in "${COLLECTIVES[@]}"; do
    run_coll "$c"
done

# Neighbor collectives are new in OMB 7.5
echo "######################################################################"
echo "# Neighborhood collectives (OMB 7.5)"
echo "######################################################################"
for c in osu_neighbor_allgather osu_neighbor_alltoall; do
    run_coll "$c"
done

rm -f /tmp/wrap_${SLURM_JOB_ID}.sh
echo "Done."
