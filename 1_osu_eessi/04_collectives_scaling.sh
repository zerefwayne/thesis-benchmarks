#!/bin/bash
#SBATCH --job-name=osu_coll_scale
#SBATCH --account=project_462000XXX
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --time=00:45:00
#SBATCH --output=results/coll_scale_%j.out
#SBATCH --constraint=eessi

# ============================================================================
# Scaling: same collectives at np=2, 4, 8 to see how the fabric scales
# inside one node.
#
# np=2 -> just GCD 0,1 (intra-package, best case)
# np=4 -> GCDs 0,1,2,3 (one socket-half worth)
# np=8 -> all 8 GCDs (full node)
# ============================================================================

source utils/common.sh

mkdir -p results/coll_scale

# Per-scale GCD selections (rank N -> GCDs[N]) and matching cpu-sets
declare -A NP_GCDS=(
    [2]="0 1"
    [4]="0 1 2 3"
    [8]="0 1 2 3 4 5 6 7"
)
declare -A NP_CPUSET=(
    [2]="49-55,57-63"
    [4]="49-55,57-63,17-23,25-31"
    [8]="49-55,57-63,17-23,25-31,1-7,9-15,33-39,41-47"
)

build_wrapper() {
    local gcds=$1
    cat > /tmp/wrap_${SLURM_JOB_ID}.sh <<EOF
#!/bin/bash
GCDS=($gcds)
export ROCR_VISIBLE_DEVICES=\${GCDS[\$OMPI_COMM_WORLD_LOCAL_RANK]}
exec "\$@"
EOF
    chmod +x /tmp/wrap_${SLURM_JOB_ID}.sh
}

run_scale() {
    local bench=$1
    local bin="${OSU_COLL}/${bench}"
    [[ -x "$bin" ]] || { echo "SKIP $bench"; return; }

    for np in 2 4 8; do
        build_wrapper "${NP_GCDS[$np]}"
        echo "--- $bench   np=$np   GCDs=[${NP_GCDS[$np]}] ---"
        mpirun -np $np \
            --bind-to cpulist:ordered \
            --cpu-set ${NP_CPUSET[$np]} \
            /tmp/wrap_${SLURM_JOB_ID}.sh \
            "$bin" -d rocm -m 1024:4194304 -i 100 -x 10
        echo
    done
}

# Pick a representative subset for scaling (running everything would be hours)
SCALE_BENCHES=(
    osu_allreduce
    osu_alltoall
    osu_bcast
    osu_allgather
    osu_reduce
)

for c in "${SCALE_BENCHES[@]}"; do
    echo "######################################################################"
    echo "# Scaling: $c"
    echo "######################################################################"
    run_scale "$c"
done

rm -f /tmp/wrap_${SLURM_JOB_ID}.sh
echo "Done."
