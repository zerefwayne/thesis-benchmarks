#!/bin/bash
#SBATCH --job-name=my_osu_coll
#SBATCH --account=project_462000226
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --exclusive
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --time=00:20:00
#SBATCH --output=results/%x_%j.out
#SBATCH --constraint=eessi

source common.sh

# ============================================================================
# Collective benchmarks across all 8 GCDs of one MI250X node, D D placement.
# ============================================================================

run_coll() {
    local bench=$1
    local note=$2
    echo
    echo "################################################################"
    echo "# $bench   (8 GCDs, D D)"
    echo "# Note: $note"
    echo "################################################################"
    ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 mpirun -n 8 ${OSU_COLL}/${bench} -d rocm
}

# ----- Latency-style collectives (small msg performance matters) ----
run_coll osu_barrier         "synchronization only"
run_coll osu_bcast           "1->all broadcast"
run_coll osu_reduce          "all->1 reduction"

# ----- Bandwidth-style collectives (large msg, all-to-all) -----------
run_coll osu_allreduce       "all->all reduction, used in DL training"
run_coll osu_allgather       "every rank gets every rank's data"
run_coll osu_alltoall        "full pairwise exchange, fabric stress"
run_coll osu_reduce_scatter  "reduce + scatter combined"

# ----- Asymmetric collectives ----------------------------------------
run_coll osu_gather          "all->root gather"
run_coll osu_scatter         "root->all scatter"

# ----- Non-blocking variants (overlap potential) ---------------------
run_coll osu_iallreduce      "non-blocking allreduce"
run_coll osu_ialltoall       "non-blocking alltoall"
run_coll osu_ibcast          "non-blocking broadcast"

echo
echo "All collectives done."
