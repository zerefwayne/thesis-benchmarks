#!/bin/bash
#SBATCH --job-name=osu_concurrent
#SBATCH --account=project_462000XXX
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --time=00:30:00
#SBATCH --output=results/concurrent_%j.out
#SBATCH --constraint=eessi

# ============================================================================
# Concurrent pair bandwidth: run 4 simultaneous pt2pt pairs at once and see
# how much aggregate bandwidth the fabric sustains. This is what real apps
# look like (many GCDs talking at once), not the isolated 2-GCD case.
#
# osu_mbw_mr is the right tool here: it's designed for many-pair bandwidth
# and message-rate measurement.
# ============================================================================

source utils/common.sh

mkdir -p results/concurrent

# All 8 GCDs, one rank each
SELECT_CPU="49-55,57-63,17-23,25-31,1-7,9-15,33-39,41-47"

cat > /tmp/wrap_${SLURM_JOB_ID}.sh <<'EOF'
#!/bin/bash
GCDS=(0 1 2 3 4 5 6 7)
export ROCR_VISIBLE_DEVICES=${GCDS[$OMPI_COMM_WORLD_LOCAL_RANK]}
exec "$@"
EOF
chmod +x /tmp/wrap_${SLURM_JOB_ID}.sh

# osu_mbw_mr with -W1 and -p<num_pairs> sets concurrent pairs
# Default pairing is rank i <-> rank i+N/2: (0,4)(1,5)(2,6)(3,7)
echo "######################################################################"
echo "# osu_mbw_mr — 4 concurrent pairs, default pairing (0-4,1-5,2-6,3-7)"
echo "######################################################################"
for placement in "D D" "H H"; do
    echo "--- placement: $placement ---"
    mpirun -np 8 \
        --bind-to cpulist:ordered \
        --cpu-set $SELECT_CPU \
        /tmp/wrap_${SLURM_JOB_ID}.sh \
        ${OSU_PT2PT}/osu_mbw_mr -d rocm $placement
    echo
done

# multi-pair latency: ranks pair up and measure aggregate latency profile
echo "######################################################################"
echo "# osu_multi_lat — 4 concurrent pair latency"
echo "######################################################################"
for placement in "D D" "H H"; do
    echo "--- placement: $placement ---"
    mpirun -np 8 \
        --bind-to cpulist:ordered \
        --cpu-set $SELECT_CPU \
        /tmp/wrap_${SLURM_JOB_ID}.sh \
        ${OSU_PT2PT}/osu_multi_lat -d rocm $placement
    echo
done

rm -f /tmp/wrap_${SLURM_JOB_ID}.sh
echo "Done."
