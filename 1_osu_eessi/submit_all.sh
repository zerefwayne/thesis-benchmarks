#!/bin/bash
# ============================================================================
# Submit the whole benchmark battery to SLURM in sensible order:
#   1. topology first (fast, must succeed for the rest to be interpretable)
#   2. focused pt2pt (quick characterization)
#   3. then everything else in parallel
#
# Each job depends on the topology one only for ordering of output files;
# all benchmarks are independent technically.
#
# Usage:  ./submit_all.sh project_462000XXX
# ============================================================================

set -e

ACCOUNT=${1:?usage: $0 PROJECT_ACCOUNT}

# Patch all scripts to use the requested account
for s in scripts/*.sh; do
    sed -i "s/^#SBATCH --account=.*/#SBATCH --account=$ACCOUNT/" "$s"
done

mkdir -p results

echo "Submitting topology probe..."
TOPO_JID=$(sbatch --parsable scripts/00_topology.sh)
echo "  -> $TOPO_JID"

echo "Submitting focused pt2pt (depends on topology)..."
FOCUS_JID=$(sbatch --parsable --dependency=afterok:$TOPO_JID scripts/02_pt2pt_focused.sh)
echo "  -> $FOCUS_JID"

# After focused completes, fan out the rest in parallel
echo "Submitting full pt2pt sweep..."
sbatch --parsable --dependency=afterok:$FOCUS_JID scripts/01_pt2pt_pairs.sh

echo "Submitting collectives (8 GCD)..."
sbatch --parsable --dependency=afterok:$FOCUS_JID scripts/03_collectives_8gcd.sh

echo "Submitting collectives scaling..."
sbatch --parsable --dependency=afterok:$FOCUS_JID scripts/04_collectives_scaling.sh

echo "Submitting one-sided RMA..."
sbatch --parsable --dependency=afterok:$FOCUS_JID scripts/05_onesided.sh

echo "Submitting H<->D study..."
sbatch --parsable --dependency=afterok:$FOCUS_JID scripts/06_host_device.sh

echo "Submitting concurrent pairs..."
sbatch --parsable --dependency=afterok:$FOCUS_JID scripts/07_concurrent_pairs.sh

echo "Submitting startup..."
sbatch --parsable --dependency=afterok:$FOCUS_JID scripts/08_startup.sh

echo
echo "All jobs queued. Use 'squeue -u $USER' to monitor."
echo "Results land in ./results/"
