#!/bin/bash
#SBATCH --job-name=osu_hd
#SBATCH --account=project_462000XXX
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=7
#SBATCH --time=00:30:00
#SBATCH --output=results/hd_%j.out
#SBATCH --constraint=eessi

# ============================================================================
# Focused H<->D characterization.
#
# Why a separate script: H/D and D/H placements stress the PCIe path between
# the host CPU and the GCD via Infinity Fabric, NOT the inter-GCD xGMI links.
# So the interesting axis is which CCD (and thus which NUMA domain) the host
# buffer sits in, relative to the GCD.
#
# We do two studies:
#   1. NUMA-local CPU<->GCD (canonical LUMI binding) — best case
#   2. NUMA-far CPU<->GCD (deliberately wrong binding) — worst case
# Compare against the H H baseline.
# ============================================================================

source utils/common.sh

mkdir -p results/hd

# GCD -> NUMA-local CPU core
declare -A GCD_LOCAL_CPU=(
    [0]=49 [1]=57 [2]=17 [3]=25
    [4]=1  [5]=9  [6]=33 [7]=41
)
# GCD -> deliberately NUMA-far CPU core (opposite-quadrant CCD)
declare -A GCD_FAR_CPU=(
    [0]=1  [1]=9   [2]=33 [3]=41
    [4]=49 [5]=57  [6]=17 [7]=25
)

run_hd_study() {
    local g=$1 binding_kind=$2 cpu=$3
    local label="GCD${g}_${binding_kind}_cpu${cpu}"

    cat > /tmp/wrap_${SLURM_JOB_ID}.sh <<EOF
#!/bin/bash
# Both ranks see GCD $g (we use 1 GCD; the host buffer is what varies)
export ROCR_VISIBLE_DEVICES=$g
exec "\$@"
EOF
    chmod +x /tmp/wrap_${SLURM_JOB_ID}.sh

    # We actually want 2 ranks, both touching the SAME GCD, to measure
    # the H<->D path between this CPU and this GCD.
    # In OSU pt2pt one rank holds H buffer, the other holds D buffer.
    for bench in osu_bw osu_bibw osu_latency; do
        local bin="${OSU_PT2PT}/${bench}"
        for placement in "H D" "D H" "H H" "D D"; do
            echo "================================================================"
            echo "  $label  $bench  [$placement]"
            echo "================================================================"
            mpirun -np 2 \
                --bind-to cpulist:ordered \
                --cpu-set ${cpu},${cpu} \
                /tmp/wrap_${SLURM_JOB_ID}.sh \
                "$bin" -d rocm $placement
            echo
        done
    done
}

# Pick a representative GCD (0) and a contrast GCD (4, far side of socket)
for g in 0 4; do
    echo "######################################################################"
    echo "# H<->D study for GCD $g, NUMA-LOCAL binding"
    echo "######################################################################"
    run_hd_study $g "local" ${GCD_LOCAL_CPU[$g]}

    echo "######################################################################"
    echo "# H<->D study for GCD $g, NUMA-FAR binding"
    echo "######################################################################"
    run_hd_study $g "far" ${GCD_FAR_CPU[$g]}
done

rm -f /tmp/wrap_${SLURM_JOB_ID}.sh
echo "Done."
