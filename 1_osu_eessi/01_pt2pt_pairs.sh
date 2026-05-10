#!/bin/bash
#SBATCH --job-name=osu_pt2pt_pairs
#SBATCH --account=project_462000XXX
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=7
#SBATCH --time=01:00:00
#SBATCH --output=results/pt2pt_pairs_%j.out
#SBATCH --constraint=eessi

# ============================================================================
# Pt2pt sweep across every interesting GCD<->GCD pair on a single MI250X node.
#
# We walk all 28 unordered pairs of {0..7} and run osu_bw, osu_bibw, osu_latency
# for D->D, H->D, D->H, H->H. Each rank is pinned to the CCD that is NUMA-local
# to its target GCD using the LUMI canonical mapping.
#
# This characterizes:
#   - Same-package GCDs (0-1, 2-3, 4-5, 6-7): internal xGMI
#   - Cross-package GCDs: external xGMI links (count varies by pair)
#   - Asymmetric H<->D paths
# ============================================================================

source utils/common.sh

mkdir -p results/pt2pt

# Canonical LUMI CCD<->GCD mapping (low-noise core per CCD)
# Index by GCD id, value is the CPU core to bind to.
declare -A GCD_CPU=(
    [0]=49 [1]=57
    [2]=17 [3]=25
    [4]=1  [5]=9
    [6]=33 [7]=41
)

# Benchmarks to run for each pair (binary, label, extra-flags)
BENCHES=(
    "osu_bw|Bandwidth (uni)|"
    "osu_bibw|Bandwidth (bi)|"
    "osu_latency|Latency|"
    "osu_mbw_mr|Multi-BW msg-rate|"
)

# Buffer placement combinations
PLACEMENTS=(
    "D D"   # device-device (the headline)
    "H D"   # host-to-device
    "D H"   # device-to-host
    "H H"   # host-host (sanity baseline)
)

# Build the list of all 28 unordered GCD pairs
PAIRS=()
for i in 0 1 2 3 4 5 6 7; do
    for j in 0 1 2 3 4 5 6 7; do
        if [[ $i -lt $j ]]; then
            PAIRS+=("$i $j")
        fi
    done
done

run_pair_bench() {
    local g0=$1 g1=$2 bench=$3 placement=$4
    local cpu0=${GCD_CPU[$g0]}
    local cpu1=${GCD_CPU[$g1]}
    local bin="${OSU_PT2PT}/${bench}"

    if [[ ! -x "$bin" ]]; then
        echo "  SKIP: $bench not found at $bin"
        return
    fi

    echo "----------------------------------------------------------------"
    echo "Pair (GCD $g0, GCD $g1)  bench=$bench  placement='$placement'"
    echo "  CPU bind: rank0=core$cpu0  rank1=core$cpu1"
    echo "  GPU bind: rank0=GCD$g0    rank1=GCD$g1"
    echo "----------------------------------------------------------------"

    # We use ROCR_VISIBLE_DEVICES per rank via a small wrapper.
    # Each rank sees only its assigned GCD, so MPI ranks 0/1 each have GPU 0.
    cat > /tmp/wrap_${SLURM_JOB_ID}.sh <<EOF
#!/bin/bash
case \$OMPI_COMM_WORLD_LOCAL_RANK in
    0) export ROCR_VISIBLE_DEVICES=$g0 ;;
    1) export ROCR_VISIBLE_DEVICES=$g1 ;;
esac
exec "\$@"
EOF
    chmod +x /tmp/wrap_${SLURM_JOB_ID}.sh

    mpirun -np 2 \
        --bind-to cpulist:ordered \
        --cpu-set ${cpu0},${cpu1} \
        --report-bindings \
        /tmp/wrap_${SLURM_JOB_ID}.sh \
        "$bin" -d rocm $placement 2>&1 | \
        sed 's/^/  /'
    echo
}

echo "######################################################################"
echo "# Pt2pt sweep: 28 GCD pairs x 4 benchmarks x 4 placements"
echo "######################################################################"
echo

for pair in "${PAIRS[@]}"; do
    read g0 g1 <<< "$pair"
    for bench_spec in "${BENCHES[@]}"; do
        IFS='|' read bench label _ <<< "$bench_spec"
        # Don't run H/D placements for osu_mbw_mr (it doesn't accept them
        # in the same way; only D D and H H are interesting)
        if [[ "$bench" == "osu_mbw_mr" ]]; then
            run_pair_bench $g0 $g1 $bench "D D"
            continue
        fi
        for placement in "${PLACEMENTS[@]}"; do
            run_pair_bench $g0 $g1 $bench "$placement"
        done
    done
done

rm -f /tmp/wrap_${SLURM_JOB_ID}.sh
echo "Done."
