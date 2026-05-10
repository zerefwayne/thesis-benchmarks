#!/bin/bash
#SBATCH --job-name=osu_pt2pt_focused
#SBATCH --account=project_462000XXX
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=7
#SBATCH --time=00:20:00
#SBATCH --output=results/pt2pt_focused_%j.out
#SBATCH --constraint=eessi

# ============================================================================
# Quick pt2pt characterization: pick representative GCD pairs that span the
# distinct topology classes on an MI250X node.
#
# CLASS A (intra-package): GCD pair sharing the same physical package,
#         connected by 1 internal xGMI link. Highest bandwidth.
# CLASS B (inter-package, multi-link): GCD pair connected by 2+ xGMI links.
# CLASS C (inter-package, single-link): GCD pair connected by 1 xGMI link.
# CLASS D (inter-package, no direct link): traverses 2 hops or PCIe.
#
# The exact class assignment comes from rocm-smi --showtopo on your node
# (run 00_topology.sh first). The pairs below are the conventional LUMI
# layout; verify against your topology output and adjust if needed.
# ============================================================================

source utils/common.sh

mkdir -p results/pt2pt_focused

declare -A GCD_CPU=(
    [0]=49 [1]=57 [2]=17 [3]=25
    [4]=1  [5]=9  [6]=33 [7]=41
)

# Representative pairs for each topology class
# (Adjust based on actual rocm-smi --showtopo output for your node)
PAIRS_LABELED=(
    "0 1 A_intra_pkg_GPU0"     # same package
    "2 3 A_intra_pkg_GPU1"
    "4 5 A_intra_pkg_GPU2"
    "6 7 A_intra_pkg_GPU3"
    "0 2 B_inter_pkg_close"    # adjacent packages, multi-link xGMI
    "0 6 C_inter_pkg_far"      # opposite packages, fewer links
    "0 7 D_inter_pkg_diag"     # diagonal, possibly 2-hop
    "1 4 C_inter_pkg_alt"      # cross-die alternate
)

run_pair() {
    local g0=$1 g1=$2 label=$3
    local cpu0=${GCD_CPU[$g0]}
    local cpu1=${GCD_CPU[$g1]}

    cat > /tmp/wrap_${SLURM_JOB_ID}.sh <<EOF
#!/bin/bash
case \$OMPI_COMM_WORLD_LOCAL_RANK in
    0) export ROCR_VISIBLE_DEVICES=$g0 ;;
    1) export ROCR_VISIBLE_DEVICES=$g1 ;;
esac
exec "\$@"
EOF
    chmod +x /tmp/wrap_${SLURM_JOB_ID}.sh

    for placement in "D D" "H D" "D H"; do
        for bench in osu_bw osu_bibw osu_latency; do
            local bin="${OSU_PT2PT}/${bench}"
            [[ -x "$bin" ]] || continue

            echo "================================================================"
            echo "  $label   GCD($g0,$g1)   $bench   placement=[$placement]"
            echo "  cpus=[$cpu0,$cpu1]"
            echo "================================================================"
            mpirun -np 2 \
                --bind-to cpulist:ordered \
                --cpu-set ${cpu0},${cpu1} \
                /tmp/wrap_${SLURM_JOB_ID}.sh \
                "$bin" -d rocm $placement
            echo
        done
    done
}

for entry in "${PAIRS_LABELED[@]}"; do
    read g0 g1 label <<< "$entry"
    run_pair $g0 $g1 "$label"
done

rm -f /tmp/wrap_${SLURM_JOB_ID}.sh
echo "Done."
