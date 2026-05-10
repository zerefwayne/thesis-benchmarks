#!/bin/bash
#SBATCH --job-name=osu_onesided
#SBATCH --account=project_462000XXX
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=7
#SBATCH --time=00:30:00
#SBATCH --output=results/onesided_%j.out
#SBATCH --constraint=eessi

# ============================================================================
# One-sided (RMA) benchmarks: put/get/accumulate.
# Run on the most interesting GCD-pair classes: intra-package (best),
# inter-package (mid), and far inter-package.
# ============================================================================

source utils/common.sh

mkdir -p results/onesided

declare -A GCD_CPU=(
    [0]=49 [1]=57 [2]=17 [3]=25
    [4]=1  [5]=9  [6]=33 [7]=41
)

# Pair set: same as focused pt2pt
PAIRS_LABELED=(
    "0 1 A_intra_pkg"
    "0 2 B_inter_close"
    "0 6 C_inter_far"
    "0 7 D_inter_diag"
)

# All osu_*put*, osu_*get*, osu_*acc* one-sided benchmarks in OMB 7.5
ONESIDED=(
    osu_put_bw
    osu_put_latency
    osu_put_bibw
    osu_get_bw
    osu_get_latency
    osu_acc_latency
    osu_fop_latency
    osu_cas_latency
    osu_get_acc_latency
)

run_pair_onesided() {
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

    for bench in "${ONESIDED[@]}"; do
        local bin="${OSU_ONESIDED}/${bench}"
        [[ -x "$bin" ]] || continue
        for placement in "D D" "H H"; do
            echo "================================================================"
            echo "  $label  GCD($g0,$g1)  $bench  [$placement]"
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
    run_pair_onesided $g0 $g1 "$label"
done

rm -f /tmp/wrap_${SLURM_JOB_ID}.sh
echo "Done."
