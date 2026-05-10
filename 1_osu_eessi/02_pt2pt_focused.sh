#!/bin/bash
#SBATCH --job-name=osu_pt2pt_focused
#SBATCH --account=project_462000226
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --exclusive
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=28
#SBATCH --time=00:30:00
#SBATCH --output=results/pt2pt_focused_%j.out
#SBATCH --constraint=eessi

# ============================================================================
# Topology-grounded pt2pt sweep on LUMI standard-g.
#
# 3 weight classes from rocm-smi --showtopo:
#   weight 15 (2 xGMI links, FAST):  12 pairs
#   weight 30 (1 xGMI link,  MID):   14 pairs
#   weight 45 (slowest direct path): 2 pairs — (1,7) and (3,5)
#
# We pick representatives from each class plus both w=45 outliers.
#
# Simpler than before: we DON'T pin to specific cores, because SLURM only
# gives us the cores it allocated (often 0..55 not 0..63), and physical-core
# numbering inside a cgroup is unpredictable. Instead we let OpenMPI bind
# each rank to a NUMA domain and pick the GCD via ROCR_VISIBLE_DEVICES.
# What matters is the GCD pair, not which exact core.
# ============================================================================

source common.sh

# All 8 GCDs visible to job; per-rank GCD set via wrapper
export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
 
# GCD -> NUMA node mapping
declare -A GCD_NUMA=(
    [0]=3 [1]=3
    [2]=1 [3]=1
    [4]=0 [5]=0
    [6]=2 [7]=2
)
 
PAIRS_LABELED=(
    "0 1 A_w15_intra_pkg_GPU0"
    "2 3 A_w15_intra_pkg_GPU1"
    "0 2 A_w15_inter_pkg_close"
    "0 6 A_w15_inter_pkg_far"
    "0 3 B_w30_inter_pkg"
    "0 4 B_w30_inter_pkg_diag"
    "0 7 B_w30_inter_pkg_far"
    "1 7 C_w45_anomaly1"
    "3 5 C_w45_anomaly2"
    "0 0 X_self_GCD0"
)
 
WRAPPER=/tmp/wrap_${SLURM_JOB_ID}.sh
 
write_wrapper() {
    local g0=$1 g1=$2
    local n0=${GCD_NUMA[$g0]}
    local n1=${GCD_NUMA[$g1]}
    cat > "$WRAPPER" <<EOF
#!/bin/bash
case \$OMPI_COMM_WORLD_LOCAL_RANK in
    0)
        export ROCR_VISIBLE_DEVICES=$g0
        exec numactl --cpunodebind=$n0 --membind=$n0 "\$@"
        ;;
    1)
        export ROCR_VISIBLE_DEVICES=$g1
        exec numactl --cpunodebind=$n1 --membind=$n1 "\$@"
        ;;
esac
EOF
    chmod +x "$WRAPPER"
}
 
run_pair() {
    local g0=$1 g1=$2 label=$3
    write_wrapper "$g0" "$g1"
 
    # for placement in "D D" "H D" "D H"; do
    for placement in "D D"; do
        # for bench in osu_bw osu_bibw osu_latency; do
        for bench in osu_bw; do
            local bin="${OSU_PT2PT}/${bench}"
            [[ -x "$bin" ]] || { echo "  SKIP: $bench"; continue; }
 
            echo "================================================================"
            echo "  $label   GCD($g0,$g1)   $bench   placement=[$placement]"
            echo "  rank0: GCD=$g0 NUMA=${GCD_NUMA[$g0]}"
            echo "  rank1: GCD=$g1 NUMA=${GCD_NUMA[$g1]}"
            echo "================================================================"
            mpirun -np 2 \
                -x UCX_LOG_LEVEL=fatal \
                -x UCX_TLS \
                -x UCX_MEMTYPE_CACHE \
                -x ROCR_VISIBLE_DEVICES \
                -x HSA_AMD_P2P \
                --bind-to none \
                "$WRAPPER" \
                "$bin" -d rocm $placement
            echo
        done
    done
}
 
echo "######################################################################"
echo "# Topology-grounded pt2pt sweep (NUMA-local binding)"
echo "# Pairs: ${#PAIRS_LABELED[@]}"
echo "# Host: $(hostname)"
echo "# SLURM cores: $SLURM_CPUS_ON_NODE"
echo "######################################################################"
echo
 
for entry in "${PAIRS_LABELED[@]}"; do
    read g0 g1 label <<< "$entry"
    echo
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    echo ">>> Starting pair: $label  (GCD $g0, GCD $g1)"
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    run_pair "$g0" "$g1" "$label"
done
 
rm -f "$WRAPPER"
echo
echo "Done."
 