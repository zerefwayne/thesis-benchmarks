#!/bin/bash
#SBATCH --job-name=osu_bibw_native
#SBATCH --account=project_462000226
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --exclusive
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=28
#SBATCH --time=00:30:00
#SBATCH --output=results/%x_%j.out

source common.sh

# Output files
RESULT_DIR="results"
mkdir -p "$RESULT_DIR"
CSV_FILE="$RESULT_DIR/osu_bibw_native_${SLURM_JOB_ID}.csv"
LOG_FILE="$RESULT_DIR/osu_bibw_native_${SLURM_JOB_ID}.log"

BINARY="/users/joglekar/osu-native/osu-micro-benchmarks-7.5/c/mpi/pt2pt/standard/osu_bibw"

# CSV header
echo "pair_label,gcd_a,gcd_b,tier,num_links,run,size_bytes,bandwidth_MBps" > "$CSV_FILE"

# Configuration
NUM_RUNS=6                # 1 warm-up + 5 recorded
WARMUP_RUN=1              # which run number is the warm-up
RECORDED_RUNS_START=2     # runs 2..6 go into the CSV

# ---- topology metadata ------------------------------------------------------
# Maps (g0,g1) -> "tier,num_links" using your extracted MI250X topology
get_topology() {
    local g0=$1 g1=$2
    # Normalise pair ordering
    if (( g0 > g1 )); then local tmp=$g0; g0=$g1; g1=$tmp; fi
    local key="${g0}_${g1}"
    case "$key" in
        # Intra-package, 4 internal xGMI links
        0_1|2_3|4_5|6_7)               echo "intra_pkg,4" ;;
        # Inter-package, 2 direct xGMI links
        0_6|2_4)                       echo "inter_pkg_2link,2" ;;
        # Inter-package, 1 direct xGMI link
        0_2|1_3|1_5|3_7|4_6|5_7)       echo "inter_pkg_1link,1" ;;
        # Zero direct links (routed via intermediate GCDs)
        *)                             echo "routed,0" ;;
    esac
}

# ---- core function ----------------------------------------------------------
run_pair() {
    local g0=$1 g1=$2 label=$3 hypothesis=$4

    local topo
    topo=$(get_topology "$g0" "$g1")
    local tier="${topo%,*}"
    local nlinks="${topo#*,}"

    {
        echo
        echo "################################################################"
        echo "# $label   GCD($g0,$g1)   tier=$tier   num_links=$nlinks"
        echo "# Hypothesis: $hypothesis"
        echo "################################################################"
    } | tee -a "$LOG_FILE"

    for run in $(seq 1 $NUM_RUNS); do
        if (( run == WARMUP_RUN )); then
            echo "  [run $run] warm-up — output discarded" | tee -a "$LOG_FILE"
            srun --ntasks=2 --ntasks-per-node=2 bash -c "export ROCR_VISIBLE_DEVICES=${g0},${g1}; exec $BINARY -d rocm D D" > /dev/null 2>&1
            continue
        fi

        echo "  [run $run] recording" | tee -a "$LOG_FILE"
        local raw
        raw=$(srun --ntasks=2 --ntasks-per-node=2 bash -c "export ROCR_VISIBLE_DEVICES=${g0},${g1}; exec $BINARY -d rocm D D" 2>/dev/null)

        # Append raw OSU output to log for inspection
        echo "$raw" >> "$LOG_FILE"

        # Parse: drop comment lines, keep "size  bandwidth" data rows
        echo "$raw" | awk -v label="$label" -v g0="$g0" -v g1="$g1" \
                          -v tier="$tier" -v nl="$nlinks" -v run="$run" '
            /^#/ { next }
            NF == 2 && $1 ~ /^[0-9]+$/ {
                printf "%s,%d,%d,%s,%d,%d,%d,%.2f\n",
                       label, g0, g1, tier, nl, run, $1, $2
            }
        ' >> "$CSV_FILE"
    done
}

# ---- self-pair (HBM ceiling, qualitatively different) -----------------------
run_self_pair() {
    local g=$1 label=$2

    {
        echo
        echo "################################################################"
        echo "# $label   GCD($g,$g)   tier=self   num_links=NA"
        echo "# Hypothesis: same GCD, intra-process HBM, ~250 GB/s"
        echo "################################################################"
    } | tee -a "$LOG_FILE"

    for run in $(seq 1 $NUM_RUNS); do
        if (( run == WARMUP_RUN )); then
            echo "  [run $run] warm-up — output discarded" | tee -a "$LOG_FILE"
            srun --ntasks=2 --ntasks-per-node=2 bash -c "export ROCR_VISIBLE_DEVICES=${g},${g}; exec $BINARY -d rocm D D" > /dev/null 2>&1
            continue
        fi

        echo "  [run $run] recording" | tee -a "$LOG_FILE"
        local raw
        raw=$(srun --ntasks=2 --ntasks-per-node=2 bash -c "export ROCR_VISIBLE_DEVICES=${g},${g}; exec $BINARY -d rocm D D" 2>/dev/null)

        echo "$raw" >> "$LOG_FILE"

        echo "$raw" | awk -v label="$label" -v g="$g" -v run="$run" '
            /^#/ { next }
            NF == 2 && $1 ~ /^[0-9]+$/ {
                printf "%s,%d,%d,self,NA,%d,%d,%.2f\n",
                       label, g, g, run, $1, $2
            }
        ' >> "$CSV_FILE"
    done
}

# ============================================================================
# Run all pairs
# ============================================================================

# ----- 4-link intra-package pairs -----
run_pair 0 1 "intra_pkg_OAM0"  "intra-package, 4 links, ~100 GB/s uni"
run_pair 2 3 "intra_pkg_OAM1"  "intra-package, 4 links, ~100 GB/s uni"
run_pair 4 5 "intra_pkg_OAM2"  "intra-package, 4 links, ~100 GB/s uni"
run_pair 6 7 "intra_pkg_OAM3"  "intra-package, 4 links, ~100 GB/s uni"

# ----- 2-link inter-package pairs -----
run_pair 0 6 "inter_pkg_2link_06"  "inter-package, 2 links, ~70 GB/s uni"
run_pair 2 4 "inter_pkg_2link_24"  "inter-package, 2 links, ~70 GB/s uni"

# ----- 1-link inter-package pairs -----
run_pair 0 2 "inter_pkg_1link_02"  "inter-package, 1 link, ~35 GB/s uni"
run_pair 1 3 "inter_pkg_1link_13"  "inter-package, 1 link, ~35 GB/s uni"
run_pair 1 5 "inter_pkg_1link_15"  "inter-package, 1 link, ~35 GB/s uni"
run_pair 3 7 "inter_pkg_1link_37"  "inter-package, 1 link, ~35 GB/s uni"
run_pair 4 6 "inter_pkg_1link_46"  "inter-package, 1 link, ~35 GB/s uni"
run_pair 5 7 "inter_pkg_1link_57"  "inter-package, 1 link, ~35 GB/s uni"

# ----- routed (no direct xGMI) pairs — representative subset -----
run_pair 0 7 "routed_07"           "no direct link, routed via 2-link path"
run_pair 1 6 "routed_16"           "no direct link, routed via 2-link path"
run_pair 0 3 "routed_03"           "no direct link, routed via 1-link path"
run_pair 1 4 "routed_14"           "no direct link, routed via 1-link path"
run_pair 1 7 "routed_17"           "no direct link, routed (anomaly pair)"
run_pair 3 5 "routed_35"           "no direct link, routed (anomaly pair)"

# ----- self pair (intra-GCD HBM ceiling) -----
run_self_pair 0 "self_GCD0"

echo
echo "==============================================================="
echo "BENCHMARK RUN COMPLETED"
echo "CSV: $CSV_FILE"
echo "Log: $LOG_FILE"
echo "==============================================================="