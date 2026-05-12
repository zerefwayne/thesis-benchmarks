#!/bin/bash
#SBATCH --job-name=osu_mbw_mr
#SBATCH --account=project_462000226
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --exclusive
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --time=00:30:00
#SBATCH --output=results/%x_%j.out
#SBATCH --constraint=eessi

source common.sh

RESULT_DIR="results"
mkdir -p "$RESULT_DIR"
CSV_MBW="$RESULT_DIR/osu_mbw_mr_${SLURM_JOB_ID}.csv"
LOG_FILE="$RESULT_DIR/osu_mbw_mr_${SLURM_JOB_ID}.log"

NUM_RUNS=6
WARMUP_RUN=1

# CSV headers
echo "config_label,pairing_desc,num_pairs,run,size_bytes,bandwidth_MBps,msg_rate_Mps" \
    > "$CSV_MBW"

# ---- topology metadata -------------------------------------------------------
get_topology() {
    local g0=$1 g1=$2
    if (( g0 > g1 )); then local tmp=$g0; g0=$g1; g1=$tmp; fi
    case "${g0}_${g1}" in
        0_1|2_3|4_5|6_7)     echo "intra_pkg,4" ;;
        0_6|2_4)              echo "inter_pkg_2link,2" ;;
        0_2|1_3|1_5|3_7|4_6|5_7) echo "inter_pkg_1link,1" ;;
        *)                    echo "routed,0" ;;
    esac
}

# ===========================================================================
# osu_mbw_mr: aggregate bandwidth across all 8 GCDs simultaneously
#
# osu_mbw_mr pairs rank i with rank i+N/2 (block assignment).
# We control which physical GCDs those ranks land on via ROCR_VISIBLE_DEVICES.
# Three configurations exercise different topology regimes:
#
#   cfg_routed_split:   ROCR order 0,1,2,3,4,5,6,7
#     pairs: (0,4) (1,5) (2,6) (3,7)  → all routed, no direct xGMI
#
#   cfg_intra_pkg:      ROCR order 0,2,4,6,1,3,5,7
#     pairs: (0,1) (2,3) (4,5) (6,7)  → all intra-package, 4 links each
#
#   cfg_mixed_1link:    ROCR order 0,1,4,5,2,3,6,7
#     pairs: (0,2) (1,3) (4,6) (5,7)  → all 1-link inter-package
# ===========================================================================

run_mbw_config() {
    local config_label=$1
    local pairing_desc=$2
    local rocr_order=$3    # comma-separated, e.g. "0,1,2,3,4,5,6,7"
    local num_pairs=4      # always 8 GCDs → 4 pairs

    {
        echo
        echo "################################################################"
        echo "# [osu_mbw_mr] $config_label"
        echo "# Pairing: $pairing_desc"
        echo "# ROCR order: $rocr_order   (rank i ↔ rank i+4)"
        echo "################################################################"
    } | tee -a "$LOG_FILE"

    for run in $(seq 1 $NUM_RUNS); do
        if (( run == WARMUP_RUN )); then
            echo "  [run $run/6] warm-up starting" | tee -a "$LOG_FILE"
            ROCR_VISIBLE_DEVICES=$rocr_order \
                mpirun -n 8 "${OSU_PT2PT}/osu_mbw_mr" -d rocm D D \
                >/dev/null 2>&1
            continue
        fi

        echo "  [run $run/6] recording" | tee -a "$LOG_FILE"
        local raw
        raw=$(ROCR_VISIBLE_DEVICES=$rocr_order \
              mpirun -n 8 "${OSU_PT2PT}/osu_mbw_mr" -d rocm D D 2>/dev/null)

        echo "$raw" >> "$LOG_FILE"

        # osu_mbw_mr outputs: Size  MB/s  Messages/s
        echo "$raw" | awk \
            -v label="$config_label" \
            -v pairing="$pairing_desc" \
            -v np="$num_pairs" -v run="$run" '
            /^#/ { next }
            NF == 3 && $1 ~ /^[0-9]+$/ {
                printf "%s,%s,%d,%d,%d,%.2f,%.2f\n",
                    label, pairing, np, run, $1, $2, $3
            }' >> "$CSV_MBW"
    done
}

echo
echo "================================================================" | tee -a "$LOG_FILE"
echo "osu_mbw_mr aggregate (8 ranks, 4 pairs, 3 configs)"    | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"

# cfg 1: default block order — pairs all go through routed paths
run_mbw_config \
    "cfg_routed_split" \
    "(0,4)(1,5)(2,6)(3,7) all_routed" \
    "0,1,2,3,4,5,6,7"

# cfg 2: reordered so each sender GCD pairs with its OAM partner (4 links)
run_mbw_config \
    "cfg_intra_pkg" \
    "(0,1)(2,3)(4,5)(6,7) all_intra_pkg_4link" \
    "0,2,4,6,1,3,5,7"

# cfg 3: reordered so all pairs are 1-link inter-package
run_mbw_config \
    "cfg_mixed_1link" \
    "(0,2)(1,3)(4,6)(5,7) all_inter_pkg_1link" \
    "0,1,4,5,2,3,6,7"

echo
echo "================================================================"
echo "ALL BENCHMARKS COMPLETE"
echo "  osu_bw CSV  : $CSV_BW"
echo "  osu_mbw CSV : $CSV_MBW"
echo "  Log         : $LOG_FILE"
echo "================================================================"