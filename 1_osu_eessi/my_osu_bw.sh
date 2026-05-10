#!/bin/bash
#SBATCH --job-name=my_osu_bw
#SBATCH --account=project_462000226
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --exclusive
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=28
#SBATCH --time=00:20:00
#SBATCH --output=results/%x_%j.out
#SBATCH --constraint=eessi

source common.sh

# ============================================================================
# All sensible GCD-pair combinations for osu_bw -d rocm D D on LUMI standard-g.
#
# Hypotheses for each pair, based on rocm-smi --showtopo weight matrix:
#
#   weight 15 (FAST, 2 xGMI link bundles) -> ~100 GB/s peak unidirectional
#                                            (~50 GB/s per SDMA engine x 2)
#   weight 30 (MID,  1 xGMI link bundle)  -> ~50 GB/s peak unidirectional
#                                            (single SDMA engine ceiling)
#   weight 45 (SLOW, weakest link path)   -> ~30-40 GB/s peak (only (1,7),(3,5))
#
# Note: peak is at 4 MiB; smaller messages dominated by latency.
# 200 GB/s+ would require osu_bibw (bidirectional) or all 4 SDMA engines.
# ============================================================================

run_pair() {
    local g0=$1 g1=$2 label=$3 hypothesis=$4
    echo
    echo "################################################################"
    echo "# $label   GCD($g0,$g1)"
    echo "# Hypothesis: $hypothesis"
    echo "################################################################"
    ROCR_VISIBLE_DEVICES=$g0,$g1 mpirun -n 2 ${OSU_PT2PT}/osu_bw -d rocm D D
}

# ----- WEIGHT 15: 2 xGMI link bundles, expect ~100 GB/s peak ------------------
run_pair 0 1 "w15_intra_pkg_GPU0"     "intra-package (same MI250X), ~100 GB/s"
run_pair 2 3 "w15_intra_pkg_GPU1"     "intra-package (same MI250X), ~100 GB/s"
run_pair 4 5 "w15_intra_pkg_GPU2"     "intra-package (same MI250X), ~100 GB/s"
run_pair 6 7 "w15_intra_pkg_GPU3"     "intra-package (same MI250X), ~100 GB/s"
run_pair 0 2 "w15_inter_pkg_close"    "inter-package, 2 links, ~100 GB/s"
run_pair 0 6 "w15_inter_pkg_far"      "inter-package, 2 links, ~100 GB/s"
run_pair 1 3 "w15_inter_pkg"          "inter-package, 2 links, ~100 GB/s"
run_pair 1 5 "w15_inter_pkg"          "inter-package, 2 links, ~100 GB/s"
run_pair 2 4 "w15_inter_pkg"          "inter-package, 2 links, ~100 GB/s"
run_pair 3 7 "w15_inter_pkg"          "inter-package, 2 links, ~100 GB/s"
run_pair 4 6 "w15_inter_pkg"          "inter-package, 2 links, ~100 GB/s"
run_pair 5 7 "w15_inter_pkg"          "inter-package, 2 links, ~100 GB/s"

# ----- WEIGHT 30: 1 xGMI link bundle, expect ~50 GB/s peak --------------------
run_pair 0 3 "w30_inter_pkg"          "inter-package, 1 link, ~50 GB/s"
run_pair 0 4 "w30_inter_pkg"          "inter-package, 1 link, ~50 GB/s"
run_pair 0 5 "w30_inter_pkg"          "inter-package, 1 link, ~50 GB/s"
run_pair 0 7 "w30_inter_pkg"          "inter-package, 1 link, ~50 GB/s"
run_pair 1 2 "w30_inter_pkg"          "inter-package, 1 link, ~50 GB/s"
run_pair 1 4 "w30_inter_pkg"          "inter-package, 1 link, ~50 GB/s"
run_pair 1 6 "w30_inter_pkg"          "inter-package, 1 link, ~50 GB/s"
run_pair 2 5 "w30_inter_pkg"          "inter-package, 1 link, ~50 GB/s"
run_pair 2 6 "w30_inter_pkg"          "inter-package, 1 link, ~50 GB/s"
run_pair 2 7 "w30_inter_pkg"          "inter-package, 1 link, ~50 GB/s"
run_pair 3 4 "w30_inter_pkg"          "inter-package, 1 link, ~50 GB/s"
run_pair 3 6 "w30_inter_pkg"          "inter-package, 1 link, ~50 GB/s"
run_pair 4 7 "w30_inter_pkg"          "inter-package, 1 link, ~50 GB/s"
run_pair 5 6 "w30_inter_pkg"          "inter-package, 1 link, ~50 GB/s"

# ----- WEIGHT 45: anomaly pairs, expect ~30-40 GB/s peak ----------------------
run_pair 1 7 "w45_anomaly1"           "weakest path on node, ~30-40 GB/s"
run_pair 3 5 "w45_anomaly2"           "weakest path on node, ~30-40 GB/s"

# ----- SELF-PAIR: intra-GCD, expect HBM-limited peak ~250 GB/s ----------------
run_pair 0 0 "self_GCD0"              "same GCD, intra-process HBM, ~250 GB/s"

echo
echo "All pairs done."