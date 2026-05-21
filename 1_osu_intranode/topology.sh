# topology.sh — MI250X xGMI link tier mapping and canonical pair sets
#
# get_topology() body identical to 3_osu_eessi_thesis/osu_bw.sh:33-47.
# Pair-list arrays are the source of truth for which pairs each benchmark
# sweeps over. A1/A2 uses 12 pairs (tier-balanced); A3 uses 4 (one per tier).
#
# Tier definitions (MI250X xGMI between GCDs on the same node):
#   intra_pkg          — 4 internal xGMI links per pair (~100 GB/s uni)
#   inter_pkg_2link    — 2 direct xGMI links (~70 GB/s uni)
#   inter_pkg_1link    — 1 direct xGMI link (~35 GB/s uni)
#   routed             — no direct link; goes through an intermediate GCD
#
# cpu_for_pair() is defined in cpu_bind.sh (sourced separately by callers).

get_topology() {
    local g0=$1 g1=$2
    if (( g0 > g1 )); then local tmp=$g0; g0=$g1; g1=$tmp; fi
    case "${g0}_${g1}" in
        0_1|2_3|4_5|6_7)               echo "intra_pkg,4" ;;
        0_6|2_4)                       echo "inter_pkg_2link,2" ;;
        0_2|1_3|1_5|3_7|4_6|5_7)       echo "inter_pkg_1link,1" ;;
        *)                             echo "routed,0" ;;
    esac
}

# A1/A2 pair sweep: 12 tier-balanced pairs.
#   3 intra_pkg + 2 inter_pkg_2link + 4 inter_pkg_1link + 3 routed
# Drops 6_7 from intra_pkg for symmetry; drops 1_3 and 4_6 from 1link to keep
# the CCD coverage balanced; picks 3 routed pairs spanning 2link-route,
# 1link-route, and the 1_7 anomaly flagged in 3_osu_eessi_thesis comments.
A1_PAIRS=(
    "0 1 intra_pkg_OAM0"
    "2 3 intra_pkg_OAM1"
    "4 5 intra_pkg_OAM2"
    "0 6 inter_pkg_2link_06"
    "2 4 inter_pkg_2link_24"
    "0 2 inter_pkg_1link_02"
    "1 5 inter_pkg_1link_15"
    "3 7 inter_pkg_1link_37"
    "5 7 inter_pkg_1link_57"
    "0 7 routed_07"
    "0 3 routed_03"
    "1 7 routed_17"
)

# A3 host-buffer baseline: 4 pairs anchored at GCD 0, one per tier.
A3_PAIRS=(
    "0 1 intra_pkg_OAM0"
    "0 6 inter_pkg_2link_06"
    "0 2 inter_pkg_1link_02"
    "0 7 routed_07"
)
