# topology.sh — MI250X intra-node tier map + inter-node pair definitions
#
# Independent copy of 4_osu/topology.sh extended with inter-node arrays.
# get_topology() classifies *intra-node* GCD pairs only — inter-node pairs
# are tagged with hop_class=inter_node and num_links=NA in the CSV rows
# directly, bypassing get_topology().
#
# Tier definitions (MI250X xGMI between GCDs on the same node):
#   intra_pkg          — 4 internal xGMI links per pair (~100 GB/s uni)
#   inter_pkg_2link    — 2 direct xGMI links (~70 GB/s uni)
#   inter_pkg_1link    — 1 direct xGMI link (~35 GB/s uni)
#   routed             — no direct link; goes through an intermediate GCD
# Inter-node hops go over Slingshot (libfabric/CXI on native; TCP-fallback
# on EESSI without CXI support).

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

# A1 inter-node bandwidth sweep — 1 intra-node reference + 2 inter-node pairs.
# Format: "gcd_a gcd_b hop_class pair_label num_links_or_NA"
#   intra-node ref:  (NODE_A:GCD0, NODE_A:GCD1) — 4-link intra_pkg ceiling
#   inter-node 0:    (NODE_A:GCD0, NODE_B:GCD0) — naive default pair
#   inter-node 7:    (NODE_A:GCD7, NODE_B:GCD7) — NIC-adjacent best case
#     (KFD topology: I/O link on node 11 -> GCD 7 @ ~200 GB/s)
A1_INTERNODE_PAIRS=(
    "0 1 intra_node intra_pkg_OAM0_ref   4"
    "0 0 inter_node inter_node_GCD0_GCD0 NA"
    "7 7 inter_node inter_node_GCD7_GCD7 NA"
)

# A2 collective rank modes — format: "num_nodes num_gcds hop_class"
A2_RANK_MODES=(
    "1 8  intra_node"
    "2 16 inter_node"
)

# A3 saturation single-pair (inter-node baseline, 1 flow)
A3_SAT_SINGLE_PAIR="0 0 inter_node_GCD0_GCD0_1flow"

# A3 saturation multi-pair (2 concurrent inter-node flows via osu_mbw_mr).
# 4 ranks: ranks 0,1 on NODE_A using GCDs 0,1; ranks 2,3 on NODE_B using GCDs 0,1.
# osu_mbw_mr pairs (i, i+N/2) -> (0,2) and (1,3), both inter-node.
A3_SAT_MULTI_LABEL="inter_node_2flow_GCD01"
