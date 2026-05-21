# topology.sh — Inter-node tier-balanced pair set for LUMI MI250X / Slingshot-11.
#
# Every entry in this file describes an INTER-node measurement — same-node /
# intra-package work belongs in 4_osu/ (Cray MPICH path) or 5_osu_internode/
# (the EESSI/CXI characterisation). Keeping the array entries scoped to
# inter-node pairs avoids confusing intra-node reference rows leaking into
# CSVs whose schema and analysis pipeline are inter-node only.
#
# Inter-node tier definitions (Slingshot-11 per-NIC PCIe topology):
#   nic_local          — GCD has direct PCIe to its OAM's local Slingshot NIC
#   nic_via_xgmi       — GCD sits on the same OAM as the NIC but is the "other"
#                        GCD of the pair, so traffic to the local NIC takes one
#                        intra-OAM xGMI hop first
#
# The nic_local/nic_via_xgmi convention follows the 5_osu_internode comment
# "KFD I/O link on GCD 7 -> ~200 GB/s": odd GCDs (1, 3, 5, 7) are the NIC-
# adjacent GCD of each OAM, even GCDs (0, 2, 4, 6) take the xGMI hop. If
# inspection on a different kernel/firmware shows this mapping is inverted,
# swap the GCDs between the two tiers — the launchers are tier-agnostic.

# A1/A8 inter-node bandwidth sweep — 8 cross-node pairs spanning both NIC tiers.
# Format: "gcd_a gcd_b pair_label nic_class"
#   nic_via_xgmi same-GCD:  even GCDs (0,2,4,6) — one xGMI hop to local NIC
#   nic_local same-GCD:     odd GCDs (1,3,5,7) — direct PCIe to local NIC
#
# 5_osu_internode measured (0,0) ≈ 22.7 GB/s and (7,7) ≈ 23.2 GB/s at 1 MiB.
# Covering all 4 OAMs in each tier exposes per-NIC variability and confirms
# (or refutes) the NIC-adjacency hypothesis end-to-end.
INTERNODE_PAIRS=(
    "0 0 inter_GCD0_GCD0 nic_via_xgmi"
    "2 2 inter_GCD2_GCD2 nic_via_xgmi"
    "4 4 inter_GCD4_GCD4 nic_via_xgmi"
    "6 6 inter_GCD6_GCD6 nic_via_xgmi"
    "1 1 inter_GCD1_GCD1 nic_local"
    "3 3 inter_GCD3_GCD3 nic_local"
    "5 5 inter_GCD5_GCD5 nic_local"
    "7 7 inter_GCD7_GCD7 nic_local"
)

# A3 host-buffer (H H) baseline — one representative pair per inter-node tier.
A3_HOST_PAIRS=(
    "0 0 inter_GCD0_GCD0 nic_via_xgmi"
    "7 7 inter_GCD7_GCD7 nic_local"
)

# A4 collective N values for inter-node sweep. Single value for now (N=16, 2x8).
COLL_N_VALUES=(16)

# A7 multi-pair concurrent ROCR orderings — 4 ranks per node, 8 total ranks.
# osu_mbw_mr pairs (i, i+N/2) which for N=8 gives (0,4) (1,5) (2,6) (3,7) —
# all cross-node. Each config picks 4 GCDs per node in a specific order so
# the (i, i+N/2) pairing maps each rank to a specific GCD class:
#   cfg_nic_local_per_node: each node uses 1,3,5,7 (all NIC-local)
#   cfg_nic_via_xgmi_per_node: each node uses 0,2,4,6 (all xGMI-hop)
#   cfg_mixed: alternating local/xgmi per node
MBW_CFG_NAMES=(cfg_nic_local_per_node cfg_nic_via_xgmi_per_node cfg_mixed)
MBW_CFG_DESCS=(
    "nodeA(1,3,5,7)_nodeB(1,3,5,7)_all_nic_local"
    "nodeA(0,2,4,6)_nodeB(0,2,4,6)_all_nic_via_xgmi"
    "nodeA(1,0,3,2)_nodeB(5,4,7,6)_mixed_per_node"
)
# Per-node ROCR_VISIBLE_DEVICES — applied on both nodes for symmetric configs.
MBW_CFG_ROCR_PER_NODE=(
    "1,3,5,7"
    "0,2,4,6"
    "1,0,3,2"   # node A in mixed cfg; node B uses the next entry
)
MBW_CFG_ROCR_PER_NODE_B=(
    "1,3,5,7"
    "0,2,4,6"
    "5,4,7,6"
)
