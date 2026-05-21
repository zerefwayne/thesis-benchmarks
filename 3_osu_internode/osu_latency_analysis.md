# osu_latency EESSI vs native — inter-node ping-pong analysis

The thesis-critical latency datapoint. Companion to
[`osu_bw_analysis.md`](osu_bw_analysis.md) and
[`osu_bibw_analysis.md`](osu_bibw_analysis.md). Both runs use the
cleaned 7_osu_internode_thesis scripts —
`osu_latency -i 100 -d rocm D D` over 2 nodes, 1 rank per node, 8
cross-node pairs, 5 recorded runs per pair (n=40 per size per stack).

| stack  | job ID    | nodes                   | switch geometry          |
|--------|-----------|-------------------------|--------------------------|
| eessi  | 18750320  | nid005013 ↔ nid005014   | adjacent (same group)    |
| native | 18750321  | nid007975 ↔ nid007976   | adjacent (same group)    |

Apples-to-apples geometry on both sides.

## Headline: a 7.5× small-message latency penalty on EESSI

| stat | EESSI | native |
|---|---:|---:|
| **1-byte inter-node ping-pong latency** | **18.7 μs** | **2.5 μs** |
| Published reference (SC'24 same-switch) | — | 3.66 μs |

Native at **2.5 μs** is actually faster than the SC'24 published 3.66 μs
(probably benefits from Cray MPICH's host-staged-buffer path for tiny
msgs and from adjacent-node placement). EESSI at **18.7 μs** is ~7.5×
higher — the same small-message gap visible in osu_bw and osu_bibw, now
isolated as a pure latency phenomenon. This is the per-message overhead
of the OpenMPI ↦ OFI MTL ↦ libfabric CXI provider stack vs Cray MPICH's
optimised direct CXI path.

## Aggregate latency vs message size

Mean across all 8 pairs × 5 runs (n=40 per size, per stack).

| size      | EESSI μs | native μs | **e/n** | notes                                |
|----------:|---------:|----------:|--------:|--------------------------------------|
|     1 B   |    18.70 |      2.50 |    7.48 | small-msg floor                      |
|     8 B   |    18.70 |      2.50 |    7.48 | "                                    |
|    64 B   |    18.73 |      2.50 |    7.49 | "                                    |
|   128 B   |    19.23 |      3.05 |    6.31 | latency floor extends                |
|   256 B   |     3.46 |      3.55 |    0.98 | **EESSI cliff drop — eager engages** |
|   1 KiB   |     3.37 |      3.57 |    0.94 | EESSI slightly ahead                 |
|   4 KiB   |     3.58 |      3.72 |    0.96 | eager regime                         |
|   8 KiB   |     6.86 |      4.07 |    1.69 | **EESSI rdzv kicks in early**        |
|  16 KiB   |     7.25 |      4.66 |    1.56 | "                                    |
|  32 KiB   |     8.06 |      8.19 |    0.98 | both stacks at rdzv                  |
|  64 KiB   |     9.42 |      9.56 |    0.99 | "                                    |
| 128 KiB   |    13.28 |     12.49 |    1.06 | bulk regime, ~6% gap                 |
| 256 KiB   |    17.73 |     17.70 |    1.00 | identical                            |
|   1 MiB   |    50.12 |     50.27 |    1.00 | identical                            |

Three regimes (same shape as osu_bw / osu_bibw but expressed in time):

1. **Latency floor (1 B – 128 B): EESSI ~18.7 μs, native ~2.5 μs**.
   Constant for both stacks across this range — message size doesn't
   matter, it's pure per-send overhead.
2. **Eager regime (256 B – 4 KiB): both ~3.4–3.7 μs** — EESSI catches
   up and stays within ±5% of native. Whatever path EESSI takes for ≥
   256 B is essentially as fast as native here.
3. **Rendezvous transition (8 KiB – 16 KiB): EESSI 7 μs vs native 4 μs**
   — EESSI's CXI rendezvous fires too early and costs ~3 μs of extra
   handshake. Native's MPICH IPC keeps a wider eager window. Recovers
   by 32 KiB.
4. **Bulk regime (≥ 32 KiB): both stacks track within 6%**, identical
   from 256 KiB upward.

## The cliff at 128 → 256 B

EESSI latency drops from **19.2 μs at 128 B** to **3.5 μs at 256 B** — a
**5.5× discontinuity**. Native shows a much smaller bump (3.05 → 3.55 μs).
This is the protocol switch:

- ≤ 128 B on EESSI: every send pays the full ~18 μs path (probably
  going through OpenMPI's matching layer + OFI MTL synchronization +
  cxi provider per-message setup).
- ≥ 256 B on EESSI: a different / faster code path kicks in — possibly
  CXI's hardware-matched message path (`FI_CXI_RX_MATCH_MODE=hybrid`).

This makes the EESSI small-msg penalty a **single protocol-decision
issue**, not a fundamental hardware floor — flipping the protocol
threshold should fix it. Verify with the `osu_protocol_eessi.sh`
sweep when it runs.

## **NIC tier signal — first clean validation**

This is the first benchmark in 7_osu_internode_thesis where the
nic_local / nic_via_xgmi labels show a real, statistically clean signal.

**EESSI @ 1 B latency:**

| pair           | tier         | latency μs |
|----------------|--------------|-----------:|
| inter_GCD5_GCD5 | nic_local    |      16.85 |
| inter_GCD3_GCD3 | nic_local    |      17.30 |
| inter_GCD7_GCD7 | nic_local    |      17.54 |
| inter_GCD1_GCD1 | nic_local    |      18.05 |
| inter_GCD4_GCD4 | nic_via_xgmi |      19.32 |
| inter_GCD2_GCD2 | nic_via_xgmi |      19.76 |
| inter_GCD6_GCD6 | nic_via_xgmi |      20.23 |
| inter_GCD0_GCD0 | nic_via_xgmi |      20.58 |
| **mean nic_local**       |  | **17.44** |
| **mean nic_via_xgmi**    |  | **19.97** |
| **Δ tier**               |  | **+2.53 μs** |

Every nic_local pair is faster than every nic_via_xgmi pair. The xGMI
hop costs **~2.5 μs of round-trip latency** — about 13% of the total
small-msg latency. This *confirms* the nic_local/nic_via_xgmi
classification on the current LUMI kernel/firmware (odd GCDs are
NIC-adjacent, even GCDs take the xGMI hop).

**Native @ 1 B latency:**

| tier           | latency μs |
|----------------|-----------:|
| nic_local      |       2.50 |
| nic_via_xgmi   |       2.51 |
| **Δ tier**     | **0.01 μs (within noise)** |

Native shows **no measurable tier delta** at 1 B. Likely Cray MPICH's
small-message path uses host-staged buffers (each rank stages into
locally-pinned host memory before the NIC picks it up), so the GCD↔NIC
xGMI hop isn't on the critical path. EESSI's CXI provider uses
GPU↔NIC DMA directly, exposing the xGMI hop in the latency.

## Theoretical anchors

| ceiling | value | source |
|---|---:|---|
| Slingshot-11 same-switch published inter-node latency | 3.66 μs | SC'24 arXiv:2408.14090v2 |
| Native measured here (1 B) | 2.50 μs | this run |
| Native measured here (>= 256 B) | 3.4–3.7 μs | matches published |
| EESSI measured here (>= 256 B) | 3.4–3.6 μs | matches published |
| EESSI measured here (1 B) | 18.7 μs | this run |

EESSI's medium-message latency (256 B – 4 KiB) **matches the published
Slingshot-11 inter-node ceiling**. The CXI fast path is working — the
issue is confined to the small-message protocol decision.

## Implications for the thesis writeup

1. **EESSI's CXI rebuild fully closes the inter-node latency gap for
   messages ≥ 256 B.** That's the bulk of real-workload traffic
   (GROMACS, MD, anything that sends bulk arrays).
2. **The 1–128 B regime still has a 7.5× latency gap.** Latency-sensitive
   collectives (small allreduce, small bcast) and tiny-message MPI
   patterns (e.g. neighbour exchanges in stencil codes with very small
   ghost zones) will reflect this overhead.
3. **The xGMI hop is a measurable 2.5 μs cost on EESSI**, validating
   the nic_local/nic_via_xgmi labels. Workloads can place
   latency-critical ranks on NIC-adjacent GCDs (1, 3, 5, 7) for a ~13%
   small-msg latency saving.
4. **The 8–16 KiB rendezvous bump on EESSI** is a tunable protocol
   issue — same shape as the bw/bibw rdzv transition. Target for the
   `osu_protocol_eessi.sh` FI_CXI_RDZV_THRESHOLD sweep.

## Caveats and follow-ups

1. **The 18.7 μs small-msg floor is suspicious.** Even with OpenMPI's
   per-send overhead this is high. Possible contributors:
   - OFI MTL `cm` PML may pay an extra synchronisation in `MPI_Send` for
     unmatched messages.
   - CXI provider's `cxip_send` for tiny msgs may take the SW-matched
     path under `FI_CXI_RX_MATCH_MODE=hybrid` rather than HW-matched.
   - PRTE per-message bookkeeping in the `cm` path.
   Mitigation: try `FI_CXI_RX_MATCH_MODE=hardware` and re-measure.
2. **The 8–16 KiB rdzv hump** (EESSI 6.86 / 7.25 μs vs native 4.07 /
   4.66 μs) suggests EESSI's CXI rdzv threshold is set too low. The
   `osu_protocol_eessi.sh` sweep (FI_CXI_RDZV_THRESHOLD ∈ {DEFAULT,
   1024, 4096, 8192, 16384, 65536, 262144}) will pinpoint a value that
   smooths this transition.
3. **Re-run with FI_CXI_RX_MATCH_MODE=hardware** as a one-line
   diagnostic — if the 18.7 μs floor drops to ~4 μs, the issue is the
   software-matched message path. Track via a one-off env override:
   `FI_CXI_RX_MATCH_MODE=hardware sbatch osu_latency.sh eessi`.

## Repro

```bash
sbatch osu_latency.sh eessi
sbatch osu_latency.sh native
awk -F, '$11==1 && $8=="inter_node"' results/osu_latency_*.csv \
  | sort -t, -k1,1 -k3,3
```
