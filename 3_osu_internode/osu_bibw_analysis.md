# osu_bibw EESSI vs native — inter-node bidirectional analysis

Phase 1 baseline pair for [`osu_bw_analysis.md`](osu_bw_analysis.md).
Both runs use the cleaned 7_osu_internode_thesis scripts —
`-m 1:1048576 -i 100 -d rocm D D` over 2 nodes, 1 rank per node,
8 cross-node pairs from `INTERNODE_PAIRS`, 5 recorded runs per pair.

| stack  | job ID    | nodes                   | switch geometry          |
|--------|-----------|-------------------------|--------------------------|
| eessi  | 18750218  | nid005014 ↔ nid005015   | adjacent (same group)    |
| native | 18750219  | nid007968 ↔ nid007969   | adjacent (same group)    |

Unlike the osu_bw run, both stacks landed on adjacent-node pairs this
time — the comparison is apples-to-apples.

## Aggregate bandwidth vs message size

Mean across all 8 pairs × 5 runs (n=40 per size, per stack).

| size      | EESSI MB/s | native MB/s | **e/n** | notes                                    |
|----------:|-----------:|------------:|--------:|------------------------------------------|
|     1 B   |       0.14 |        1.57 |   0.089 | small-msg floor                          |
|     8 B   |       1.10 |       12.59 |   0.087 | "                                        |
|   128 B   |      17.51 |      200.82 |   0.087 | ~11× small-msg gap                       |
|   256 B   |     510.83 |      396.55 |   1.288 | **eager regime: EESSI is +29% ahead**    |
|   1 KiB   |    2038.76 |     1588.09 |   1.284 | "                                        |
|   4 KiB   |    8155.67 |     6336.55 |   1.287 | "                                        |
|   8 KiB   |   13913.26 |    13070.48 |   1.064 | rendezvous transition (EESSI catches up) |
|  16 KiB   |   22123.54 |    23727.17 |   0.932 | **EESSI dip; native crosses ahead**      |
|  32 KiB   |   30055.37 |    28632.61 |   1.050 | post-rdzv                                |
|  64 KiB   |   37413.40 |    36448.87 |   1.026 | bulk                                     |
| 128 KiB   |   41093.56 |    40722.56 |   1.009 | converging                               |
| 256 KiB   |   42811.20 |    42822.64 |   1.000 | **identical**                            |
| 512 KiB   |   43950.38 |    44078.91 |   0.997 | identical                                |
|   1 MiB   |   44478.42 |    44574.48 |   0.998 | identical                                |

Four regimes:

1. **Latency floor (1 B – 128 B): e/n ≈ 0.09.** Same small-message penalty
   as osu_bw — OpenMPI's OFI/cxi stack vs Cray MPICH per-send overhead.
   Note: ~11× here vs ~20× in unidirectional osu_bw, because bidirectional
   traffic amortizes some of the bookkeeping cost.
2. **Eager (256 B – 4 KiB): e/n ≈ 1.28 — EESSI is ~29% AHEAD.** Different
   shape from osu_bw (where EESSI was at 85%). Likely the CXI provider
   handles bidirectional eager more efficiently than Cray MPICH's IPC
   path at these sizes. Worth investigating further.
3. **Rendezvous transition (8 KiB – 16 KiB): e/n bounces 1.06 → 0.93.**
   EESSI dips at 16 KiB (similar shape to osu_bw). Native catches up
   and briefly crosses ahead.
4. **Bulk (≥ 32 KiB): e/n converges to ~1.00.** At 1 MiB both stacks
   deliver **44.5 GB/s** — within 100 MB/s of each other.

## Theoretical ceilings

| ceiling | value | source |
|---|---:|---|
| Slingshot-11 bidirectional NIC ceiling | 50.0 GB/s | 25 + 25 GB/s |
| osu_bibw 1 MiB observed (this run, both stacks) | ~44.5 GB/s | this run |
| Saturation as % of ceiling | **89%** | this run |

Both stacks achieve **~89% of the Slingshot-11 bidirectional ceiling** at
1 MiB. The 11% gap to line rate is consistent with PCIe Gen4 GPU↔NIC
overhead, MPI framing, and the cassini-headers / libfabric framing
bytes — none of it stack-attributable.

## Per-pair @ 1 MiB

| pair                | tier         | EESSI MB/s ¹       | native MB/s |
|---------------------|--------------|-------------------:|------------:|
| inter_GCD0_GCD0     | nic_via_xgmi | 41,748 (1 outlier) |      45,987 |
| inter_GCD2_GCD2     | nic_via_xgmi |             46,108 |      45,547 |
| inter_GCD4_GCD4     | nic_via_xgmi |             46,107 |      41,187 |
| inter_GCD6_GCD6     | nic_via_xgmi |             44,522 |      45,996 |
| inter_GCD1_GCD1     | nic_local    |             43,088 |      45,996 |
| inter_GCD3_GCD3     | nic_local    |             44,240 |      43,703 |
| inter_GCD5_GCD5     | nic_local    |             46,111 |      42,190 |
| inter_GCD7_GCD7     | nic_local    |             43,904 |      45,989 |
| **tier mean nic_local**    | | **44,336** | **44,470** |
| **tier mean nic_via_xgmi** | | **44,621** | **44,679** |

¹ EESSI `inter_GCD0_GCD0` includes one run at 35,280 MB/s (other 4 ≈ 43,800);
EESSI `inter_GCD6_GCD6` has one at 42,031 (other 4 ≈ 45,000). Native
`inter_GCD4_GCD4` has one at 38,024 (other 4 ≈ 41,000–42,500).

**The NIC tier signal is again statistically invisible at 1 MiB:**
- EESSI: tier delta is 285 MB/s (0.6%); within-pair runs vary by up to
  8 GB/s on the worst-affected pair.
- Native: tier delta is 210 MB/s (0.5%); same magnitude.

Same conclusion as osu_bw: at the per-NIC line-rate plateau, the GCD↔NIC
xGMI hop is below the noise floor — dragonfly switch routing dominates.
The nic_local / nic_via_xgmi labels remain useful for small/medium-msg
analysis but stop differentiating at saturation.

## Comparison vs osu_bw (unidirectional)

| metric                          | osu_bw (uni) | osu_bibw (bi) | bi/uni |
|---------------------------------|-------------:|--------------:|-------:|
| EESSI 1 MiB                     |  24.0 GB/s   |   44.5 GB/s   |  1.85  |
| native 1 MiB                    |  23.3 GB/s   |   44.6 GB/s   |  1.91  |
| Theoretical: 2× uni              | 48 / 47 GB/s | -            |   -    |
| Per-direction bibw / uni        |     -        |    ≈ 0.93     |   -    |

Bidirectional throughput is **~1.85–1.91× the unidirectional number** on
both stacks. Theoretical doubling would give 2.0; the 5–7% shortfall is
PCIe and NIC contention from running both directions concurrently — same
phenomenon observed on every Slingshot system.

## Geometry caveat resolution

The osu_bw run had a structural asymmetry (EESSI same-group, native
cross-group). This osu_bibw run **does not** — both stacks got
adjacent-node pairs within the same group. The fact that EESSI's
plateau in osu_bibw matches native within 100 MB/s suggests the 3–4%
EESSI edge observed in osu_bw was mostly the cross-group penalty
hitting native. **On clean apples-to-apples geometry, the two stacks
converge.**

## Caveats and follow-ups

1. **EESSI eager regime is ~29% faster** at 256 B – 4 KiB. Worth a
   diagnostic — is it (a) a different rendezvous threshold making
   EESSI's eager regime go further, or (b) a genuine CXI-eager
   advantage in OpenMPI's OFI MTL path? The `osu_protocol_eessi.sh`
   FI_CXI_RDZV_THRESHOLD sweep will answer (a).
2. **The 16 KiB EESSI dip (e/n = 0.93)** is the canonical eager→rdzv
   handoff — same target as osu_bw's 8–16 KiB dip.
3. **Three per-pair outliers** (EESSI GCD0 / GCD6; native GCD4) drag the
   tier means. Either re-run those pairs or use median instead of mean
   for tier comparisons. The pairs with all 5 runs clustered (EESSI
   GCD2, GCD4, GCD5; native GCD0, GCD1, GCD6, GCD7) all sit at the
   46.0 GB/s plateau — that's the real ceiling on this hardware.
4. **The 11× small-msg gap (1–128 B)** is the same OpenMPI-OFI-cxi vs
   Cray-MPICH per-send-overhead story as osu_bw. Best analysed via
   `osu_latency.sh`.

## Repro

```bash
sbatch osu_bibw.sh eessi
sbatch osu_bibw.sh native
awk -F, '$11==1048576 && $8=="inter_node"' results/osu_bibw_*.csv \
  | sort -t, -k1,1 -k3,3
```
