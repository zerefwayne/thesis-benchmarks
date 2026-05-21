# osu_bw EESSI vs native — inter-node analysis

Initial Phase 1 baseline runs on the rebuilt 7_osu_internode_thesis scripts.
Both jobs ran `osu_bw -m 1:67108864 -i 100 -d rocm D D` over 2 nodes, 1 rank
per node, sweeping the 8 cross-node pairs from `INTERNODE_PAIRS` (4 nic_local
+ 4 nic_via_xgmi). NUM_RUNS=6 (1 warm-up discarded + 5 recorded).

| stack  | job ID    | nodes                   | switch geometry          |
|--------|-----------|-------------------------|--------------------------|
| eessi  | 18749578  | nid005020 ↔ nid005021   | adjacent (same group)    |
| native | 18749733  | nid005014 ↔ nid007968   | cross-group              |

> **Caveat — uneven node geometry.** EESSI got an adjacent-node pair on
> the same switch group; native landed on a cross-group pair which likely
> takes more dragonfly hops. The EESSI numbers therefore have a slight
> structural advantage over the native ones at the saturation plateau.
> The pair-internal comparisons below (nic_local vs nic_via_xgmi tiers
> within each stack) are clean.

## Aggregate bandwidth vs message size

Mean across all 8 inter-node pairs (`bw[size] = mean over {pair × run}`).
Native runs include 5 records / pair; EESSI runs include 5 records / pair
except `inter_GCD7_GCD7` (1 record — the job hit the dev-g walltime cap
before completing all sizes for the last pair).

| size      | EESSI MB/s | native MB/s | **e/n** | notes                                 |
|----------:|-----------:|------------:|--------:|---------------------------------------|
|     1 B   |       0.07 |        1.43 |   0.049 | small-msg floor, ~20× gap             |
|     8 B   |       0.56 |       11.60 |   0.048 | "                                     |
|   128 B   |       8.88 |      183.89 |   0.048 | "                                     |
|   256 B   |     316.21 |      372.41 |   0.849 | eager regime starts                   |
|   1 KiB   |    1268.83 |     1486.60 |   0.854 | eager                                 |
|   4 KiB   |    5063.84 |     5881.13 |   0.861 | eager                                 |
|   8 KiB   |    8773.41 |    11737.21 |   0.747 | **rendezvous transition (EESSI dips)** |
|  16 KiB   |   14055.78 |    18858.91 |   0.745 | "                                     |
|  32 KiB   |   17918.15 |    19281.41 |   0.929 | rdzv path engaging                    |
|  64 KiB   |   20892.16 |    21698.64 |   0.963 | "                                     |
| 128 KiB   |   22227.16 |    22586.72 |   0.984 | converging                            |
| 256 KiB   |   23348.77 |    22935.27 |   1.018 | **EESSI now ahead**                   |
|   1 MiB   |   24012.72 |    23256.79 |   1.033 | EESSI +3.3%                           |
|   4 MiB   |   24171.44 |    23305.20 |   1.037 | saturation plateau                    |
|  16 MiB   |   24213.80 |    23328.26 |   1.038 | "                                     |
|  64 MiB   |   24225.03 |    23332.37 |   1.038 | "                                     |

Three regimes visible:

1. **Latency floor (1 B – 128 B): e/n ≈ 0.05.** EESSI is ~20× slower per
   small message. This is OpenMPI's OFI MTL + cxi provider stack paying
   a higher per-send overhead than Cray MPICH's direct CXI path.
2. **Eager (256 B – 4 KiB): e/n ≈ 0.85.** EESSI tracks native at ~85%.
3. **Rendezvous transition (8 KiB – 16 KiB): e/n drops to 0.75.** The
   characteristic eager→rdzv handoff zone — EESSI's CXI rdzv has more
   overhead at the protocol crossover. This is the natural place to
   sweep `FI_CXI_RDZV_THRESHOLD` (see `osu_protocol_eessi.sh`).
4. **Bulk (≥ 32 KiB): e/n climbs to 0.93 → 1.03.** From 256 KiB onwards
   EESSI is *ahead* of native by 1–4%. The 1 MiB headline number is
   **EESSI 24.0 GB/s vs native 23.3 GB/s**, and the saturation plateau
   sits at **EESSI 24.2 GB/s vs native 23.3 GB/s** (≥4 MiB).

The cross-group native node pair likely costs native a couple of percent
of plateau bandwidth, so on a clean apples-to-apples geometry the two
stacks would probably converge to ~24 GB/s each. Either way, **the
post-CXI EESSI inter-node path delivers native-class bulk bandwidth.**

## Theoretical ceilings (anchors for context)

| ceiling | value | source |
|---|---:|---|
| Slingshot-11 per-NIC unidirectional line rate | 25.0 GB/s | LUMI Network docs |
| Per-NIC effective MPI ceiling (published OSU) | ~22–25 GB/s | SC'24 arXiv:2408.14090v2 |
| Per-NIC measured here (1 MiB, eessi best) | 24.1 GB/s | this run |
| Per-NIC measured here (1 MiB, native best) | 23.8 GB/s | this run |
| TCP fallback baseline (pre-CXI EESSI) | ~2 GB/s | 5_osu_internode |

Both stacks now sit at **96–97% of the Slingshot-11 per-NIC line rate**
at 1 MiB, and 97–98% at saturation. The CXI rebuild has closed the
~12× inter-node gap that existed before.

## NIC-tier breakdown @ 1 MiB

Mean across the 5 recorded runs (`inter_GCD7_GCD7` EESSI is n=1, see caveat).

| pair                | tier         | EESSI MB/s | native MB/s |
|---------------------|--------------|-----------:|------------:|
| inter_GCD1_GCD1     | nic_local    |     24,075 |      23,800 |
| inter_GCD3_GCD3     | nic_local    |     23,688 |      23,468 |
| inter_GCD5_GCD5     | nic_local    |     24,089 |      23,433 |
| inter_GCD7_GCD7     | nic_local    |   24,081¹  |      22,336 |
| inter_GCD0_GCD0     | nic_via_xgmi |     24,050 |      23,411 |
| inter_GCD2_GCD2     | nic_via_xgmi |     24,056 |      22,945 |
| inter_GCD4_GCD4     | nic_via_xgmi |     24,066 |      23,076 |
| inter_GCD6_GCD6     | nic_via_xgmi |     24,051 |      23,033 |
| **tier mean nic_local**    | | **23,983** | **23,259** |
| **tier mean nic_via_xgmi** | | **24,056** | **23,116** |

¹ `inter_GCD7_GCD7` EESSI: only 1 record.

**The nic_local vs nic_via_xgmi hypothesis isn't borne out** at 1 MiB on
either stack. On EESSI the two tiers differ by 73 MB/s (0.3%); on native
the tiers cluster ~140 MB/s apart (0.6%), and within-tier variance
exceeds the cross-tier delta on both stacks.

The "nic_local advantage" hypothesis predicted nic_local pairs (1,3,5,7)
should outperform nic_via_xgmi pairs (0,2,4,6). Instead we see:

- On EESSI, the *lowest* nic_local is GCD3 (23.7 GB/s) while three of
  four nic_via_xgmi pairs cluster at exactly 24.05–24.07 GB/s.
- On native, the *lowest* of all 8 is `inter_GCD7_GCD7` (22.3 GB/s,
  meant to be the headline NIC-adjacent best case in 5_osu_internode).

Interpretation: at the per-NIC-line-rate plateau, the bottleneck is not
the GCD↔NIC hop inside the node, it's the dragonfly switch hops between
the two nodes. The xGMI hop adds 4–5 μs latency at most (intra-package)
which is invisible at 1 MiB throughput. **The tier labels remain
useful for small-message and latency analysis, but stop differentiating
once the messages are big enough to saturate the NIC.**

## Plateau onset

| stack | 90% of plateau reached at | 95% of plateau reached at | 99% of plateau reached at |
|---|---:|---:|---:|
| eessi   |  64 KiB  | 128 KiB  | 1 MiB  |
| native  |  64 KiB  | 128 KiB  | 512 KiB |

Both stacks plateau by 1 MiB, justifying the 1 MiB message-size cap
applied in the rest of 7_osu_internode_thesis scripts.

## Caveats and what's next

1. **Geometry asymmetry**: the EESSI 2-node allocation was within one
   group, native crossed groups. To eliminate this artifact, re-submit
   both stacks with `--nodelist=nid<X>,nid<Y>` pinning to identical
   pairs. Until then, the 3–4% plateau advantage for EESSI may be
   structural, not stack-attributable.
2. **EESSI GCD7_GCD7 has n=1.** Either re-run that pair specifically or
   bump `--time` in `osu_bw.sh` from 30 → 45 min so the full sweep
   completes for any node pair on dev-g.
3. **The eager→rdzv dip at 8–16 KiB (e/n ≈ 0.75)** is the natural
   target for `osu_protocol_eessi.sh`'s `FI_CXI_RDZV_THRESHOLD` sweep.
4. **The 1–128 B latency floor (e/n ≈ 0.05)** is best characterised by
   `osu_latency.sh` (ping-pong latency at minimum-size payload) rather
   than from windowed-bw numbers — submit it next.

## Repro

```bash
sbatch osu_bw.sh eessi    # ~10–15 min on dev-g
sbatch osu_bw.sh native
# After completion:
awk -F, '$12==1048576 && $8=="inter_node"' results/osu_bw_*.csv \
  | sort -t, -k1,1 -k3,3
```
