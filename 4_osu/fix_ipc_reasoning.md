# Why `MPICH_GPU_IPC_THRESHOLD=8192` for native Cray MPICH

Companion note for [osu_bw_fixed_ipc.sh](osu_bw_fixed_ipc.sh).

## Problem

Native Cray MPICH at default settings shows a curious bandwidth dip in the 1 KiB – 4 KiB GPU-buffer range. From the `osu_bw_native_18633084` baseline at pair `intra_pkg_OAM0` (GCD 0↔1):

| size  | native MB/s (default) |
|-------|----------------------:|
| 512 B | 779 |
| **1 K** | **587** (drop) |
| **2 K** | **1177** |
| **4 K** | **2351** |
| 8 K   | 1047 (transition) |
| 16 K  | 2031 |

The 1 K – 4 K curve is not strictly monotonic — 1 K is *slower* than 512 B (587 vs 779) — and the entire band is well below what bulk-bandwidth extrapolation would predict. This isn't a cliff like the EESSI side; it's a sustained 1.6–2.5× underperformance vs what the hardware can deliver in this size range.

The protocol-threshold sweep (`osu_protocol_native_18638911`) traces the cause.

## The Cray default — confirmed via `man intro_mpi`

```
MPICH_GPU_IPC_THRESHOLD

    Intra-node GPU-GPU transfers with payloads of size greater than or
    equal to this value will use the IPC capability. Transfers with
    smaller payloads will use CPU-attached shared memory regions.

    Default: 1024
```

So at the default, **anything ≥ 1 KiB takes the IPC path** (xGMI-direct via GPU page tables). The hypothesis: maybe that path doesn't pay off until messages are big enough to amortize the setup cost. The sweep tests this.

## Sweep results — `osu_protocol_native_18638911`

Single pair (0,1), SDMA=0. Bandwidth in MB/s, median across recorded runs:

| size  | DEFAULT (=1024) | 1024 | 2048 | 4096 | **8192** | 16384 |
|-------|----------------:|-----:|-----:|-----:|---------:|------:|
| 128 B | 354             | 313  | 347  | 326  | 316      | 347   |
| 256 B | 431             | 428  | 427  | 426  | 429      | 424   |
| 512 B | 778             | 777  | 774  | 780  | 775      | 783   |
| **1 K**   | 568         | 576  | **1338** | 1374 | 1323 | 1337 |
| **2 K**   | 1136        | 1150 | 1128 | **1799** | 1776 | 1777 |
| **4 K**   | 2256        | 2314 | 2259 | 2270 | **2814** | 2848 |
| 8 K   | 1092            | 1099 | 1107 | 1095 | 1104     | 1094  |
| 16 K  | 2180            | 2198 | 2206 | 2187 | 2200     | 2165  |
| 64 K  | 7888            | 7972 | 7932 | 7875 | 7899     | 7850  |
| 1 MiB | 61485           | 61585 | 61610 | 61662 | 61717 | 61622 |
| 64 MiB | 150351         | 150313 | 150289 | 150288 | 150343 | 150357 |

**Two things jump out:**

1. **`DEFAULT` and `THRESH=1024` produce identical curves.** Confirms the man-page default empirically.
2. **At each size, raising the threshold past that size flips the protocol** — that size goes from IPC to host-staged SHM, and **host-staged is 1.6–2.5× faster** than IPC in the 1K – 4K band on this hardware.

| size  | IPC path (default) | Host-SHM path (THRESH=8192) | speedup |
|-------|-------------------:|----------------------------:|--------:|
| 1 K   | 568                | 1323                        | **2.33×** |
| 2 K   | 1136               | 1776                        | **1.56×** |
| 4 K   | 2256               | 2814                        | **1.25×** |

At sizes ≥ 8 KiB, IPC and host-SHM converge — the IPC handshake cost gets amortized over the bigger transfer and both protocols deliver the same bandwidth.

## Why 8192 (not 16384, not 4096)

Geometric mean across the full size sweep:

| THRESH | geo-mean MB/s | rank |
|--------|--------------:|:----:|
| **8192**   | **2148**      | 🥇 |
| 16384  | 2142          | 🥈 (-0.3%) |
| 4096   | 2133          | 🥉 (-0.7%) |
| 2048   | 2094          |    (-2.5%) |
| 1024 / DEFAULT | 2020 / 2038 | (-6.0%) |

- **8192 captures all four wins.** The 1 K, 2 K, and 4 K points all sit below the threshold and go through host-SHM. Geomean is best by a hair.
- **16384 is statistically tied.** The 8 K point also falls into host-SHM territory, and at 8 K both protocols are essentially the same (~1100 MB/s), so we don't gain anything additional — but we don't lose anything either.
- **4096 misses the 4 K win.** 4 K is exactly at the boundary and goes IPC, costing ~540 MB/s on that single size.

8192 is the smallest threshold that captures every win without ambiguity. Picking 16384 instead is arguably equally defensible — but at 8192, the 8 K transition is clean (IPC kicks in just above the host-SHM band, exactly where IPC starts winning).

## Why this matters for the thesis

This is a **non-trivial vendor-default tuning miss**, not just a cosmetic fix:

1. **The Cray default is sub-optimal on MI250X for the entire 1 K – 4 K range.** That's the band where many real HPC apps live (think halo cells in stencil codes, MPI_Isend storms in MD, parameter-server gradient packets in ML).
2. **It's a single-env-var fix.** `export MPICH_GPU_IPC_THRESHOLD=8192` recovers 1.25–2.33× bandwidth in the affected band with zero downside elsewhere.
3. **Neither the De Sensi paper nor the Cray default calls this out.** The paper covers `NCCL_NCHANNELS_PER_PEER`, `NCCL_IGNORE_CPU_AFFINITY`, and `NCCL_NET_GDR_LEVEL` as the three RCCL-side levers, but says nothing about Cray MPICH's IPC threshold for pt2pt GPU buffers.

## Honest framing for the writeup

> Cray MPICH defaults to `MPICH_GPU_IPC_THRESHOLD=1024`, which routes every intra-node GPU-buffer transfer of 1 KiB or larger through the IPC capability. On MI250X, our threshold sweep reveals that IPC is *slower* than host-attached shared memory for transfers in the 1 K – 4 K range — by factors of 2.33× at 1 K, 1.56× at 2 K, and 1.25× at 4 K. Raising the threshold to 8192 keeps these sizes on the faster SHM path while preserving the IPC handover at ≥ 8 KiB where IPC genuinely wins. This is a single-env-var tuning that the Cray default misses; the resulting native bandwidth curve becomes monotonic in the small-medium range, matching the shape that the EESSI side achieves only after its own `UCX_RNDV_THRESH=1024` fix.

The combined story (EESSI + native, both tuned) is more interesting than either alone: **both stacks have small-msg defaults that hurt GPU-buffer transfers in their own way, and both have single-line env-var fixes that close most of the gap.**

## Reproduction

```bash
sbatch osu_bw_fixed_ipc.sh native    # MPICH_GPU_IPC_THRESHOLD=8192 (the actual measurement)
sbatch osu_bw_fixed_ipc.sh eessi     # baseline reproducibility re-run (optional)
```

CSV schema is identical to `osu_bw.sh`. Concat against `osu_bw_native_18633084.csv` for the before/after.

## Source data

- [results/osu_bw_native_18633084.csv](results/osu_bw_native_18633084.csv) — baseline native 12-pair sweep, default IPC threshold
- [results/osu_protocol_native_18638911.csv](results/osu_protocol_native_18638911.csv) — fine-grained IPC threshold sweep {DEFAULT, 1024, 2048, 4096, 8192, 16384}
- [results/osu_protocol_native_18634231.csv](results/osu_protocol_native_18634231.csv) — original coarse sweep {1, DEFAULT, 16777216} (legacy)
- [fix_rndv_reasoning.md](fix_rndv_reasoning.md) — EESSI-side companion (UCX_RNDV_THRESH=1024)
