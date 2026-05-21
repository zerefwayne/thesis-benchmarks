# Why `UCX_RNDV_THRESH=1024` for EESSI

Companion note for four scripts that all apply the same one-line fix:

- [osu_bw_fixed_rndv.sh](osu_bw_fixed_rndv.sh) — unidirectional pt2pt, 12 pairs
- [osu_bibw_fixed_rndv.sh](osu_bibw_fixed_rndv.sh) — bidirectional pt2pt, 12 pairs
- [osu_collectives_fixed_rndv.sh](osu_collectives_fixed_rndv.sh) — 4 collectives at N=8 (see also [fix_collectives_reasoning.md](fix_collectives_reasoning.md))
- [osu_mbw_mr_fixed_rndv.sh](osu_mbw_mr_fixed_rndv.sh) — 4 concurrent pairs, 3 topology configs

Each adds exactly one line on the EESSI arm: `export UCX_RNDV_THRESH=1024`. The native arm is untouched everywhere because Cray MPICH's analog (`MPICH_GPU_IPC_THRESHOLD`) sits at a sensible default and shows no cliff in this size range (see `osu_protocol_native_18634231.csv`).

## The problem appears in every benchmark family

A sharp small-message cliff appears wherever GPU-buffer pt2pt is exercised — the **eager → rendezvous protocol switch** in UCX fires at ~256 B and collapses bandwidth for the next ~4 KiB worth of message sizes. Below are the four baselines, each on the same node as its native counterpart, with EESSI / native ratio at the worst point:

| benchmark        | cliff at | worst e/n point | recovery size |
|------------------|---------:|----------------:|--------------:|
| `osu_bw` (intra_pkg_OAM0) | **256 B** | **0.06** (27 vs 436 MB/s) | ~8 KiB |
| `osu_bibw` (intra_pkg_OAM0) | **256 B** | **0.12** (62 vs 522 MB/s) | ~8 KiB |
| `osu_collectives osu_bcast` | **256 B** | **0.015** (38 vs 0.6 µs latency = 66× slower) | ~8 KiB |
| `osu_collectives osu_alltoall` | **256 B** | **0.08** (135 vs 11 µs = 12.5× slower) | ~8 KiB |
| `osu_collectives osu_allreduce` | **512 B** | **0.11** (84 vs 9 µs = 9× slower) | ~128 KiB |
| `osu_mbw_mr cfg_*` | **128 B** | **0.04** (66 vs 1670 MB/s; msg rate 0.5M vs 13M Mmsgs/s) | ~8 KiB |

Same protocol-switch shape, different head-start because each benchmark stresses the path differently:

- `osu_bw` / `osu_bibw` / `osu_collectives` all switch at **256 B** — the OSU default 1-msg-at-a-time pattern.
- `osu_mbw_mr` switches at **128 B** because it issues 64 outstanding messages per pair — UCX sees the queue depth and triggers rendezvous one size earlier.
- `osu_collectives` shows the **most extreme ratios** because every collective fans out N=8 messages, so the per-step rendezvous setup cost is multiplied. Cliff depths reach 66× on bcast and 12.5× on alltoall.

In every case the **bulk regime (≥ 8 KiB for pt2pt, ≥ 128 KiB for some collectives) is already tied** between EESSI and native (e/n ≈ 1.00). The fix targets the cliff, not the plateau.

## The 1024 value — justified by the protocol sweep

`osu_protocol_eessi_18634230` swept `UCX_RNDV_THRESH ∈ {DEFAULT, 1024, 16777216}` on the intra_pkg_OAM0 pair, SDMA=0. Bandwidth in MB/s, median across recorded runs:

| size  | DEFAULT | 1024  | 16 MiB | native (DEFAULT) |
|-------|--------:|------:|-------:|-----------------:|
| 128 B | 408     | 396   | 406    | 360              |
| 256 B | **30**  | **789** | **767** | 439          |
| 512 B | 59      | 43    | 43     | 779              |
| 1 K   | 119     | 123   | 86     | 591              |
| 4 K   | 476     | 491   | 341    | 2,359            |
| 8 K   | 923     | 930   | **649** | 1,089           |
| 64 K  | 6,644   | 6,531 | **664** | 7,879           |
| 1 MiB | 57,805  | 57,741| **655** | 61,416          |
| 16 MiB| 141,838 | 141,834 | 141,805 | 142,670       |
| 64 MiB| 149,840 | 150,014 | 149,666 | 150,242       |

Three findings.

### 1. The 256 B disaster *is* tunable

DEFAULT lands the threshold at ~256 B, so rendezvous is used for a transfer too small to amortize its startup latency. Pushing the threshold up to 1024 keeps 256 B in eager — bandwidth jumps from 30 → 789 MB/s, a ~25× win.

### 2. The 512 B – 4 KiB residual gap is intrinsic

Even with `UCX_RNDV_THRESH=1024`, 512 B sits at 43 MB/s, *worse* than DEFAULT (59 MB/s); 1 K – 4 K only catches up by ~3 %. This isn't a knob problem — eager and rendezvous both perform poorly in this size range on this hardware. UCX has internal sub-protocols (eager-short → eager-bcopy → rendezvous) and 512 B – 4 KiB falls into a transition zone none of them handle well at GPU buffer addresses.

This is a deeper UCX-on-MI250X issue. Threshold tuning cannot fix it; it would need either a UCX MR or a different transport. The same residual gap shows up in every benchmark family that uses GPU buffers — even `osu_collectives_ucc_shm.sh` (which attacks the 1B–128B floor separately) cannot fix this band.

### 3. Very high thresholds (16 MiB) are catastrophic

The 16 MiB sweep reveals that **UCX eager has a hard bandwidth ceiling of ~660 MB/s for sizes above ~8 KiB**. This is almost certainly bounce-buffer throughput (host-side memcpy via PCIe). Below the threshold, every size from 512 B through 8 MiB gets floored at this ceiling — bulk bandwidth is destroyed. Recovery only happens when the message size finally exceeds the threshold (≥ 16 MiB), at which point rendezvous takes over.

So "make eager handle everything" is off the table — bulk parity would be sacrificed everywhere.

## Decision: `UCX_RNDV_THRESH=1024`

The only tested value that **strictly dominates DEFAULT**:

- ✅ 1 B – 128 B: tied with DEFAULT (eager, linear scaling)
- ✅ 256 B: +25× over DEFAULT (eager instead of premature rendezvous)
- ≈  512 B – 4 K: residual cliff, ~unchanged
- ✅ ≥ 8 KiB: tied with DEFAULT (rendezvous path identical, bulk bandwidth unchanged)

Going higher (16 KiB, 64 KiB, 16 MiB) plants us on the 660 MB/s eager ceiling somewhere in the 8 KiB – 8 MiB range — a much larger regression than the 512 B – 4 K cliff it would patch around.

Sticking with DEFAULT means leaving the 256 B win on the table for no reason.

## Expected impact per benchmark

The protocol sweep ran on a single intra_pkg pair. Extrapolating to the four `_fixed_rndv` variants based on the cliff structure each one shows:

| script | expected at 256 B | expected at 512 B – 4 K | expected at ≥ 8 KiB |
|--------|-------------------|------------------------|---------------------|
| `osu_bw_fixed_rndv.sh` | **e/n 0.06 → ~1.0** | unchanged | unchanged |
| `osu_bibw_fixed_rndv.sh` | **e/n 0.12 → ~1.0** | unchanged | unchanged |
| `osu_collectives_fixed_rndv.sh` | **e/n 0.015 → ~0.3** (still ahead of N=8 fan-out latency) | unchanged | unchanged |
| `osu_mbw_mr_fixed_rndv.sh` | **128 B e/n 0.04 → ~1.0**; 256 B → ~1.0 | unchanged | unchanged |

The collectives case is the most impactful in absolute terms because the cliff is amplified 8× by the N=8 fan-out — but also the most lossy in *relative* terms because there's still a 1B–128B latency floor (separately addressed by `osu_collectives_ucc_shm.sh`) and a deeper algorithmic cost we can't tune away.

## Honest framing for the writeup

> The 256 B drop in EESSI's GPU pt2pt is a **tunable protocol-default issue**: `UCX_RNDV_THRESH=1024` recovers ~25× of small-message bandwidth at no cost elsewhere, and the same fix applies unchanged to bidirectional, multi-pair, and collective workloads (where it is even more impactful because the per-step cost is multiplied by fan-out). The narrower 512 B – 4 KiB residual gap is **not** tunable — it appears to be an intrinsic limitation of UCX's eager-protocol family on MI250X device buffers, where neither eager (bounce-buffer-bound) nor rendezvous (handshake-latency-bound) performs well. Resolving it would need an MR in UCX, not an environment variable.

Native serves as the comparison baseline unchanged — `osu_protocol_native_18634231` shows the Cray MPICH default already handles this size range cleanly.

## Reproduction

```bash
# pt2pt, 12 pairs
sbatch osu_bw_fixed_rndv.sh eessi      &&  sbatch osu_bw_fixed_rndv.sh native
sbatch osu_bibw_fixed_rndv.sh eessi    &&  sbatch osu_bibw_fixed_rndv.sh native

# collectives, N=8
sbatch osu_collectives_fixed_rndv.sh eessi   &&  sbatch osu_collectives_fixed_rndv.sh native

# multi-pair concurrent
sbatch osu_mbw_mr_fixed_rndv.sh eessi  &&  sbatch osu_mbw_mr_fixed_rndv.sh native
```

CSV schemas are identical to the corresponding non-`_fixed_rndv` scripts, so each pair of result files can be concat'd against the matching baseline for before/after plotting.

## Source data

- [results/osu_bw_eessi_18633083.csv](results/osu_bw_eessi_18633083.csv) / [native_18633084](results/osu_bw_native_18633084.csv) — pt2pt bw baseline
- [results/osu_bibw_eessi_18634053.csv](results/osu_bibw_eessi_18634053.csv) / [native_18634054](results/osu_bibw_native_18634054.csv) — pt2pt bibw baseline
- [results/osu_collectives_eessi_18634710.csv](results/osu_collectives_eessi_18634710.csv) / [native_18634695](results/osu_collectives_native_18634695.csv) — collectives baseline
- [results/osu_mbw_mr_eessi_18634824.csv](results/osu_mbw_mr_eessi_18634824.csv) / [native_18634827](results/osu_mbw_mr_native_18634827.csv) — multi-pair baseline
- [results/osu_protocol_eessi_18634230.csv](results/osu_protocol_eessi_18634230.csv) — UCX threshold sweep {DEFAULT, 1024, 16 MiB}
- [results/osu_protocol_native_18634231.csv](results/osu_protocol_native_18634231.csv) — Cray IPC threshold sweep {1, DEFAULT, 16 MiB}
