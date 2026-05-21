# Inter-node collectives: EESSI (before/after IDC fix) vs native Cray MPICH

**Scope:** `osu_collectives` (`allreduce`, `bcast`, `allgather`, `alltoall`), `-d rocm` device
buffers, **N=16 (2 nodes × 8 GCDs)**, inter-node.
**Data (means over recorded runs):**
- EESSI **before** fix = job **18750788** (`osu_collectives_eessi_18750788.csv`)
- EESSI **after** fix = job **18761906** (inherits `FI_CXI_DISABLE_NON_INJECT_MSG_IDC=1` from `common.sh`)
- **native** Cray MPICH = job **18750651** (`osu_collectives_native_18750651.csv`)

**Date:** 2026-05-21. `imprv` = before/after (fix speedup); `af/na` = after/native (residual gap).

---

## 1. Headline

The IDC fix carries straight over to collectives — every collective is built from point-to-point
sends, so the per-send GPU→host staging penalty compounded across each algorithm's rounds. Removing
it gives large small-message wins:

| collective | small-msg speedup from fix | after-fix vs native (small msg) |
|------------|---------------------------:|--------------------------------:|
| bcast | **8.7–9.0×** | 3.4–4.0× |
| alltoall (1 B) | **10.4×** | 5.4× |
| allreduce | **4.1–4.2×** | 6.7–7.2× |
| allgather | **2.1–2.2×** | 10–11× |

**But the fix does not close the gap to native**, and it exposes **two further bottlenecks it was
never going to address** (§3–4): many-to-many message-rate stalls, and large-message collective
algorithm/bandwidth. The fix solves the *per-send latency* axis; collectives also stress *message
rate* and *collective algorithm quality*, where EESSI still trails native substantially.

---

## 2. Per-collective tables (mean µs, N=16)

### allreduce
| size | before | after | native | imprv | af/na |
|------|-------:|------:|-------:|------:|------:|
| 8 B | 185.82 | 45.63 | 6.34 | 4.1× | 7.2× |
| 64 B | 179.81 | 42.63 | 6.38 | 4.2× | 6.7× |
| 256 B | 45.97 | 44.62 | 13.38 | 1.0× | 3.3× |
| 1 KiB | 47.96 | 46.74 | 15.03 | 1.0× | 3.1× |
| 8 KiB | 80.52 | 78.47 | 69.43 | 1.0× | 1.1× |
| 64 KiB | 194.73 | 194.55 | 107.46 | 1.0× | 1.8× |
| 512 KiB | 1223.2 | 1220.7 | 226.2 | 1.0× | 5.4× |

The IDC penalty in allreduce was a ~140 µs adder on ≤64 B (185→45), now gone. The residual ~44 µs
small-message floor is the OMPI `tuned`/`han` allreduce over 16 ranks + per-message overhead — 3–7×
native's 6 µs. Fix has **no effect ≥256 B** (those messages were already on the DMA path).

### bcast
| size | before | after | native | imprv | af/na |
|------|-------:|------:|-------:|------:|------:|
| 1 B | 81.35 | 9.18 | 2.68 | 8.9× | 3.4× |
| 8 B | 94.80 | 10.85 | 2.69 | 8.7× | 4.0× |
| 64 B | 81.83 | 9.09 | 2.71 | 9.0× | 3.4× |
| 256 B | 9.72 | 9.52 | 4.72 | 1.0× | 2.0× |
| 1 KiB | 9.96 | 9.88 | 12.73 | 1.0× | **0.78×** |
| 8 KiB | 21.31 | 21.13 | 27.37 | 1.0× | **0.77×** |
| 64 KiB | 31.71 | 31.72 | 33.66 | 1.0× | 0.94× |
| 512 KiB | 255.11 | 257.12 | 78.12 | 1.0× | 3.3× |

Best case for the fix: bcast is a tree of point-to-point sends, each one benefited → **~9× at small
sizes**. EESSI even **beats native at 1–8 KiB**. Large (512 KiB) is 3.3× off (algorithm/bandwidth).

### allgather
| size | before | after | native | imprv | af/na |
|------|-------:|------:|-------:|------:|------:|
| 1 B | 197.24 | 92.48 | 8.96 | 2.1× | 10.3× |
| 8 B | 194.16 | 90.24 | 7.82 | 2.2× | 11.5× |
| 64 B | 136.15 | 92.77 | 11.18 | 1.5× | 8.3× |
| 256 B | 94.97 | 95.84 | 17.63 | 1.0× | 5.4× |
| 8 KiB | 154.91 | 155.94 | 86.39 | 1.0× | 1.8× |
| 64 KiB | 527.81 | 541.68 | 121.47 | 1.0× | 4.5× |
| 512 KiB | 3587.8 | 3601.1 | 441.5 | 1.0× | 8.2× |

Allgather is the weakest: only a 2× small-message win, still **8–11× off native**, and the gap
persists across all sizes (each rank collects every other rank's data → heavy many-to-many + data
volume). This is dominated by message rate and algorithm, not per-send latency.

### alltoall
| size | before | after | native | imprv | af/na |
|------|-------:|------:|-------:|------:|------:|
| 1 B | 767.26 | 73.48 | 13.74 | 10.4× | 5.4× |
| 8 B | 1400.0 | 1397.9 | 13.87 | **1.0×** | **100.8×** |
| 64 B | 1377.2 | 1390.5 | 18.98 | **1.0×** | **73.3×** |
| 256 B | 1405.8 | 1392.9 | 22.73 | **1.0×** | **61.3×** |
| 1 KiB | 74.96 | 74.40 | 71.82 | 1.0× | 1.0× |
| 8 KiB | 99.38 | 99.56 | 155.83 | 1.0× | **0.64×** |
| 64 KiB | 424.62 | 446.23 | 148.57 | 1.0× | 3.0× |
| 512 KiB | 3146.9 | 4087.5 | 691.1 | **0.8×** | 5.9× |

Alltoall is the most revealing: the fix helps **only the 1 B point** (10.4×), while **8–256 B sit
pinned at ~1390 µs both before and after** — a flat ~70–100× gap to native that the IDC fix does
**not** touch. A flat multi-hundred-µs plateau under a many-to-many pattern (16 senders each hitting
15 peers at once) is the signature of a **separate** problem (§3), not GPU staging. Note also a
possible large-message **regression** at 512 KiB (3147→4088 µs, 0.8×) — likely run-to-run variance
under congestion; worth a repeat to confirm.

(`size=1` allreduce and all `4 MiB` rows were NA — allreduce's min element is 4 B, and the 4 MiB
point was not captured within the run's cap/timeout.)

---

## 3. Bottleneck #2 (NOT fixed): many-to-many message-rate / unexpected-message stalls

The alltoall 8–256 B plateau (~1390 µs, ~70–100× native) and allgather's persistent 8–11× gap point
to a distinct cause: under concurrent many-to-many traffic, many **unexpected messages** arrive at
each rank at once. On the `cxi` provider this can exhaust the **hardware match list** and trigger
flow control / a software-matching fallback, stalling the whole collective. The IDC fix is orthogonal
to this — hence no change before vs after.

**Candidate follow-ups (separate experiment, not in this fix):**
- Sweep `FI_CXI_RX_MATCH_MODE` (`hybrid` vs `software`) specifically *for the all-to-all pattern* —
  unlike the pt2pt case (where `hardware` did nothing), the many-to-many regime is exactly where
  match-mode and buffering matter.
- Enlarge unexpected/request buffering: `FI_CXI_REQ_BUF_SIZE`, `FI_CXI_REQ_BUF_MIN_POSTED`,
  `FI_CXI_OFLOW_BUF_SIZE`, `FI_CXI_DEFAULT_CQ_SIZE` (already 131072).
- Consider a GPU-aware collective backend (RCCL) for alltoall/allgather. Note: UCC was disabled here
  because its inter-node team construction hangs on this hermetic CXI stack (no UCX inter-node
  transport) — see `osu_collectives.sh` comments; RCCL via `coll/accelerator` is the alternative.

## 4. Bottleneck #3 (NOT fixed): large-message collective algorithm / GPU-awareness

At 512 KiB every collective is 3–8× off native (allreduce 5.4×, bcast 3.3×, allgather 8.2×,
alltoall 5.9×). This is **collective algorithm quality + GPU-awareness**, not per-send latency:
OMPI's `tuned`/`han` algorithms over the CM PML vs Cray MPICH's optimized, GPU-aware collectives.
The mid-range bcast/alltoall cases where EESSI *beats* native (0.64–0.78×) show the transport itself
is competitive — the deficit is in algorithm selection and the lack of collective offload.

---

## 5. Conclusion

`FI_CXI_DISABLE_NON_INJECT_MSG_IDC=1` is a real, transferable win for collectives — **up to ~9–10×**
on small-message bcast/alltoall(1 B)/allreduce, with EESSI matching or beating native in several
mid-size cases. It fully resolves the *per-send latency* axis.

It does **not** make EESSI collectives match native overall, because collectives stress two further
axes: (#2) many-to-many message-rate/unexpected-message handling (alltoall/allgather small–mid:
8–100× off native), and (#3) large-message collective algorithm + GPU-awareness (512 KiB: 3–8× off).
Both are distinct from the pt2pt latency fix and are the right targets for any follow-up collectives
tuning. For the thesis, the honest framing is: **pt2pt latency is now competitive with native; the
IDC fix also substantially improves small-message collectives; collective performance at scale
remains gated by message-rate and algorithm/offload factors inherent to the OpenMPI-over-OFI path.**

**Raw CSVs:** `results/osu_collectives_eessi_18750788.csv` (before),
`results/osu_collectives_eessi_18761906.csv` (after),
`results/osu_collectives_native_18750651.csv` (native).
See also `IDC_LATENCY_FIX_ANALYSIS.md` (root cause + references) and `latency_fixed_analysis.md`.
