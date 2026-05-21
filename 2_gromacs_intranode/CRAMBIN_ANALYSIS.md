# Crambin Benchmark — EESSI 2025.06 vs Native cpeAMD-25.03 on LUMI-G

Analysis of the Crambin (~20 k atoms) GROMACS 2025.1 runs, one LUMI-G node
(8 MI250X GCDs), comparing the EESSI CVMFS-delivered stack against a
hand-built native LUMI EasyBuild stack.

Crambin is the **small-system / launch-overhead-bound** tier of the suite.
Its companion analysis for the compute-bound regime is
[STMV_ANALYSIS.md](STMV_ANALYSIS.md); read both together — the contrast
between them is the actual result.

Source jobs: native `18658570`, EESSI `18658571`.
Raw data: [results/gromacs_crambin_native_18658570.csv](results/gromacs_crambin_native_18658570.csv),
[results/gromacs_crambin_eessi_18658571.csv](results/gromacs_crambin_eessi_18658571.csv).

---

## 1. Build provenance

Identical stacks to the STMV runs — see [STMV_ANALYSIS.md](STMV_ANALYSIS.md)
§1 for the full toolchain table. In brief: both are GROMACS 2025.1 /
SYCL (AdaptiveCpp) → HIP gfx90a / VkFFT-HIP. EESSI uses OpenMPI 5.0.7 +
Clang 19 + AdaptiveCpp 25.10 + ROCm 6.4.1 from CVMFS; native uses Cray
MPICH + cpeAMD-25.03 + AdaptiveCpp 25.02 + ROCm 6.3.4 from a local
EasyBuild prefix.

mdrun config (identical between stacks): `-nb gpu -pme gpu -update gpu
-bonded gpu -nsteps 100000 -resethway -noconfout -npme 1`. Note Crambin is
the only system in the suite that runs `-update gpu` explicitly and uses
`-resethway` (timer reset at half-time) rather than `-resetstep`.

## 2. Run statistics — 6 recorded runs each (1 warm-up discarded)

Per-run ns/day:

```
EESSI:  422.473, 414.943, 414.512, 411.088, 406.241, 403.143
Native: 413.057, 414.003, 382.546, 408.951, 398.681, 413.521
```

| Stack | mean | median | sd | CV | range | 95% CI |
|---|---|---|---|---|---|---|
| **Native** (cpeAMD-25.03-VkFFT-rocm) | 405.13 | 411.00 | 12.48 | 3.08 % | 382.5 – 414.0 | ±9.98 |
| **EESSI** (rfoss-2025a-SYCL) | 412.07 | 412.80 | 6.89 | 1.67 % | 403.1 – 422.5 | ±5.51 |

Wall time per run was ~21 s for both stacks (vs ~145 s for STMV) — Crambin
is ~7× shorter, which is central to interpreting the numbers below.

## 3. The headline: at Crambin scale the stacks are statistically indistinguishable

The mean delta is −1.68 % (EESSI *higher* than native), but the **median
delta is only −0.44 %** and the 95 % CIs overlap heavily:

- Native: 395.1 – 415.1
- EESSI: 406.6 – 417.6

Unlike STMV — where native led by a cleanly-resolved +2.6 % with
near-disjoint CIs — here the difference is well inside the noise band. At
Crambin scale the EESSI and native stacks are **performance-equivalent**.

This is the expected and coherent result. Crambin (~20 k atoms, ~210 µs per
step) is **launch-overhead / CPU-GPU-synchronisation bound**, not
sustained-compute bound. The stack-level differences that produced the STMV
gap (Cray MPICH vs OpenMPI 5; Cray cpeAMD vs Clang 19 host compiler) act on
sustained kernel/communication throughput — at Crambin's per-step time
scale they are dominated by per-step scheduling jitter.

## 4. Native variance is one outlier, not instability

Native run 4 = **382.5 ns/day**, while the other five cluster 398.7 – 414.0.
Excluding it: native mean 409.6, sd 6.45, **CV 1.57 %** — essentially
identical to EESSI's 1.67 %. So native's headline CV of 3.08 % is a single
anomalous run, not systematic instability.

With only 6 samples one outlier shifts the mean by ~4 ns/day, which is why
the **median (411.0) is the more robust statistic here** — and the native
median sits right on top of the EESSI median (412.8). Reporting medians for
Crambin is the honest choice; see also §5.

## 5. EESSI shows a monotonic downward drift — a methodology note

The EESSI runs decline monotonically across the job:

```
422.5 → 414.9 → 414.5 → 411.1 → 406.2 → 403.1     (−19.3 ns/day, −4.6 %)
```

Native does **not** show this pattern — it is noisy but not trending. The
monotonic shape is the signature of **GPU boost-clock decay**: the first
recorded run still has thermal/clock headroom from idle, and consecutive
~21 s runs progressively settle toward base clocks. The single discarded
warm-up run is not enough to fully settle clocks for a benchmark this short.

Consequences:
- For Crambin specifically, report the **median** (drift-robust) or note
  the drift explicitly. The mean is biased upward by the first run or two.
- This does **not** affect STMV: at 148 s per run the warm-up is ample and
  no drift was observed (STMV EESSI CV = 0.38 %, no monotonic trend).
- A future refinement for the small-system tier would be 2–3 warm-up runs,
  or a short fixed cooldown between runs, so all recorded runs sit at the
  same clock state. The effect is small (~5 %) and within the noise band,
  so it does not invalidate the current Crambin numbers — it just argues
  for median-based reporting.

## 6. Context

- **No AMD ROCm Blog comparator** exists for Crambin — the blog benchmarks
  STMV and ADH multidir only. Crambin is a HECBioSim system; its only
  external anchor is HECBioSim's own cross-cluster reference tables.
- **vs the legacy untuned run:** [2_gbs_eessi Crambin](../2_gbs_eessi/results/gromacs_crambin_18543985.out)
  measured 318.9 ns/day (single run, no `OMP_NUM_THREADS`, no GPU-aware MPI
  enforcement, no CPU binding). Both tuned stacks now sit at ~405–412 ns/day
  — a **+27–29 % lift** attributable to the AMD ROCm Blog recipe applied in
  `common.sh`.
- **CPU-binding fix confirmed again:** EESSI Crambin CV fell from ≈10 %
  (unpinned, job 18655778) to 1.67 % once `--map-by ppr:1:l3cache:PE=7` was
  added to the EESSI launcher. Without it the small EESSI/native difference
  would be unmeasurable.

## 7. For the thesis — the two-benchmark contrast

The Crambin and STMV results together form the actual finding:

| Benchmark | Atoms | Regime | Native vs EESSI | Statistically resolved? |
|---|---|---|---|---|
| Crambin | ~20 k | launch-overhead bound | −0.4 % (median) | No — stacks indistinguishable |
| STMV | ~1.06 M | sustained-compute bound | +2.6 % (native) | Yes — CIs near-disjoint |

**The stack-level performance gap is a function of how compute-bound the
workload is.** On small systems the EESSI CVMFS stack and the hand-built
native cpeAMD stack are interchangeable; the ~2.6 % native edge only
emerges once the GPUs are genuinely saturated (STMV). This is a clean,
defensible thesis claim and it strengthens the EESSI portability argument
for the small/medium-system regime: a user running modest systems on LUMI
loses nothing by taking the zero-effort CVMFS stack over a hand-built one.

## 8. Plot-ready summary

```
benchmark,stack,mean_ns_per_day,median_ns_per_day,sd,n
crambin,eessi,412.067,412.800,6.886,6
crambin,native,405.127,411.004,12.476,6
```

Error bars: sample SD over the 6 recorded runs. Given the native outlier
and the EESSI drift, the **median** is the preferred central estimator for
Crambin — consider plotting median with an IQR or min–max whisker rather
than mean ± SD.

## References

- See [STMV_ANALYSIS.md](STMV_ANALYSIS.md) and [README.md](README.md) for
  the full toolchain table, methodology, and the complete reference list.
- HECBioSim — "HPC Benchmarking Results" (Crambin TPR provenance and
  cross-cluster reference numbers). <https://www.hecbiosim.ac.uk/access-hpc/hpc-benchmarking>
