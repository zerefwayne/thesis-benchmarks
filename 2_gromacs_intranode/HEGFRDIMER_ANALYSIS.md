# hEGFRDimer Benchmark — EESSI 2025.06 vs Native cpeAMD-25.03 on LUMI-G

Analysis of the hEGFRDimer (~465 k atoms) GROMACS 2025.1 runs, one LUMI-G
node (8 MI250X GCDs), comparing the EESSI CVMFS-delivered stack against a
hand-built native LUMI EasyBuild stack.

hEGFRDimer is the **mid-size / PME-on-GPU** tier of the suite — between
Crambin (small, launch-overhead bound) and STMV (large, sustained-compute
bound). Read this together with [CRAMBIN_ANALYSIS.md](CRAMBIN_ANALYSIS.md)
and [STMV_ANALYSIS.md](STMV_ANALYSIS.md); the cross-benchmark trend (§3) is
the actual result.

Source jobs: native `18658925`, EESSI `18658924`.
Raw data: [results/gromacs_hEGFRDimer_native_18658925.csv](results/gromacs_hEGFRDimer_native_18658925.csv),
[results/gromacs_hEGFRDimer_eessi_18658924.csv](results/gromacs_hEGFRDimer_eessi_18658924.csv).

---

## 1. Build provenance

Identical stacks to the STMV and Crambin runs — see
[STMV_ANALYSIS.md](STMV_ANALYSIS.md) §1 for the full toolchain table. In
brief: both are GROMACS 2025.1 / SYCL (AdaptiveCpp) → HIP gfx90a /
VkFFT-HIP. EESSI = OpenMPI 5.0.7 + Clang 19 + AdaptiveCpp 25.10 +
ROCm 6.4.1 from CVMFS; native = Cray MPICH + cpeAMD-25.03 +
AdaptiveCpp 25.02 + ROCm 6.3.4 from a local EasyBuild prefix.

Run nodes: EESSI on `nid007956`, native on `nid005001` — different
cabinets, identical hardware spec.

mdrun config (identical between stacks): `-nb gpu -pme gpu -bonded gpu
-nsteps 100000 -resetstep 20000 -noconfout -npme 1`.

## 2. Run statistics — 6 recorded runs each (1 warm-up discarded)

Per-run ns/day:

```
EESSI:  84.959, 85.316, 84.774, 76.396, 86.020, 85.606
Native: 85.481, 85.298, 85.813, 85.643, 85.516, 85.608
```

| Stack | mean | median | sd | CV | 95% CI |
|---|---|---|---|---|---|
| **Native** (cpeAMD-25.03-VkFFT-rocm) | 85.560 | 85.562 | 0.173 | **0.20 %** | ±0.139 |
| **EESSI** (rfoss-2025a-SYCL), all 6 | 83.845 | 85.138 | 3.677 | 4.38 % | ±2.942 |
| **EESSI**, excluding run-5 outlier | 85.335 | 85.316 | 0.500 | 0.59 % | — |

Wall time per run was ~161 s for both stacks (the EESSI outlier ran 181 s).

Native's CV of 0.20 % is the tightest run set in the entire suite.

## 3. The EESSI run-5 outlier is GROMACS, not the stack

EESSI run 5 = **76.4 ns/day** while the other five EESSI runs sit
84.8 – 86.0. Inspection of the run-5 `md.log` identifies the cause as
**GROMACS' own DLB / PME auto-tuning landing in a poor decomposition**:

- **Run 5 (anomalous):** dynamic load balancing (DLB) engaged and stayed
  on, with repeated `DD step … load imb.: force 7.x %  pme mesh/force
  1.6x`. PME mesh work ran at ~1.6× the force work and force imbalance
  held at 7–8 % for the whole run.
- **Run 4 (representative):** `DLB was off during the run due to low
  measured imbalance. Average load imbalance: 4.7 %`.

This is a known GROMACS characteristic: the launch-time PME-grid tuning
and the subsequent DLB on/off decision are stochastic and *sticky* for the
remainder of a run. A single launch can converge to a suboptimal PP/PME
balance and stay there. The same event can occur under the native stack —
native simply drew six good launches in these six runs.

**Conclusion:** the outlier is an artefact of GROMACS' internal tuning, not
a defect of the EESSI stack. It must be handled with robust statistics
(median, or mean after outlier exclusion), not attributed to the stack.

## 4. EESSI vs Native — with robust statistics, identical

| Comparison | Δ (native − eessi) |
|---|---|
| Means, all 6 runs | +2.05 % — *inflated by the run-5 outlier* |
| **Medians** | **+0.50 %** |
| Means, EESSI excluding run-5 | **+0.26 %** |

Native CI is ±0.14; EESSI excluding-outlier CI is ±0.44 — they overlap.
**At hEGFRDimer scale the two stacks are statistically indistinguishable.**
The robust native-vs-EESSI delta (+0.3–0.5 %) is well within run-to-run
noise.

## 5. Cross-benchmark trend — the real result

Placing hEGFRDimer alongside the other completed systems:

| Benchmark | Atoms | Regime | Native vs EESSI (robust) | Statistically resolved? |
|---|---|---|---|---|
| Crambin | ~20 k | launch-overhead bound | −0.4 % (median) | No — tie |
| **hEGFRDimer** | **~465 k** | **PME-on-GPU, mid-size** | **+0.5 % (median)** | **No — tie** |
| STMV | ~1.06 M | sustained-compute bound | +2.6 % | Yes — CIs near-disjoint |
| hEGFRDimerPair | ~3 M | memory / xGMI bound | ~+5 % *(preliminary, 3/6 runs)* | Pending |

**The native advantage scales monotonically with system size.** It is
effectively zero on the small and mid-size systems, where launch overhead
and PME-tuning stochasticity dominate, and only emerges once the workload
is genuinely compute- and communication-bound (STMV at ~1 M atoms;
preliminarily growing at hEGFRDimerPair's ~3 M).

Interpretation: the components that actually differ between the two stacks
— Cray MPICH on the Slingshot/CXI fabric vs EESSI's OpenMPI-on-UCX, and
Cray cpeAMD vs Clang 19 host code — only influence performance once
inter-GCD communication and sustained kernel throughput drive the runtime.
hEGFRDimer at 465 k atoms sits below that threshold.

## 6. What this means for the thesis

1. **EESSI and native are performance-equivalent for small and mid-size
   systems** (≤ ~0.5 M atoms). A LUMI user in this regime loses nothing by
   taking the zero-effort CVMFS-delivered EESSI stack over a hand-built
   native one.
2. **The native advantage is a function of compute-boundedness**, not a
   fixed stack overhead. It emerges at ~1 M atoms and remains single-digit
   percent even at 3 M.
3. **GROMACS-internal stochasticity (DLB / PME tuning)** is a real source
   of run-to-run variance — already the second such event in this suite
   after Crambin's boost-clock drift. Median-based reporting absorbs it;
   `-notunepme` with a fixed PME grid would remove it entirely at the cost
   of departing from the AMD ROCm Blog recipe.
4. Native's 0.20 % CV demonstrates that, with the AMD ROCm Blog recipe
   (`OMP_NUM_THREADS=7`, GPU-aware MPI, L3CC binding) applied, the
   measurement protocol itself is sound — the variance we do see is in the
   application's tuning layer, not the harness.

## 7. Plot-ready summary

```
benchmark,stack,mean_ns_per_day,median_ns_per_day,sd,n,note
hEGFRDimer,eessi,83.845,85.138,3.677,6,run-5 DLB outlier included
hEGFRDimer,eessi_clean,85.335,85.316,0.500,5,run-5 outlier excluded
hEGFRDimer,native,85.560,85.562,0.173,6,
```

Recommended for figures: plot the **median** with an IQR or min–max
whisker. If using mean ± SD, use the outlier-excluded EESSI row and note
the exclusion in the caption.

## References

- See [STMV_ANALYSIS.md](STMV_ANALYSIS.md) and [README.md](README.md) for
  the full toolchain table, methodology, and complete reference list.
- HECBioSim — "HPC Benchmarking Results" (hEGFRDimer TPR provenance and
  cross-cluster reference numbers). <https://www.hecbiosim.ac.uk/access-hpc/hpc-benchmarking>
