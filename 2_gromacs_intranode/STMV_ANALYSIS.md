# STMV Benchmark — EESSI 2025.06 vs Native cpeAMD-25.03 on LUMI-G

Analysis of the STMV (~1.06 M atoms) GROMACS 2025.1 runs, one LUMI-G node
(8 MI250X GCDs), comparing the EESSI CVMFS-delivered stack against a
hand-built native LUMI EasyBuild stack, and cross-validating both against
the published AMD ROCm Blog reference numbers.

Source jobs: native `18656108`, EESSI `18656109`.
Raw data: [results/gromacs_stmv_native_18656108.csv](results/gromacs_stmv_native_18656108.csv),
[results/gromacs_stmv_eessi_18656109.csv](results/gromacs_stmv_eessi_18656109.csv).

---

## 1. Build provenance

Both stacks resolve to **GROMACS 2025.1 / SYCL (AdaptiveCpp) → HIP gfx90a /
VkFFT-HIP**. From the per-job `.meta` snapshots:

| Component | EESSI 2025.06 | Native (LUMI EasyBuild-user) |
|---|---|---|
| GROMACS | 2025.1 | 2025.1 |
| GPU backend | SYCL (AdaptiveCpp) → HIP gfx90a | SYCL (AdaptiveCpp) → HIP gfx90a |
| AdaptiveCpp | 25.10.0 | 25.02.0-rocm6 |
| ROCm | 6.4.1 | 6.3.4 |
| GPU FFT | VkFFT 1.3.1 (HIP backend) | VkFFT 1.3.1 (HIP backend) |
| CPU FFT | FFTW 3.3.10 (AVX2) | commercial FFTW 3.3.10 (AVX2) |
| Host compiler | Clang 19.0.0 (rfoss/rompi) | Cray cpeAMD-25.03 |
| MPI | OpenMPI 5.0.7 (GPU-aware HIP) | Cray MPICH (GPU-aware) |
| SIMD | AVX2_256 | AVX2_256 |
| Module root | `/cvmfs/software.eessi.io/.../2025.06/` | `/users/joglekar/EasyBuild/SW/LUMI-25.03/G/` |
| Build site | EESSI CI (CVMFS-delivered) | local user EasyBuild on LUMI |

**Identical between stacks:** GROMACS major+minor, SYCL/AdaptiveCpp/HIP
backend family, GPU FFT library, SIMD width, mdrun flags, run protocol,
MI250X hardware, CPU binding, OpenMP thread count.

**Different between stacks:** AdaptiveCpp minor (25.10 vs 25.02), ROCm
patch (6.4.1 vs 6.3.4), host compiler, MPI implementation, software-stack
root (CVMFS vs local EasyBuild prefix). These are exactly the differences
the EESSI portability claim is meant to absorb.

Run nodes: EESSI on `nid007966` (x1405c7s1b0n0), native on `nid005024`
(x1100c1s4b0n0) — different cabinets, identical hardware spec, identical
`rocm-smi` NUMA-to-GCD topology.

## 2. Run statistics — 6 recorded runs each (1 warm-up discarded)

| Stack | mean (ns/day) | sd | CV | 95% CI | range | wall (s) |
|---|---|---|---|---|---|---|
| **Native** (cpeAMD-25.03-VkFFT-rocm) | **95.365** | 0.673 | **0.71 %** | ±0.538 | 94.58 – 96.18 | 143.7 – 146.2 |
| **EESSI** (rfoss-2025a-SYCL) | **92.949** | 0.353 | **0.38 %** | ±0.283 | 92.40 – 93.35 | 148.1 – 149.6 |

Per-run ns/day:

```
Native: 95.745, 96.183, 95.854, 94.608, 94.580, 95.222
EESSI:  93.209, 92.402, 92.742, 93.347, 93.157, 92.836
```

Methodology:
- 6 recorded runs + 1 discarded warm-up per job — matches the AMD ROCm Blog
  LUMI guide ("averaged each configuration over 6 benchmark runs"). Páll et
  al. CUG'24 report median-of-3-or-5 with no SD, so this protocol is at the
  upper end of literature replicate counts and additionally reports SD/CI.
- Steady-state isolation: `-resetstep 20000` of `-nsteps 100000` — the first
  20 % of steps are timer-discarded; the reported ns/day reflects the
  remaining 160 ps of simulated time.
- **CV < 1 % under both stacks.** The EESSI tightness is a direct
  consequence of the `--map-by ppr:1:l3cache:PE=7` CPU binding added after
  an unpinned Crambin EESSI run showed CV ≈ 10 % (OpenMP threads floating
  across NUMA nodes between runs). Without that binding the 2.6 % EESSI–
  native gap below would have been buried in noise.

## 3. Cross-validation against the AMD ROCm Blog

The STMV input is the **exact** tarball behind the AMD blog's tables
(`amd/InfinityHub-CI/.../stmv/stmv.tar.gz`). AMD ROCm Blog Tables 5–6
(1 LUMI-G node, 8 GCDs, STMV):

| Reference build | nstlist=100 (our config) | nstlist=400 (their optimum) |
|---|---|---|
| AMD HIP-enabled (custom) | 99.3 | 111.2 |
| GROMACS 2025.4-gpu (SYCL+AdaptiveCpp) | **94.5** | 109.2 |
| **Our native (2025.1-cpeAMD-25.03-VkFFT-rocm)** | **95.4** | — |
| **Our EESSI (2025.1-rfoss-2025a-SYCL)** | **92.9** | — |

Our native build is in the same family as the blog's "GROMACS 2025.4-gpu
(SYCL)" row — both SYCL/AdaptiveCpp/VkFFT, single-node 8-GCD, identical
TPR. **Our 95.4 ns/day reproduces the blog's 94.5 ns/day within +0.9 %**,
i.e. within reporting precision. This is independent confirmation that the
LUMI EasyBuild `GROMACS-2025.1-cpeAMD-25.03-VkFFT-rocm` recipe, installed
via EasyBuild-user, produces a faithful GROMACS 2025.1.

The HIP build's advantage (+5 % at nstlist=100, +2 % at nstlist=400)
reflects the SYCL-vs-HIP NBNXM kernel gap reported "within 10–20 %" by
Páll et al. CUG'24. At single-node STMV the workload is increasingly
bandwidth- and PME-bound, diluting the kernel difference to the low end of
that range.

## 4. EESSI vs Native — headline comparison

| Metric | EESSI | Native | Δ (native − eessi) |
|---|---|---|---|
| ns/day (mean) | 92.949 | 95.365 | **+2.42** |
| Relative | 1.000 (ref) | 1.026 | **+2.60 %** |
| Wall (s, mean) | 148.7 | 145.0 | −3.8 |

**EESSI is within 2.6 % of a hand-built native cpeAMD-25.03 stack on a
paper-grade single-node STMV run.** The 95 % CIs are essentially
non-overlapping (native ≥ 94.83, EESSI ≤ 93.23), so the gap is resolved at
α = 0.05 — but it is very small in absolute terms. This is the empirical
statement that backs the EESSI portability claim on a real, GPU-saturated
MD workload at LUMI scale.

## 5. Sources of the residual 2.6 % gap

Eliminated by construction (identical between stacks): GROMACS version,
compute backend, GPU FFT library, mdrun flags, hardware, CPU binding,
OpenMP thread count, run protocol.

Remaining candidates, ranked by likely contribution:

1. **MPI implementation — Cray MPICH (native) vs OpenMPI 5.0.7 (EESSI).**
   Cray MPICH is purpose-built for the Slingshot-11/CXI fabric and uses the
   libfabric `cxi` provider natively; EESSI's OpenMPI routes intra-node GPU
   communication through UCX `rocm_ipc`. STMV's halo-exchange and PME-PP
   transfer pattern is plausibly 1–2 % sensitive to this. **Most likely
   single contributor.**
2. **Host compiler — Cray cpeAMD vs Clang 19 (EESSI).** STMV retains
   substantial PP-side CPU work even with `-bonded gpu`; Cray's
   Trento-tuned host code could account for a few %.
3. **AdaptiveCpp 25.10.0 (EESSI) vs 25.02.0-rocm6 (native).** EESSI uses
   the *newer* AdaptiveCpp yet runs slower — so this is unlikely to be the
   cause unless 25.10 introduced a gfx90a codegen regression. Worth a
   targeted test.
4. **ROCm 6.4.1 (EESSI) vs 6.3.4 (native).** EESSI again uses the *newer*
   ROCm and is slower — counter-intuitive; may indicate the cpeAMD build
   carries Cray-specific HIP-runtime patches.
5. **CVMFS first-call latency.** Largely amortised after warm-up (the
   0.38 % CV argues it is), but stat overhead could marginally penalise
   steady state.

Candidate (1) is cheaply testable by re-running native under a non-Cray
MPI. (3)/(4) would require rebuilding EESSI against older AdaptiveCpp/ROCm
— future work.

## 6. Claims supported for the thesis

1. **The native build is correct** — independent reproduction within
   +0.9 % of an AMD-published reference number on the same input, using a
   different toolchain than the LUMI-shipped reference.
2. **EESSI portability holds within 3 % on a paper-grade MD workload** —
   same GROMACS major version, same compute backend, entirely different
   stack root, 2.6 % slower. For a CVMFS-delivered scientific software
   stack vs a hand-built local one this is an exceptionally tight delta.
3. **CPU binding is essential** for variance-controlled measurement at
   LUMI/MI250X scale — pre-binding Crambin EESSI CV ≈ 10 %, post-binding
   STMV EESSI CV = 0.38 %. A methodological contribution in its own right.
4. **The 2.6 % gap is small enough to be plausibly closed** by aligning MPI
   implementation, AdaptiveCpp version and ROCm patch level — future work.

## 7. Plot-ready summary

```
benchmark,stack,mean_ns_per_day,sd,n
stmv,eessi,92.949,0.353,6
stmv,native,95.365,0.673,6
```

Error bars: sample SD over the 6 recorded runs. The AMD blog reference
(94.5 ns/day, SYCL) and HIP build (99.3 ns/day) can be drawn as horizontal
annotation lines on the same axis for context.

## References

- AMD ROCm Blog — "Installing AMD HIP-Enabled GROMACS on HPC Systems: A
  LUMI Supercomputer Case Study." <https://rocm.blogs.amd.com/artificial-intelligence/gromacs-lumi-guide/README.html>
  (Tables 5–6, the STMV 1-node numbers cross-validated here.)
- Páll, S. et al. "GROMACS on AMD GPU-Based HPC Platforms: Using SYCL for
  Performance and Portability." CUG'24. <https://arxiv.org/abs/2405.01420>
- AMD InfinityHub-CI STMV tarball (the input TPR, shared with the AMD blog).
  <https://github.com/amd/InfinityHub-CI/tree/main/gromacs/docker/benchmark/stmv>
- LUMI EasyBuild documentation — GROMACS. <https://lumi-supercomputer.github.io/LUMI-EasyBuild-docs/g/GROMACS/>

See [README.md](README.md) for the full reference list and the
benchmark-suite methodology.
