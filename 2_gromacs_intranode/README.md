# 6_gromacs — GROMACS 2025.1 EESSI vs Native on LUMI-G (1 node, 8 GCDs)

Single phase directory comparing **EESSI** (`GROMACS/2025.1-rfoss-2025a-SYCL`, AdaptiveCpp SYCL on EESSI's CVMFS stack) against the **native** LUMI build (`GROMACS/2025.1-cpeAMD-25.03-VkFFT-rocm`, AdaptiveCpp SYCL on the Cray PE + ROCm + cpeAMD-25.03 toolchain).

Same GROMACS major version, same compute backend family (AdaptiveCpp SYCL), same `mdrun` flags. The only intentional differences are: software stack root (CVMFS EESSI vs `/appl/lumi`), MPI implementation (OpenMPI vs Cray MPICH), and the FFT/ROCm/Boost stack details. Anything we see in the comparison can be attributed to the stack, not the benchmark configuration.

## Layout

| File | Purpose |
|---|---|
| `common.sh`               | Mirrors `4_osu/common.sh` — defines `setup_native()` and `setup_eessi()` as functions (no module loads at source time). The benchmark script picks one based on `STACK=$1`. Also defines `node_metadata_dump()`. **Does not** enable `set -u` — EESSI's CVMFS init references unbound vars and would crash. `setup_native` sets `MPICH_GPU_SUPPORT_ENABLED=1` **and** `GMX_FORCE_GPU_AWARE_MPI=1` — Cray MPICH supports GPU-aware MPI under the first flag, but GROMACS only auto-detects it for OpenMPI+UCX, so the second flag is mandatory or every halo exchange routes D→H→MPI→H→D. |
| `benchmark_crambin.sh`        | ~20k atoms. `--time=01:00:00`. `-update gpu` enabled. |
| `benchmark_hEGFRDimer.sh`     | ~465k atoms. `--time=02:30:00`. No `-update gpu`. |
| `benchmark_hEGFRDimerPair.sh` | ~3M atoms. `--time=03:00:00` (dev-g ceiling). No `-update gpu`. |
| `benchmark_stmv.sh`           | ~1M atoms. `--time=02:30:00`. Requires `bash fetch_benchmarks.sh` first. |
| `fetch_benchmarks.sh`     | Reconstructs `GROMACS_Benchmark_Suite/`: clones HECBioSim + MPSBench from `eth-cscs/GROMACS_Benchmark_Suite`, then downloads the STMV TPR (AMD InfinityHub-CI → Zenodo 3893789 fallback). Idempotent. Run on a **login node**. |
| `parse_results.py`        | Walks `results/*.csv`, concatenates into `results/perf.csv`, prints summary. |
| `plot.py`                 | `results/perf.csv` → `perf_bars.png` + `perf_pairs.png`. |
| `STMV_ANALYSIS.md`        | Thesis-grade writeup of the STMV EESSI-vs-native comparison + AMD ROCm Blog cross-validation. |
| `CRAMBIN_ANALYSIS.md`     | Thesis-grade writeup of the Crambin (small-system / overhead-bound) EESSI-vs-native comparison. |
| `HEGFRDIMER_ANALYSIS.md`  | Thesis-grade writeup of the hEGFRDimer (mid-size / PME-on-GPU) EESSI-vs-native comparison. |
| `GROMACS_Benchmark_Suite/`| HECBioSim + MPSBench + STMV TPR inputs. **Not committed to git** (the .tpr blobs exceed GitHub's file-size limits — STMV ~134 MB, hEGFRtetramerPair ~74 MB); rebuild it locally with `bash fetch_benchmarks.sh`. |
| `results/`                | All `.out`, `.log`, `.csv`, `.meta` outputs end up here, stack-tagged. Per-job mdrun artifacts (per-run `md.log`, `.cpt`, `.edr`, `.xtc`) land in a `<FILE_BASE>_<jobid>_work/` subdir. |

Each benchmark script is self-contained: parses `STACK=$1`, sources `common.sh`, calls the appropriate `setup_*` function, runs `NUM_RUNS=7` mdrun trials (1 warm-up discarded + 6 recorded — see "Methodology" below) with `-deffnm $WORKDIR/run<r>`, parses `Performance:` / `Time:` out of each `md.log`, writes one CSV row per recorded run. Pattern intentionally mirrors `4_osu/osu_bw.sh`.

## Why these 4 benchmarks

| Benchmark | Atoms | Role |
|---|---|---|
| Crambin | ~20k | Kernel launch overhead-bound (small-system / ADH-class regime). |
| hEGFRDimer | ~465k | PME-on-GPU sweet spot at 1 LUMI-G node. |
| STMV | ~1M | Canonical MI250X comparator (Páll et al. CUG'24; AMD ROCm Blog). |
| hEGFRDimerPair | ~3M | Memory + xGMI bandwidth-bound. |

Background and citations are in [../../../.claude/plans/let-s-move-to-gromacs-cuddly-fern.md](../../../.claude/plans/let-s-move-to-gromacs-cuddly-fern.md).

## How to run

### 1. One-time setup (login node)

```bash
cd thesis-benchmarks/2_gromacs_intranode
bash fetch_benchmarks.sh
```

Confirm the native module is loadable:
```bash
module purge
module load LUMI/25.03 partition/G GROMACS/2025.1-cpeAMD-25.03-VkFFT-rocm
gmx_mpi --version | grep -E '(GROMACS version|GPU support|FFT library|Compiler)'
```

### 2. Submit a stack × benchmark pair

Every script carries `#SBATCH --constraint=eessi` in its header — required for the EESSI side (CVMFS must be mounted on the node) and a harmless scheduling narrowing for the native side. So you only need to pass the stack:

```bash
# EESSI
sbatch benchmark_crambin.sh         eessi
sbatch benchmark_hEGFRDimer.sh      eessi
sbatch benchmark_hEGFRDimerPair.sh  eessi
sbatch benchmark_stmv.sh            eessi

# Native
sbatch benchmark_crambin.sh         native
sbatch benchmark_hEGFRDimer.sh      native
sbatch benchmark_hEGFRDimerPair.sh  native
sbatch benchmark_stmv.sh            native
```

Each job runs **NUM_RUNS=7** internally — `run1` is a discarded warm-up (cold cache / GPU JIT / MPI connection setup amortisation), `run2`…`run7` are the 6 recorded trials. mdrun outputs go to `results/<FILE_BASE>_<jobid>_work/run<r>.{log,cpt,edr,xtc}` and one CSV row per recorded run goes to `results/<FILE_BASE>_<jobid>.csv`.

### 3. Aggregate + plot

```bash
python parse_results.py     # walks results/*.csv → results/perf.csv + console summary
python plot.py              # results/perf.csv → perf_bars.png + perf_pairs.png
```

## Output convention

`<FILE_BASE>` = `gromacs_<benchmark>_<stack>`, e.g. `gromacs_crambin_native`. Each job produces under `results/`:

- `<FILE_BASE>_<jobid>.csv`       — one row per **recorded** run (6 rows total); columns `benchmark,stack,jobid,run,perf_ns_per_day,wall_s,core_s,ntmpi,toolchain`.
- `<FILE_BASE>_<jobid>.log`       — top-level run log (stack banner, per-run timing, mdrun stdout/stderr `tee`'d in).
- `<FILE_BASE>_<jobid>.meta`      — gmx_mpi version, modules loaded, hostname, kernel, rocm-smi topology.
- `<FILE_BASE>_<jobid>.out`       — SLURM stdout. The script renames it from the original `${SLURM_JOB_NAME}_<jobid>.out` (the SBATCH-time `--job-name`) at the end via an inode-level `mv`, transparent to SLURM's still-open fd. The pre-rename file no longer exists once the job finishes cleanly.
- `<FILE_BASE>_<jobid>_work/run<r>.{log,cpt,edr,xtc,…}` — full per-trial mdrun outputs (r=1..7).

## Methodology (paper-grade — AMD ROCm Blog LUMI recipe)

We follow the [AMD ROCm Blog "Installing AMD HIP-Enabled GROMACS on LUMI"](https://rocm.blogs.amd.com/artificial-intelligence/gromacs-lumi-guide/README.html) recipe verbatim, applied to both stacks. Specifically:

- **NUM_RUNS=7** per job: 1 warm-up (discarded) + 6 recorded. The 6 recorded matches the AMD blog ("averaged each configuration over 6 benchmark runs"); the warm-up absorbs cold-cache / GPU JIT / MPI connection setup. Páll et al. CUG'24 (arXiv:2405.01420) use median-of-3-or-5; our 6 sits at the upper end.
- **Steady-state measurement inside each run**: Crambin uses `-resethway` (timer reset at half-time); the bigger systems use `-resetstep 20000` of `-nsteps 100000`.
- **GPU offload**: `-nb gpu -pme gpu -bonded gpu` everywhere. `-update gpu` is explicit on Crambin; for the rest, `GMX_FORCE_UPDATE_DEFAULT_GPU=1` (from `common.sh`) defaults to GPU update when supported.
- **`OMP_NUM_THREADS=7`** (exported in `common.sh`): SLURM allocates 7 cores per task, but SMT2 makes that 14 logical threads. Without this, GROMACS oversubscribes 2× and tanks single-rank throughput — observed as a >10× native slowdown vs EESSI in the early runs.
- **CPU binding** (native only, via `srun --cpu-bind="$CPU_BIND"`): the L3-cache-complex mask from the AMD blog — each rank gets 7 cores of one L3CC, ordered to match the xGMI-near GCD numbering.
  ```
  mask_cpu:fe000000000000,fe00000000000000,fe0000,fe000000,fe,fe00,fe00000000,fe0000000000
  ```
- **Per-rank GPU selection** (native only, via `./select_gpu` wrapper): sets `ROCR_VISIBLE_DEVICES=$((LOCAL_ID % GPUS_PER_NODE))` so each rank sees exactly one GCD. The wrapper is **not** used under EESSI mpirun — under EESSI 2025.06's OpenMPI 5/PRRTE, none of the standard local-rank env vars (`SLURM_LOCALID`, `OMPI_COMM_WORLD_LOCAL_RANK`, `OMPI_COMM_WORLD_NODE_RANK`, `MPI_LOCALRANKID`) propagate to the wrapped process, so every rank falls through to `LOCAL_ID=0` and the GPU mapping collapses to `PP:0,PP:0,…,PME:0`. Without the wrapper GROMACS auto-distributes ranks across the 8 GCDs (`PP:0,PP:1,…,PME:7`), which is what we want anyway.
- **GPU-aware MPI**: EESSI gets it via UCX `rocm_ipc` / `rocm_copy` automatically. Native needs `MPICH_GPU_SUPPORT_ENABLED=1` AND `GMX_FORCE_GPU_AWARE_MPI=1` (GROMACS' auto-detect only knows OpenMPI+UCX; without the second var it staged every halo D→H→MPI→H→D).
- **AMD ROCm tuning** (both stacks, `common.sh`): `GMX_ENABLE_DIRECT_GPU_COMM=1`, `ROC_ACTIVE_WAIT_TIMEOUT=0`, `AMD_DIRECT_DISPATCH=1`. Native also gets `MPICH_SMP_SINGLE_COPY_MODE=CMA` and `MPICH_MALLOC_FALLBACK=1` for Cray MPICH intra-node tuning.

## Conventions inherited from the parent

- One stack per SLURM job — see [../../CLAUDE.md](../../CLAUDE.md). The dispatch is `STACK=$1`.
- Plotting follows [feedback_simple_comparison_plots](../../../.claude/projects/-pfs-lustrep2-users-joglekar-code/memory/feedback_simple_comparison_plots.md): grouped bars + paired dots, no heatmaps, no pure-ratio panels.
- Helpers are independent files (no symlinks across phase dirs).

## References

Sources that shaped the design of this phase — each one is load-bearing for some specific decision below. Cited inline above by short form; the full URLs and what we took from each are here.

### Performance-tuning recipe (what we copied verbatim)

1. **AMD ROCm Blog — "Installing AMD HIP-Enabled GROMACS on HPC Systems: A LUMI Supercomputer Case Study."** ROCm Blogs, 2024. <https://rocm.blogs.amd.com/artificial-intelligence/gromacs-lumi-guide/README.html>
   - Source of the **6-runs-averaged** methodology, the `-nsteps 100000 -resetstep 90000` template, the full env-var set (`OMP_NUM_THREADS=7`, `GMX_ENABLE_DIRECT_GPU_COMM=1`, `GMX_FORCE_UPDATE_DEFAULT_GPU=1`, `GMX_FORCE_GPU_AWARE_MPI=1`, `MPICH_GPU_SUPPORT_ENABLED=1`, `MPICH_SMP_SINGLE_COPY_MODE=CMA`, `MPICH_MALLOC_FALLBACK=1`, `ROC_ACTIVE_WAIT_TIMEOUT=0`, `AMD_DIRECT_DISPATCH=1`), the **L3CC `CPU_BIND` mask**, and the `select_gpu` per-rank GPU-selection wrapper.

2. **Páll, S. et al. "GROMACS on AMD GPU-Based HPC Platforms: Using SYCL for Performance and Portability."** *Proc. Cray User Group (CUG'24)*, 2024. Preprint: <https://arxiv.org/abs/2405.01420> (HTML: <https://arxiv.org/html/2405.01420v1>). ACM: <https://dl.acm.org/doi/10.1145/3725789.3725797>
   - Source of the **median-of-3-or-5 runs** comparator (we use 6 ⇒ at the upper end of literature norms). Single-/multi-node STMV results on LUMI MI250X; reports SYCL within 10–20% of HIP for the NBNXM kernel. Establishes the small/medium/large size-tier framing we mirror with Crambin/hEGFRDimer/hEGFRDimerPair.

3. **PDC/KTH — "GROMACS Performance Optimisation on AMD GPUs."** PDC Newsletter 2024 No. 2. <https://www.pdc.kth.se/about/publications/pdc-newsletter-2024-no-2/gromacs-performance-optimisation-on-amd-gpus-1.1371681>
   - Independent confirmation of the AMD-blog tunings on LUMI-G. Discusses SYCL/AdaptiveCpp build options.

4. **PDC/KTH — "GROMACS 2023: Readiness on the AMD GPU Heterogeneous Platform."** PDC Newsletter 2023 No. 1. <https://www.pdc.kth.se/about/publications/pdc-newsletter-2023-no-1/gromacs-2023-readiness-on-the-amd-gpu-heterogeneous-platform-1.1261129>
   - Earlier-version (2023) sibling of the 2024 piece — useful for noting maturation of the AMD-GROMACS path between releases.

5. **LUMI — "Efficient molecular dynamics simulations on LUMI."** LUMI blog. <https://lumi-supercomputer.eu/efficient-molecular-dynamics-simulations-on-lumi/>
   - Operational guidance (CPU/GPU binding, NUMA awareness) for MD on LUMI-G.

### Benchmark input provenance

6. **HECBioSim — "HPC Benchmarking Results."** <https://www.hecbiosim.ac.uk/access-hpc/hpc-benchmarking>
   - Source for the Crambin (~20k), hEGFRDimer (~465k), and hEGFRDimerPair (~3M) TPRs under [GROMACS_Benchmark_Suite/HECBioSim/](GROMACS_Benchmark_Suite/HECBioSim/). Reference numbers from other clusters live here.

7. **Kutzner, C. et al. "GROMACS heterogeneous parallelization benchmark info and systems (JCP)."** Zenodo, 2020. <https://zenodo.org/record/3893789>
   - Source archive for the **STMV** (~1M atoms) TPR pulled by [fetch_benchmarks.sh](fetch_benchmarks.sh). Companion to: *Kutzner, C.; Páll, S.; Fechner, M.; Esztermann, A.; de Groot, B. L.; Grubmüller, H. "More bang for your buck: Improved use of GPU nodes for GROMACS 2018." J. Comput. Chem.* 2019, 40, 2418–2431.

8. **Grubmüller group — "A free GROMACS benchmark set."** Max Planck Institute for Multidisciplinary Sciences. <https://www.mpinat.mpg.de/grubmueller/bench>
   - Sister archive to Zenodo 3893789 — hosts benchMEM, benchPEP, benchRIB. STMV itself is on the Zenodo mirror, not this page. Useful if we later add other system sizes.

### Native LUMI build provenance

9. **LUMI EasyBuild documentation — GROMACS.** <https://lumi-supercomputer.github.io/LUMI-EasyBuild-docs/g/GROMACS/>
    - Lists the available LUMI cpeAMD / cpeGNU GROMACS easyconfigs. Source for the choice of `GROMACS/2025.1-cpeAMD-25.03-VkFFT-rocm` (vs HeFFTe-rocm) — VkFFT is the faster single-GPU PME backend; HeFFTe is for multi-GPU PME decomposition (`-npme > 1`), not the current configuration.

### Adjacent context (not directly cited above, but read during design)

10. **HPCWire — "GROMACS and LUMI Supercomputer Transform Molecular Dynamics Simulations with AMD MI250X GPUs."** 2024. <https://www.hpcwire.com/off-the-wire/gromacs-and-lumi-supercomputer-transform-molecular-dynamics-simulations-with-amd-mi250x-gpus/>
    - Press write-up of the Páll et al. CUG'24 work; useful one-paragraph framing for the thesis intro.

11. **GROMACS user forum — "Benchmarking GROMACS 2023 using STMV — PME rank outside cutoff of domain decomposition."** BioExcel forum. <https://gromacs.bioexcel.eu/t/benchmarking-gromacs-2023-using-stmv-pme-rank-outside-cutoff-of-domain-decomposition/10801>
    - Discusses STMV at 8-GCD with `-npme 1` — same `-npme 1` choice as ours; documents an edge case that doesn't apply to our flag set but is worth knowing if we expand to multi-node.

### Code repositories used

12. **EasyBuild framework / easyblocks / easyconfigs.** Upstream checkouts under `code/easybuild-{framework,easyblocks,easyconfigs}/`. Treat as read-only references; the native build was produced from the LUMI cpeAMD easyconfig (see ref. 9).

13. **EESSI software-layer.** Upstream checkout under `code/software-layer/`. The EESSI module `GROMACS/2025.1-rfoss-2025a-SYCL` is delivered via CVMFS from the EESSI 2025.06 release; this is the read-only reference for what it contains.
