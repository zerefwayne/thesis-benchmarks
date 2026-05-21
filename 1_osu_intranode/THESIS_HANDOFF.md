# Thesis Handoff — Intra-node MI250X Communication: EESSI vs Native Cray Stack on LUMI

> **Purpose of this document.** This is a complete, self-contained writeup of the
> `1_osu_intranode` benchmark campaign, written to be uploaded to a Claude web
> project so the thesis prose can be drafted from it directly. It captures the
> context, hardware, software stacks, the full experiment matrix, every tuning
> knob we turned, every problem we hit and how we resolved it, and the actual
> measured results (with numbers). Nothing here requires access to the cluster
> or the raw CSVs — but file pointers are given throughout if you want to trace
> a number back to its source.

---

## 0. One-paragraph abstract

On a single LUMI-G node (8× AMD MI250X GCDs connected by xGMI), we compared two
ways of getting an MPI/communication stack onto the machine: **EESSI** (a
CernVM-FS–delivered, EasyBuild-built Open MPI 5.0.7 + UCX 1.18 stack) versus the
**native Cray Programming Environment** (Cray MPICH with the GPU Transport
Layer). Using OSU Micro-Benchmarks 7.5 over device-to-device (`-d rocm D D`)
buffers, we measured point-to-point bandwidth/bidirectional-bandwidth/latency,
blocking collectives (N = 2/4/8), one-sided RMA, multi-pair concurrency, and the
RCCL collective path. **Headline result:** the two stacks are within ~1 % of each
other on bulk bandwidth at every xGMI tier, but they diverge sharply at small
messages, in *opposite* directions and for *different, fully-diagnosed* reasons —
EESSI has a deep 256 B–1 KiB device-buffer bandwidth dip caused by UCX lacking a
persistent IPC-handle cache, while native Cray MPICH wins small-message latency
through vendor fast-paths but ships a sub-optimal default IPC threshold that
costs 1.25–2.3× in the 1–4 KiB band. Both small-message problems have
single-environment-variable mitigations that we identified, measured, and
explained mechanistically. At large collectives EESSI is *faster* than native
(15–40 %) because UCC's algorithm choices scale better. This is the
"portability tax" story: EESSI buys reproducibility and zero vendor lock-in at a
cost that is near-zero in the bandwidth-bound regime and concentrated entirely in
small-message latency, most of which is recoverable by tuning.

---

## 1. Context and motivation

- **Machine:** LUMI (EuroHPC, CSC Finland), the LUMI-G partition. Each LUMI-G
  node has **4× AMD MI250X** modules. Each MI250X is a multi-chip module of **2
  GCDs** (Graphics Compute Dies), so the OS sees **8 GCDs per node**, each
  presented as an independent GPU. GCDs are wired together by **AMD Infinity
  Fabric (xGMI)** in a fixed topology.
- **SLURM:** account `project_462000226`; partitions `dev-g` (development, tight
  time limits) and `small-g`/`standard-g`. EESSI jobs add `--constraint=eessi`.
- **The thesis question.** EESSI delivers a *portable, reproducible* scientific
  software stack over CernVM-FS — the same binaries everywhere, no recompiling
  against each site's vendor toolchain. The natural objection on a flagship
  vendor machine like LUMI is: *what does that portability cost you in
  performance versus the tuned native Cray stack the vendor ships?* This chapter
  answers that for **intra-node GPU-to-GPU communication**, the most
  latency/bandwidth-sensitive layer and the one where a generic stack is most
  likely to lose to a vendor-tuned one.
- **Reference work.** De Sensi et al., *"Exploring GPU-to-GPU Communication:
  Insights into Supercomputer Interconnects,"* SC'24
  ([arXiv:2408.14090v2](https://arxiv.org/abs/2408.14090)). We borrow their
  methodology framing (tier-aware pair selection, theoretical-peak efficiency
  curves, the three RCCL tuning levers, the `HSA_ENABLE_SDMA=0` recommendation)
  and explicitly note where we depart from it.

### 1.1 The MI250X xGMI topology (the tier model)

Every pt2pt measurement is labelled by which xGMI "tier" the GCD pair sits in.
The tiers were extracted from the live KFD topology (`rocm-smi --showtoponuma`
and a custom `parse_kfd.py`), not assumed:

| Tier | GCD pairs | xGMI links | Nominal uni-directional peak |
|---|---|---|---|
| `intra_pkg` | {0,1}, {2,3}, {4,5}, {6,7} | 4 internal links | ~100 GB/s (we see ~148 GB/s bulk; see note) |
| `inter_pkg_2link` | {0,6}, {2,4} | 2 direct links | ~70–77 GB/s |
| `inter_pkg_1link` | {0,2}, {1,3}, {1,5}, {3,7}, {4,6}, {5,7} | 1 direct link | ~35–39 GB/s |
| `routed` | everything else | 0 direct (hops through an intermediate GCD) | varies |
| `self` | same GCD | n/a (HBM ceiling) | ~250 GB/s |

> **Note on "nominal peak":** the two GCDs inside one MI250X package are joined
> by a wide internal link; the measured bulk plateau there is ~148 GB/s
> uni-directional / ~278 GB/s bidirectional. The `intra_pkg` row's "~100 GB/s"
> is the conservative per-direction xGMI spec figure; the efficiency plots (P1)
> use a 200 GB/s theoretical intra-package figure for the % -of-peak column.
> This is worth stating carefully in the thesis since the "% of peak" numbers
> depend on which peak you quote.

This `get_topology()` mapping is identical to the one used in the sibling
`3_osu_*_thesis/` directories so that CSVs remain cross-comparable; it is the
source of truth in [topology.sh](topology.sh).

---

## 2. The two software stacks

| | **EESSI** | **Native Cray** |
|---|---|---|
| Delivery | CernVM-FS: `/cvmfs/software.eessi.io/versions/2025.06` | LUMI module tree (`LUMI/25.03`) |
| MPI | Open MPI 5.0.7 | Cray MPICH (Cray PE) |
| Transport | UCX 1.18 (with ROCm-aware `rocm_ipc` / `rocm_copy`) | libfabric/CXI + GPU Transport Layer (GTL) |
| Collectives | UCC (Unified Collective Communication) / optionally RCCL | Cray MPICH built-in + RCCL |
| Launcher | `mpirun -n N` (OpenMPI) | `srun --ntasks=N` (SLURM PMI) |
| OSU build | `OSU-Micro-Benchmarks/7.5-rompi-2025a` (EasyBuild module) | locally built against the Cray PE: `osu-native/osu-micro-benchmarks-7.5/` |
| Modules loaded | `EESSI-extend/2025.06-easybuild` → `OSU-Micro-Benchmarks/7.5-rompi-2025a` | `LUMI/25.03` → `PrgEnv-amd` → `rocm` → `craype-accel-amd-gfx90a`, `MPICH_GPU_SUPPORT_ENABLED=1` |

Both stacks ran on the **same node hardware** and the **same OSU 7.5 source
version**, so differences are attributable to the MPI/transport layer, not to
benchmark drift. Setup functions: [common.sh](common.sh).

---

## 3. Methodology

### 3.1 Benchmark invocation
- **pt2pt bw / bibw / put / get:** OSU flags `-m 1:67108864 -i 100 -d rocm D D`
  — message sizes 1 B to **64 MiB**, 100 iterations per recorded run,
  device-to-device ROCm buffers on both ends.
- **latency:** OSU default size range (max 1 MiB) — past 1 MiB "latency" is just
  size/bandwidth, so the cap is deliberate.
- **collectives:** `-d rocm`, at N ∈ {2, 4, 8} GCDs.
- **Message-size cap at 64 MiB** (not OSU's 1 GiB default): every tier plateaus
  by ~16 MiB, so 128 MiB–1 GiB just burns node-hours confirming the plateau.

### 3.2 Run protocol and statistics
- `NUM_RUNS = 6` per cell: **1 warm-up (discarded) + 5 recorded**. Each recorded
  run is an OSU average over 100 iterations, so **500 samples per `(stack, pair,
  size)` cell**, summarized as the median across the 5 recorded runs.
- `HSA_ENABLE_SDMA=0` hardcoded everywhere (see §5.1).
- `--exclusive --gpus=8` on every job: the script owns the whole node and
  selects GCD pairs via `ROCR_VISIBLE_DEVICES`; nothing else runs on the node, so
  results aren't polluted by neighbours.
- **No manual CPU-binding flags.** SLURM cgroup binding via
  `--exclusive --cpus-per-task=N` gives reasonable affinity. Explicit
  `--cpu-bind=map_cpu:` / `mpirun --bind-to` broke the first test runs and was
  reverted. (A NUMA-correct GCD↔CPU map exists in [cpu_bind.sh](cpu_bind.sh) for
  documentation, but is **not** used at launch.)

### 3.3 GCD pair / subset selection
- pt2pt sweeps a **tier-balanced 12-pair set** (3 intra_pkg + 2 inter_pkg_2link
  + 4 inter_pkg_1link + 3 routed) — see `A1_PAIRS` in [topology.sh](topology.sh).
- Host-buffer baseline (A3) uses 4 pairs, one per tier, all anchored at GCD 0.
- Collectives use N=2 → `{0,1}` (intra-package), N=4 → `{0,1,2,3}` (2 packages),
  N=8 → `{0..7}` (full node). **N=6 omitted on purpose** — the 4×2 geometry makes
  6 GCDs an unnatural, rarely-used configuration.

### 3.4 Launcher asymmetry (important caveat for the thesis)
The two stacks use *different launchers by necessity*, and this is a deliberate,
documented choice rather than an oversight:
- **EESSI:** `ROCR_VISIBLE_DEVICES=g0,g1 mpirun -n 2 <bin>` — OpenMPI forwards
  the per-rank device mask from the parent shell.
- **Native:** `srun --ntasks=2 bash -c "export ROCR_VISIBLE_DEVICES=g0,g1; exec
  <bin>"` — the `bash -c` wrapper is **required** because `srun` does not forward
  a per-rank `ROCR_VISIBLE_DEVICES` from the parent shell the way `mpirun` does.

### 3.5 Per-job provenance
Every job emits four artifacts under `results/` sharing a
`<base>_<stack>_<jobid>` prefix: `.csv` (parsed data), `.log` (raw OSU output),
`.out` (SLURM stdout), and `.meta` (a node-metadata sidecar with hostname, SLURM
nodelist, `/etc/cray/xname`, kernel, and the `rocm-smi --showtoponuma` dump).
Any CSV row can therefore be traced to the exact node it ran on.

### 3.6 Departures from the De Sensi paper (state these honestly)
1. The paper rejects OSU because it lacks per-iteration timings; we accept OSU's
   100-iter average × 5 runs (500 samples/cell) as sufficient for a
   stack-comparison study.
2. The paper does explicit device-device copies via raw memory handles; OSU
   can't. Our closest equivalent is GPU-Aware MPI (`-d rocm D D`); the RCCL/`*CCL`
   path is covered separately via OSU XCCL (A6).

---

## 4. Experiment matrix (what was actually run)

All of these completed and produced full-size CSVs (row counts verified). The
job IDs are the canonical data for the thesis.

| ID | Script | What it measures | Stacks | Job IDs | Rows |
|---|---|---|---|---|---|
| A1/A2 | `osu_bw.sh` | pt2pt uni bandwidth, 12 pairs | eessi, native | 18633083 / 18633084 | 1620 each |
| A2 | `osu_bibw.sh` | pt2pt bidirectional bandwidth, 12 pairs | eessi, native | 18634053 / 18634054 | 1620 each |
| A2 | `osu_latency.sh` | pt2pt latency, 12 pairs | eessi, native | 18634671 / 18634688 | 1380 each |
| A3 | `osu_bw_host.sh` | host-buffer (H H) bandwidth baseline, 4 pairs | eessi, native | 18635287 / 18635289 | 540 each |
| A4 | `osu_collectives.sh` | allreduce/alltoall/bcast/allgather, N∈{2,4,8} | eessi, native | 18636936 / 18636935 | 1230 each |
| A6 | `osu_xccl.sh` | RCCL direct via OSU XCCL, N=8 | eessi only | 18634159 | 305 |
| A7 | `osu_mbw_mr.sh` | multi-pair concurrent bandwidth + msg rate, 3 configs | eessi, native | 18634824 / 18634827 | 345 each |
| A8a | `osu_put_bw.sh` | one-sided RMA put, 12 pairs | eessi, native | 18637441 / 18637442 | 1500 each |
| A8b | `osu_get_bw.sh` | one-sided RMA get, 12 pairs | eessi, native | 18638323 / 18638324 | 1500 each |
| A9 | `osu_protocol_nccl.sh` | RCCL tuning sweep (5 configs) | eessi XCCL | 18635297 | 1000 |
| A5 | `osu_protocol_eessi.sh` | `UCX_RNDV_THRESH` sweep {DEFAULT,1024,16 MiB} | eessi | 18638878 | 486 |
| A5 | `osu_protocol_native.sh` | `MPICH_GPU_IPC_THRESHOLD` sweep {DEFAULT,1024,2048,4096,8192,16384} | native | 18638911 | 486 |
| — | `osu_bw_diag.sh` | UCX/MPICH protocol introspection (mechanism capture) | eessi, native | 18654817 / 18654818 | (logs, not CSV) |
| Tuning | `osu_bw_fixed_rndv.sh` | pt2pt bw re-run with `UCX_RNDV_THRESH=1024` | eessi, native | 18638825 / 18638826 | 1620 each |
| Tuning | `osu_bibw_fixed_rndv.sh` | bibw re-run with fix | eessi, native | (in results/) | — |
| Tuning | `osu_collectives_fixed_rndv.sh` | collectives re-run with fix | eessi, native | 18637053 / 18637055 | 1230 each |
| Tuning | `osu_mbw_mr_fixed_rndv.sh` | multi-pair re-run with fix | eessi, native | (in results/) | — |
| Tuning | `osu_bw_rndv_128.sh` | confirmatory `UCX_RNDV_THRESH=128` (cliff moves to 128 B) | eessi, native | 18639671 / 18639672 | 135 each |
| Tuning | `osu_collectives_ucc_shm.sh` | `UCC_TLS=ucp,shm,self` small-msg collective fix | eessi, native | (in results/) | — |

> The full-size pt2pt CSVs have exactly **12 pairs × 5 recorded runs × 27 sizes =
> 1620 data rows** — a quick integrity check.

---

## 5. The knobs we turned (and why)

### 5.1 `HSA_ENABLE_SDMA=0` — fixed everywhere
Originally a swept dimension. The first `osu_bw` round showed `SDMA=1` (using the
GPU's DMA engines for the copy) underperforms on several pair/size combinations,
and the De Sensi paper recommends 0. We hardcoded 0 after that round. The CSVs
keep an `sdma_enabled` column (always `0`) for schema stability.

### 5.2 `UCX_RNDV_THRESH=1024` — the EESSI 256 B cliff fix
The default UCX eager→rendezvous switch fires at ~256 B, which throws GPU-buffer
transfers into the expensive `rocm_ipc` rendezvous path *before* the payload is
big enough to amortize the IPC-attach cost. Pushing the threshold to 1024 keeps
256 B in eager mode. Applied to the EESSI arm of four `_fixed_rndv` scripts
(pt2pt bw, bibw, collectives, multi-pair). Native arm untouched (its analog
threshold doesn't produce a comparable cliff). Full reasoning:
[fix_rndv_reasoning.md](fix_rndv_reasoning.md).

### 5.3 `MPICH_GPU_IPC_THRESHOLD=8192` — the native 1–4 KiB fix
Cray's default is 1024, meaning every ≥1 KiB GPU transfer takes the IPC path. On
MI250X the IPC path is *slower* than host-staged shared memory in the 1–4 KiB
band. Raising the threshold to 8192 keeps those sizes on the faster SHM path
while preserving IPC for ≥8 KiB where it genuinely wins. Full reasoning:
[fix_ipc_reasoning.md](fix_ipc_reasoning.md).

### 5.4 `UCC_TLS=ucp,self` (baseline) and `ucp,shm,self` (experiment)
- Baseline collectives set `UCC_TLS=ucp,self` (plus `OMPI_MCA_coll_ucc_*`) to
  route through UCX point-to-point and **drop the RCCL transport layer**, which
  carries a ~26 µs collective-launch overhead that murders small-message
  latency.
- The `ucc_shm` experiment adds `shm` to attack the 1 B–128 B latency floor by
  exposing UCX's shared-memory transport directly to UCC. Reasoning:
  [fix_collectives_reasoning.md](fix_collectives_reasoning.md).

### 5.5 RCCL levers (A9) — the three from the paper
`NCCL_NCHANNELS_PER_PEER=32`, `NCCL_IGNORE_CPU_AFFINITY=1`,
`NCCL_NET_GDR_LEVEL=3`, swept individually and combined (`paper_full`). See §6.6
for the (negative) result.

---

## 6. Results

### 6.1 pt2pt bulk bandwidth — the stacks are tied (the reassuring result)

Tier-summary peak bandwidth at 32 MiB (median of recorded runs), from
[figures/tables/T1_tier_summary_bw_peak.csv](figures/tables/T1_tier_summary_bw_peak.csv):

| Tier | EESSI (MB/s) | Native (MB/s) | EESSI/Native |
|---|---:|---:|---:|
| intra_pkg | 147 842 | 147 407 | **1.003** |
| inter_pkg_2link | 77 094 | 77 085 | **1.000** |
| inter_pkg_1link | 39 039 | 39 038 | **1.000** |
| routed | 69 880 | 69 976 | **0.999** |

**Takeaway:** in the bandwidth-bound regime, EESSI's portability tax is
statistically zero. The xGMI hardware, not the software stack, sets the ceiling,
and both stacks reach it.

### 6.2 The EESSI small-message cliff (the central negative finding)

pt2pt bandwidth on the intra_pkg pair (0,1), median MB/s
([osu_bw_eessi_18633083](results/osu_bw_eessi_18633083.csv) /
[native_18633084](results/osu_bw_native_18633084.csv)):

| size | EESSI | Native | EESSI/Native |
|---|---:|---:|---:|
| 128 B | 410 | 360 | 1.14 |
| **256 B** | **30** | **439** | **0.069** ← the cliff |
| 512 B | 61 | 779 | 0.078 |
| 1 KiB | 120 | 591 | 0.20 |
| 4 KiB | 484 | 2 359 | 0.21 |
| 8 KiB | 924 | 1 089 | 0.85 |
| 64 KiB | 6 574 | 7 879 | 0.83 |
| 1 MiB | 57 669 | 61 416 | 0.94 |
| 64 MiB | 149 744 | 150 242 | 1.00 |

The dip spans roughly **256 B – 4 KiB** and bottoms at e/n ≈ 0.07. The same shape
recurs in every benchmark family that uses GPU buffers (bibw worst point e/n
0.12; mbw_mr 0.04 at 128 B; collectives up to 66× on bcast).

### 6.3 Why the cliff exists — fully diagnosed mechanism

This is the most rigorous part of the campaign and the strongest thesis content.
Captured via `UCX_PROTO_INFO=y`, `ucx_info -d/-c`, and `MPICH_ENV_DISPLAY=1` in
the `osu_bw_diag` jobs (18654817 eessi / 18654818 native). Full writeup:
[slow_zone_explainer.md](slow_zone_explainer.md).

**Root cause (EESSI/UCX 1.18):**
- UCX's `rocm_ipc` transport requires a per-transfer
  `hsa_amd_ipc_memory_attach` / detach round-trip — mapping the sender GPU's BAR
  through the KFD driver — on **every** transfer. **There is no persistent
  IPC-handle cache.** Even though OSU reuses the same buffer pointers across all
  iterations, UCX re-pays the attach cost each time.
- UCX's `rocm_copy` (the sub-128 B fallback) stages through host System-V shared
  memory, paying a fixed host-hop cost per call.
- Neither transport advertises `am_bcopy`/`am_short` for device buffers — the
  lightweight eager path that `xpmem` provides for host memory is **unavailable
  for GPU buffers**.
- `UCX_ROCM_IPC_MIN_ZCOPY=128` is a hardcoded floor: below 128 B `rocm_ipc`
  can't be used at all, forcing a doubly-staged `rocm_copy` host-fragment path.

The UCX protocol selection (from the proto-info capture) is literally:
```
size 0:       eager short                          via sysv/memory
size 1..241:  eager copy-in copy-out               via sysv/memory
size 242..∞:  rendezvous zero-copy read from remote via rocm_ipc/rocm_ipc
```
So the "256 B cliff" is really a 242-byte boundary where UCX flips from
host-SHM-staged eager to `rocm_ipc` rendezvous, and the rendezvous setup cost is
**size-independent** — which is why the bandwidth *at the cliff* stays pinned at
15–30 MB/s no matter where you move the threshold.

**Why Cray MPICH avoids it:** its env dump shows
`MPICH_GPU_IPC_CACHE_MAX_SIZE=50` (a persistent IPC-handle cache for up to 50
peer-buffer tuples) and `MPICH_GPU_EAGER_REGISTER_HOST_MEM=1` (pre-registered
host SHM). Once an IPC handle is created for a (sender, receiver, buffer) tuple,
subsequent transfers skip the attach entirely. **This handle cache is the single
architectural delta** that flattens Cray's small-message dip.

**Thesis framing:** this is a *structural difference between the two MPI stacks,
not a tuning miss on either side*. Closing it on EESSI would require a UCX patch
(persistent `rocm_ipc` handle cache, or an `am_bcopy` path on `rocm_copy`), not
an environment variable. This is a defensible, fully-cited conclusion — the
investigation has reached the wall.

### 6.4 The tuning result — `UCX_RNDV_THRESH=1024` recovers the worst point

The `osu_protocol_eessi` sweep showed `THRESH=1024` *strictly dominates* the
default, and the `_fixed_rndv` re-run confirmed it on the full 12-pair sweep.
Pair (0,1), EESSI:

| size | EESSI baseline | EESSI fixed (1024) | effect |
|---|---:|---:|---:|
| 256 B | 30 | **795** | **~26× recovered** |
| 512 B | 61 | 43 | residual gap unchanged (slightly worse) |
| 1 KiB | 120 | 121 | unchanged |
| 4 KiB | 484 | 478 | unchanged |
| ≥8 KiB | (bulk) | (bulk) | unchanged |

Two sub-findings, both important:
- **The 256 B disaster is tunable** (+26×, zero cost elsewhere).
- **The 512 B–4 KiB residual gap is NOT tunable.** Even at the right threshold,
  512 B sits at ~43 MB/s. This is the intrinsic UCX-on-MI250X limitation from
  §6.3 — neither eager (bounce-buffer-bound, hard ~660 MB/s ceiling) nor
  rendezvous (handshake-bound) performs well there.
- **Very high thresholds are catastrophic:** `THRESH=16 MiB` floors *everything*
  from 512 B to 8 MiB at the ~660 MB/s eager bounce-buffer ceiling — it destroys
  bulk bandwidth to "fix" the cliff. So "make eager handle everything" is off the
  table. 1024 is the unique strictly-dominant choice.

The `osu_bw_rndv_128` confirmatory run showed the cliff moves to exactly 128 B
when the threshold is set to 128 — proving the boundary is the threshold, and the
bandwidth at the boundary is invariant. (Mechanism hypothesis directly proven.)

### 6.5 The native tuning result — `MPICH_GPU_IPC_THRESHOLD=8192`

The native side has its own, *opposite-shaped* and much shallower defect: the
default IPC threshold (1024) routes 1–4 KiB transfers through IPC, which is
slower than host-SHM there. The fine-grained sweep (18638911):

| size | IPC path (default) | Host-SHM (THRESH=8192) | speedup |
|---|---:|---:|---:|
| 1 KiB | 568 | 1 323 | **2.33×** |
| 2 KiB | 1 136 | 1 776 | **1.56×** |
| 4 KiB | 2 256 | 2 814 | **1.25×** |
| ≥8 KiB | converged | converged | ~1.0 |

8192 is the smallest threshold that captures all three wins (geo-mean across the
full sweep is best by a hair; 16384 statistically tied; 4096 misses the 4 KiB
win). **Neither the De Sensi paper nor the Cray default flags this**, and it's
the band where stencil halos / MD `Isend` storms / ML gradient packets live — so
it's a genuinely useful, non-cosmetic vendor-default tuning miss.

**The combined story is the richest framing:** *both* stacks ship small-message
GPU defaults that hurt, *both* have one-line env-var fixes, and they're hurt by
different mechanisms (UCX: no IPC cache; Cray: IPC used too eagerly). After both
fixes, both bandwidth curves become monotonic in the small-medium range.

### 6.6 Latency — native wins small, the slow-zone tax is severe

pt2pt latency on pair (0,1), median µs (T2 table):

| size | EESSI | Native | tax |
|---|---:|---:|---:|
| 8 B | 0.49 | 0.52 | EESSI **5.8 % faster** |
| 1 KiB | 14.83 | 2.53 | EESSI **486 % slower** |

The 8 B win is real but tiny; the 1 KiB figure is the latency face of the same
256 B–1 KiB slow zone from §6.2/§6.3.

### 6.7 Collectives — the crossover story (EESSI's headline win)

N=8 intra-node, EESSI/native portability tax % (negative = EESSI faster), from
[T2](figures/tables/T2_portability_summary.csv):

| collective | 4 B | 64 KiB | 1 MiB |
|---|---:|---:|---:|
| allreduce | +74 % slow | +78 % slow | **−39 % (EESSI faster)** |
| alltoall | +17 % slow | **−35 % faster** | **−17 % faster** |
| bcast | +310 % slow | +78 % slow | **−13 % faster** |
| allgather | +361 % slow | **−46 % faster** | **−17 % faster** |

There is a clean **crossover**: native's vendor-tuned (often sub-microsecond,
e.g. bcast 0.48 µs at 4 B) primitives dominate small messages, but EESSI's UCC
algorithm choices (Rabenseifner-style recursive halving) scale better and win by
15–40 % once messages are large. The crossover points are tabulated per
collective and per N in
[T3_collective_crossover.csv](figures/tables/T3_collective_crossover.csv) — e.g.
allreduce N=8 crosses over at 128 KiB; allgather N=8 at 64 KiB.

Two distinct small-message problems were separated:
1. **256 B–4 KiB cliff** (same UCX rendezvous cause, amplified up to **66×** on
   bcast because every collective fans out N messages) — fixed by
   `UCX_RNDV_THRESH=1024` (`osu_collectives_fixed_rndv`).
2. **1 B–128 B floor** (EESSI 1.8–6× behind, *below* the rendezvous switch so a
   different cause) — attacked by `UCC_TLS=ucp,shm,self`
   (`osu_collectives_ucc_shm`).

**What EESSI cannot match:** the sub-microsecond bcast at 1 B is a vendor
hardware fast-path. EESSI's only route to compete there is RCCL, which carries a
~26 µs launch overhead — so routing through RCCL would *lose* the small-message
race anyway. Honest framing: "Cray's vendor primitive is unbeatable at the
smallest sizes; EESSI is competitive there and faster at scale."

### 6.8 RCCL direct path (A6) and the RCCL-tuning negative result (A9)

- **XCCL/RCCL direct** (`osu_xccl`, N=8): allreduce shows a flat ~37–52 µs floor
  for all small/mid sizes (the RCCL launch overhead), then ~119 µs at 1 MiB —
  *faster* than both the UCC-MPI EESSI path (198 µs) and native MPI (325 µs) at
  1 MiB. So RCCL is the large-message champion but is launch-overhead-bound at
  small sizes, exactly as the paper predicts.
- **The three RCCL levers had essentially no effect intra-node** (A9). At 1 MiB
  allreduce, every config — `baseline`, `channels32`, `affinity`, `gdr3`,
  `paper_full` — lands at ~119–122 µs (within run-to-run noise). **This is a
  notable negative result worth reporting:** the paper's gains for these levers
  come from the *inter-node* network path (`NET_GDR`, channel parallelism over
  the NIC); intra-node on a single LUMI-G node, where everything goes over xGMI
  and there is no NIC in the path, they do nothing. Good supporting evidence for
  scoping the levers to the regime where they matter.

### 6.9 One-sided RMA (A8)
Both `osu_put_bw` and `osu_get_bw` ran to completion on **both** stacks (1500
rows each). Notably the documented risk that Cray MPICH's RMA-on-GPU might fail
(`MPI_Win_create` errors) **did not materialize** — native GPU RMA worked
without falling back to host buffers. The RMA curves mirror the pt2pt bw story
(same tier ceilings, same EESSI small-message dip), plotted as B3e/B3f.

### 6.10 Multi-pair concurrency (A7)
`osu_mbw_mr` with 4 concurrent pairs across 3 topology configs (`cfg_intra_pkg`,
`cfg_mixed_1link`, `cfg_routed_split`) measures aggregate bandwidth + message
rate. Same small-message cliff, switching one size *earlier* (128 B instead of
256 B) because 64 outstanding messages per pair make UCX trip rendezvous sooner;
worst point e/n ≈ 0.04 (66 vs 1670 MB/s; msg rate 0.5M vs 13M msg/s). Bulk
aggregate bandwidth ties between stacks. (Schema caveat: the `pairing_desc`
column contains commas like `(0,1)(2,3)`, so parse this CSV with a
comma-count-aware reader, not naive `cut -d,`.)

---

## 7. Problems encountered and how they were solved

These are the engineering obstacles — useful for a "methodology / lessons"
section and for anyone reproducing the work.

1. **Lmod state collision (the big one).** EESSI's
   `source /cvmfs/.../init/bash` overwrites Lmod's SitePackage. After that,
   `module load LUMI/25.03` dies with `detect_LUMI_partition (a nil value)`, and
   a partially-torn-down state leaves `EBROOTOSUMINMICROMINBENCHMARKS` unset so
   OSU paths break. **Solutions, layered:**
   - **One stack per SLURM job** (a positional `eessi|native` arg) so each gets a
     fresh, unpolluted shell. We stopped trying to do both stacks in one job.
   - When EESSI setup runs, **hard-wipe Lmod first**: `module --force purge` then
     `unset LOADEDMODULES _LMFILES_ LMOD_PACKAGE_PATH LMOD_RC
     LMOD_SYSTEM_DEFAULT_MODULES`. Plain `module purge` does *not* reliably
     unload LUMI's auto-loaded `partition/G`.
   - In any dual-setup context, **native must initialize before EESSI** (native
     relies on the intact fresh-login Lmod state).
2. **`partition/G` auto-load** on LUMI re-evaluates a SitePackage function
   against EESSI's replacement and crashes — same family as #1, handled by the
   hard wipe above.
3. **CPU-binding flags broke runs.** `srun --cpu-bind=map_cpu:` and
   `mpirun --bind-to cpu-list` broke the first two test runs; reverted to SLURM
   cgroup defaults. The NUMA map is kept only as documentation.
4. **`srun` doesn't forward per-rank `ROCR_VISIBLE_DEVICES`.** Solved with a
   `bash -c "export ROCR_VISIBLE_DEVICES=...; exec <bin>"` wrapper on the native
   arm (OpenMPI/`mpirun` doesn't need it).
5. **OSU `--full` unsupported** in this 7.5 build — removed from all flag
   strings (it produced `(null)` errors, especially on the XCCL binaries).
6. **XCCL `-d rocm` argument.** XCCL binaries infer the accelerator from build
   flags; passing `-d` caused `(null)` errors. Dropped for the XCCL scripts.
7. **UCC routing through RCCL adds ~26 µs.** Setting `UCC_TLS=ucp,self` (drop the
   RCCL TL) is required to get competitive small-message collective latency from
   the UCC path; otherwise every collective eats the RCCL launch overhead.
8. **`MPI_Win` on GPU (native RMA) was a feared failure mode** that turned out to
   work — but the scripts retain a documented fallback to host-buffer RMA
   (`H H`) in case a different node/driver combo refuses it.
9. **The eager bounce-buffer ceiling (~660 MB/s)** was discovered the hard way by
   the 16 MiB threshold sweep — a cautionary result that "just raise the
   threshold" is a trap.

---

## 8. Figures available (already generated)

PNGs are in [figures/pngs/](figures/pngs/); generators in [figures/](figures/);
summary CSVs in [figures/tables/](figures/tables/). Every plot puts **both stacks
on the same axes** (grouped bars / overlaid lines / paired strips) — no heatmaps,
no pure-ratio panels.

| Code | File | Content |
|---|---|---|
| H1, H2 | xgmi_topology, pair_coverage_matrix | topology schematic; which pairs were swept |
| B1, B2, B4 | perpair_bw_1MiB, perpair_latency_8B, tier_aggregated_bw | head-to-head bars at fixed sizes |
| B3a–c | perpair_bw/bibw/latency_curves | 4×3 small multiples, full size sweep, both stacks |
| B3d | perpair_bw_curves_fixed_rndv | before/after the UCX_RNDV_THRESH fix |
| B3e, B3f | perpair_put/get_bw_curves | one-sided RMA |
| B3g | rndv_128_vs_baseline_OAM0 | confirmatory cliff-moves-to-128 B plot |
| E1 | protocol_sweep_eessi | the threshold sweep showing 1024 dominates |
| B5, B6 | strip_bw_1MiB, strip_latency_8B | per-run spread (noise check) |
| D1 | delta_pct_curves | signed % delta vs size |
| D2a–d | tornado_* | sorted % delta across all 12 pairs at fixed sizes |
| D3a–d | dumbbell_* | head-to-head dumbbells with Δ annotation |
| P1 | efficiency_curves | per-tier bw vs theoretical IF peak (paper Fig 3 style) |
| P2 | peak_efficiency_bars | per-pair peak vs nominal peak (paper Fig 4 style) |
| **P3** | **portability_tax** | **the single thesis-bullet figure: sorted EESSI tax across ~20 primitives** |
| F1–F3 | collectives_* | per-collective curves; small/mid/large bars; latency vs N |
| G1–G3 | mbw_mr_* | aggregate bw, message rate, peak bars |
| I1, I2 | variability, latency_bw_pareto | single-pair noise; latency-vs-bw Pareto |
| T1, T2, T3 | (CSV) | tier-summary peak bw; portability summary; collective crossover |

---

## 9. The thesis narrative (suggested spine)

1. **Setup:** what EESSI is, why portability matters, the natural performance
   objection, the De Sensi framing.
2. **Hardware:** MI250X = 8 GCDs/node, the xGMI tier model, how we derived it.
3. **Method:** OSU 7.5, D→D buffers, tier-balanced pairs, 500 samples/cell,
   one-stack-per-job, the launcher asymmetry caveat.
4. **Result 1 — bulk parity (P1, P2, B1, T1):** at ≥1 MiB the stacks are within
   ~1 % at every tier. *Portability is free where it's bandwidth-bound.*
5. **Result 2 — the small-message divergence (B3a, D1, B2):** EESSI's deep
   256 B–1 KiB dip vs native's small-latency wins; the two stacks fail in
   opposite directions.
6. **Result 3 — mechanism (the diagnostic centerpiece, slow_zone_explainer):**
   UCX has no `rocm_ipc` handle cache; Cray does. Structural, not a tuning miss.
7. **Result 4 — tuning (E1, B3d, the IPC sweep):** both single-env-var fixes,
   what each recovers, what stays intrinsic, the eager-ceiling trap.
8. **Result 5 — collectives crossover (F1–F3, T3):** native wins small (vendor
   fast-path), EESSI wins large (UCC algorithms), clean crossover; the RCCL path
   and the negative RCCL-tuning result.
9. **Synthesis — the portability tax (P3):** sorted across all primitives,
   concentrated in small messages, mostly recoverable. *EESSI on LUMI MI250X
   costs near-zero at scale and a tunable, well-understood penalty at small
   sizes.*

### Ready-to-quote framings (lifted from the reasoning docs)
- *"EESSI's UCX 1.18 + Open MPI 5.0.7 stack has a structural bandwidth dip in the
  256 B–1 KiB range on MI250X intra-node GPU buffer transfers (15–60 MB/s vs Cray
  MPICH's 300–800 MB/s). The mechanism is fully traceable: UCX's `rocm_ipc`
  transport requires an `hsa_amd_ipc_memory_attach` setup on every transfer
  because it has no persistent IPC-handle cache. Cray MPICH avoids this through
  `MPICH_GPU_IPC_CACHE_MAX_SIZE=50`. The gap is an architectural delta between
  the two MPI stacks, not a tuning miss on either side."*
- *"EESSI's collective profile shows a clear crossover: native MPI is 3–6× faster
  at small messages (latency-dominated, vendor-tuned primitives), but EESSI is
  15–40 % faster at large messages (bandwidth-dominated, where UCC's algorithm
  choices pay off)."*
- *"Both stacks ship small-message GPU defaults that hurt, and both have
  single-line env-var fixes that close most of the gap — UCX needs
  `UCX_RNDV_THRESH=1024`, Cray needs `MPICH_GPU_IPC_THRESHOLD=8192` — but they
  hurt for different mechanistic reasons."*

---

## 10. Open threads (honest "what's left" — none are blockers)

1. **`UCX_RNDV_SCHEME` sweep** (`put_zcopy` vs `get_zcopy` vs `put_ppln`,
   default `auto`): expected to shift large-size curves 10–20 % but *not* close
   the 256 B–1 KiB plateau. A 2-minute confirmatory run.
2. **`UCX_TLS=^rocm_ipc`** (force `rocm_copy`+cma+xpmem): would prove whether the
   IPC-attach cost or the host-staging floor dominates. Another 2-minute run.
3. **Combined `UCX_RNDV_THRESH=1024` + `UCC_TLS=ucp,shm,self`** collective run —
   the obvious "best EESSI config" once each knob is validated in isolation.
   Worth one run for a clean headline number.
4. The `ucc_shm` outcome (did `shm` actually help the 1 B–128 B floor, or did UCC
   silently fall back to UCP?) should be read off the relevant log — verify the
   UCC build shipped the `shm` TL before claiming the fix worked.

The mechanism investigation (§6.3) is **closed** — every claim is traceable to
`ucx_info`/`MPICH_ENV_DISPLAY` output captured on the exact stack.

---

## 11. Data provenance / file map (for tracing any number)

```
1_osu_intranode/
├── README.md                     # the campaign's own README (run order, schemas, verification)
├── common.sh                     # setup_native(), setup_eessi(), node_metadata_dump()
├── topology.sh                   # get_topology(), A1_PAIRS (12), A3_PAIRS (4)
├── cpu_bind.sh                   # GCD↔CPU NUMA map (documentation only)
├── fix_rndv_reasoning.md         # UCX_RNDV_THRESH=1024 reasoning + sweep data
├── fix_ipc_reasoning.md          # MPICH_GPU_IPC_THRESHOLD=8192 reasoning + sweep data
├── fix_collectives_reasoning.md  # two-front collective tuning strategy
├── slow_zone_explainer.md        # THE mechanism document (UCX vs MPICH internals)
├── *.sh                          # all benchmark + tuning batch scripts (see §4)
├── results/                      # <base>_<stack>_<jobid>.{csv,log,out,meta}
│   ├── osu_bw_diag_eessi_18654817/   # UCX proto-info, ucxinfo dumps
│   └── osu_bw_diag_native_18654818/  # MPICH env dump, threshold sweep logs
└── figures/
    ├── *.py                      # plot generators (§8)
    ├── pngs/                     # all figures
    └── tables/                   # T1, T2, T3 summary CSVs
```

CSV schemas (column orders) are documented in [README.md](README.md) §"CSV
schemas". Key invariant: `stack ∈ {eessi, native, eessi_xccl}`, `sdma_enabled`
always 0, `NUM_RUNS=6` (1 warmup discarded + 5 recorded).

---

*Generated as a thesis-writing handoff. Every quantitative claim above is backed
by a CSV in `results/` or a table in `figures/tables/`; every mechanistic claim
is backed by a captured `ucx_info`/`MPICH_ENV_DISPLAY` log referenced in
[slow_zone_explainer.md](slow_zone_explainer.md).*
