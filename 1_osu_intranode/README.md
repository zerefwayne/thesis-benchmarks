# 4_osu — paper-grade EESSI vs native OSU benchmarks on LUMI MI250X

Reference: De Sensi et al., "Exploring GPU-to-GPU Communication: Insights into
Supercomputer Interconnects," SC'24. [arXiv:2408.14090v2](https://arxiv.org/abs/2408.14090)

## What's different from `3_osu_*_thesis/`

- **One stack per job, via a positional arg.** Submit each benchmark
  twice — once per stack (`eessi`, `native`) — so each gets a fresh
  shell with no module pollution. The dual-stack-in-one-job approach
  kept hitting LUMI's `partition/G` auto-load colliding with EESSI's
  Lmod SitePackage takeover, so we stop fighting Lmod and let SLURM
  hand us clean shells. Cross-stack comparisons now span SLURM jobs
  rather than sections of one job, but node-to-node variance is small
  for intra-node measurements.
- **`HSA_ENABLE_SDMA=0` hardcoded everywhere.** Originally swept as a
  measured dimension; the sweep showed `SDMA=1` has poor performance on
  several pair/size combinations and the paper-recommended value is 0.
  Hardcoded after the first round of `osu_bw` analysis. CSVs keep the
  `sdma_enabled` column for schema stability — it's always 0.
- **Pair set is tier-balanced (12 pairs)** instead of the 19-pair union
  from 3_osu: 3 intra_pkg + 2 inter_pkg_2link + 4 inter_pkg_1link + 3 routed.
- **Stack-aware output filenames.** Scripts rename SLURM's
  `<job-name>_<jobid>.out` to `<job-name>_<stack>_<jobid>.out` at the
  end, so all four artifacts of a job share the same `<base>_<stack>_<jobid>`
  prefix. Identical CSV schemas across `eessi` / `native` runs let pairs
  of files be concat'd directly.
- **Node metadata sidecar**: every job writes
  `<jobname>_<jobid>.meta` with hostname, slurm_nodelist, /etc/cray/xname,
  rocm-smi topology. Lets any CSV row trace back to a known hardware state.
- **Message-size cap at 64 MiB (67108864).** `osu_bw` plateaus by ~16 MiB
  on every tier; going to 1 GiB just spends compute confirming the plateau.
  Caps applied to all pt2pt and one-sided bw scripts. `osu_latency` keeps
  OSU's 1 MiB default (past that "latency" is just size/bandwidth).
- **Larger benchmark suite**: A1–A5 from the original spec plus
  A6 XCCL/RCCL direct, A7 multi-pair bandwidth, A8 one-sided RMA put/get,
  A9 NCCL tuning sweep.
- **Tuning experiments produced during the iteration** — see [Tuning
  experiments](#tuning-experiments) below. These are not in 3_osu.
- **Collectives sweep N ∈ {2, 4, 8}** like 3_osu did. Updated after a
  comparison showed 4_osu's N=8-only coverage was narrower than the 3_osu
  baseline.

## Run order

```bash
# Phase 1: pt2pt baseline + sanity checks
sbatch osu_bw.sh         eessi             # ~1h each
sbatch osu_bw.sh         native
sbatch osu_bibw.sh       eessi
sbatch osu_bibw.sh       native
sbatch osu_latency.sh    eessi
sbatch osu_latency.sh    native
sbatch osu_bw_host.sh    eessi             # A3
sbatch osu_bw_host.sh    native

# Phase 2: collectives + RCCL comparison
sbatch osu_collectives.sh eessi            # ~45 min each, 3 N values × 4 binaries
sbatch osu_collectives.sh native
sbatch osu_xccl.sh                          # A6 (EESSI-only, N=8)

# Phase 3: deep dives (protocol thresholds + multi-pair)
sbatch osu_mbw_mr.sh     eessi             # A7
sbatch osu_mbw_mr.sh     native
sbatch osu_protocol_eessi.sh                # A5 (UCX_RNDV_THRESH sweep)
sbatch osu_protocol_native.sh               # A5 (MPICH_GPU_IPC_THRESHOLD sweep)
sbatch osu_protocol_nccl.sh                 # A9 (NCCL tuning levers, EESSI XCCL)

# Phase 4: one-sided RMA
sbatch osu_put_bw.sh     eessi             # A8a, ~2.5h each
sbatch osu_put_bw.sh     native
sbatch osu_get_bw.sh     eessi             # A8b
sbatch osu_get_bw.sh     native

# Phase 5: tuning re-runs (after analyzing the above)
sbatch osu_bw_fixed_rndv.sh         eessi  # apply UCX_RNDV_THRESH=1024
sbatch osu_bw_fixed_rndv.sh         native
sbatch osu_bibw_fixed_rndv.sh       eessi
sbatch osu_bibw_fixed_rndv.sh       native
sbatch osu_collectives_fixed_rndv.sh eessi
sbatch osu_collectives_fixed_rndv.sh native
sbatch osu_collectives_ucc_shm.sh   eessi  # UCC_TLS=ucp,shm,self
sbatch osu_collectives_ucc_shm.sh   native
sbatch osu_mbw_mr_fixed_rndv.sh     eessi
sbatch osu_mbw_mr_fixed_rndv.sh     native
```

## File layout

```
common.sh                 setup_native(), setup_eessi(), node_metadata_dump()
topology.sh               get_topology(), A1_PAIRS (12), A3_PAIRS (4)
cpu_bind.sh               GCD ↔ CPU map (documentation only; not used in launch)

# A1 + A2 dual-stack pt2pt (12 pairs)
osu_bw.sh                 osu_bibw.sh                osu_latency.sh

# A3 host-buffer baseline (4 pairs)
osu_bw_host.sh

# A4 intra-node collectives (N ∈ {2, 4, 8})
osu_collectives.sh        allreduce + alltoall + bcast + allgather

# A5 protocol thresholds (single pair 0,1)
osu_protocol_eessi.sh     UCX_RNDV_THRESH ∈ {1024, default, 16 MiB}
osu_protocol_native.sh    MPICH_GPU_IPC_THRESHOLD ∈ {1, default, 16 MiB}

# A6 RCCL direct via OSU XCCL (EESSI only, N=8)
osu_xccl.sh

# A7 multi-pair bandwidth (3 ROCR orderings)
osu_mbw_mr.sh

# A8 one-sided RMA (12 pairs)
osu_put_bw.sh             osu_get_bw.sh

# A9 NCCL tuning sweep (EESSI XCCL, paper Sec III-B levers)
osu_protocol_nccl.sh

# Tuning re-runs — see "Tuning experiments" below
osu_bw_fixed_rndv.sh                 osu_bibw_fixed_rndv.sh
osu_collectives_fixed_rndv.sh        osu_collectives_ucc_shm.sh
osu_mbw_mr_fixed_rndv.sh

# Reasoning docs
fix_rndv_reasoning.md           — central doc for all four _fixed_rndv variants
fix_collectives_reasoning.md    — two-front strategy for collectives (rndv + shm)

results/                  auto-created; <jobname>_<stack>_<jobid>.{csv,log,out,meta}
```

## Tuning experiments

After analyzing the baselines we identified two tunable EESSI weaknesses and
created paired re-run scripts to attack each:

### `UCX_RNDV_THRESH=1024` — the 256B-cliff fix

Applies to four `_fixed_rndv` scripts (pt2pt bw + bibw, collectives, mbw_mr).
Pushes UCX's eager → rendezvous protocol switch from the default ~256 B up to
1024 B, eliminating the 256 B small-message cliff (e/n drops as low as 0.06 on
bw, 0.12 on bibw, 0.015 on bcast). The native arm is untouched everywhere —
Cray MPICH's analog threshold doesn't produce a comparable cliff. Justified by
the `osu_protocol_eessi` sweep showing 1024 strictly dominates DEFAULT; higher
values (16 MiB) destroy bulk bandwidth by floor-clamping eager at ~660 MB/s.
Full reasoning: [fix_rndv_reasoning.md](fix_rndv_reasoning.md).

### `UCC_TLS=ucp,shm,self` — the small-msg collective fix

Single experiment in [osu_collectives_ucc_shm.sh](osu_collectives_ucc_shm.sh).
Adds UCX's shared-memory transport layer to UCC, attacking the 1B–128B
latency floor where native is ~4× faster on bcast/allgather. Hypothesis: with
`UCC_TLS=ucp,self`, UCC routes every step through UCX P2P; adding `shm`
exposes the shared-memory transport directly, removing one layer of
indirection that Cray's vendor primitive doesn't pay. Full reasoning:
[fix_collectives_reasoning.md](fix_collectives_reasoning.md).

## CSV schemas

| Script(s) | Columns |
|---|---|
| `osu_bw.sh`, `osu_bibw.sh`, `osu_bw_host.sh`, `osu_put_bw.sh`, `osu_get_bw.sh`, plus their `_fixed_rndv` variants | `stack, sdma_enabled, pair_label, gcd_a, gcd_b, tier, num_links, run, size_bytes, bandwidth_MBps` |
| `osu_latency.sh` | same but `bandwidth_MBps` → `latency_us` |
| `osu_collectives.sh`, `osu_xccl.sh`, `osu_collectives_fixed_rndv.sh`, `osu_collectives_ucc_shm.sh` | `stack, sdma_enabled, benchmark, num_gcds, run, size_bytes, latency_us` |
| `osu_protocol_eessi.sh` | …pt2pt schema + extra column `ucx_rndv_thresh` |
| `osu_protocol_native.sh` | …pt2pt schema + extra column `mpich_gpu_ipc_threshold` |
| `osu_mbw_mr.sh`, `osu_mbw_mr_fixed_rndv.sh` | `stack, sdma_enabled, config_label, pairing_desc, num_pairs, run, size_bytes, bandwidth_MBps, msg_rate_Mps` |
| `osu_protocol_nccl.sh` | `stack, sdma_enabled, config_label, nccl_nchannels, nccl_ignore_aff, nccl_net_gdr, benchmark, num_gcds, run, size_bytes, latency_us` |

`stack ∈ {eessi, native, eessi_xccl}`. `sdma_enabled` is always `0` in current
runs. `NUM_RUNS=6` (1 warmup discarded + 5 recorded). Each recorded run is OSU
`-i 100` iterations, so 500 samples per `(stack, pair, size_bytes)` cell.

CSV schemas are **identical between a baseline script and its `_fixed_rndv` /
`_ucc_shm` counterpart**, so pairs of CSVs concat cleanly for before/after
plotting.

## Methodology notes

1. **OSU flags** for pt2pt bw / put / get: `-m 1:67108864 -i 100 -d rocm D D`.
   No `--full` (this OSU 7.5 build does not support it). `osu_latency` uses
   OSU's default size range (max 1 MiB).
2. **No CPU binding flags.** The 3_osu scripts don't use them; SLURM's cgroup
   binding via `--exclusive --cpus-per-task=N` provides reasonable affinity.
   Adding `mpirun --bind-to cpu-list` or `srun --cpu-bind=map_cpu:` broke the
   first two test runs and was reverted.
3. **One stack per job.** EESSI's `source /cvmfs/.../init/bash` overwrites
   Lmod's SitePackage, after which `module load LUMI/25.03` fails with
   `detect_LUMI_partition (a nil value)`. We don't fight this — each script
   takes a stack arg and only sets up one.
4. **UCC tuning is per-script, not in `common.sh`.** Only `osu_collectives.sh`
   (and the `_fixed_rndv` / `_ucc_shm` variants) need `UCC_TLS=ucp,self` etc.
   (3_osu_eessi_thesis/osu_allreduce.sh:17-36 for rationale — drops RCCL TL to
   avoid ~26 µs launch overhead at small sizes). `osu_xccl.sh` deliberately
   does not set UCC vars (we want raw RCCL).
5. **GCD subsets for collectives:**
   - N=2 → `0,1`        (intra-package, 4-link xGMI)
   - N=4 → `0,1,2,3`    (2 full packages)
   - N=8 → `0..7`       (full node)
   - N=6 omitted on purpose (4 pkgs × 2 GCDs geometry).
6. **Caveats vs the paper**:
   - Paper rejects OSU because it lacks per-iteration timings. We accept
     that; OSU's average over 100 iters per recorded run × 5 recorded
     runs = 500 samples per cell, plotted as mean across 5 runs.
   - Paper does explicit device-device copies (memory handles). OSU can't.
     The closest OSU equivalent is GPU-Aware MPI via `-d rocm D D`. The
     `osu_xccl.sh` script covers the *CCL/RCCL path separately.

## Verification (after `osu_bw.sh eessi` and `osu_bw.sh native` complete)

```bash
# CVMFS check (eessi run)
grep "\[setup_eessi\] mpirun=/cvmfs" results/osu_bw_eessi_<jobid>.log

# Native module check (native run)
grep "\[setup_native\] srun=/usr/bin/srun" results/osu_bw_native_<jobid>.log

# Schema sanity — should be (eessi,0) and (native,0)
awk -F, 'NR>1{print $1,$2}' results/osu_bw_*_<jobid>.csv | sort -u

# Row count per CSV: 12 pairs × 5 recorded × 27 sizes = 1620 + 1 header
wc -l results/osu_bw_eessi_<jobid>.csv

# Cross-check pair (0,1) at 64 MiB — both stacks should be ~150 GB/s
awk -F, '$3=="intra_pkg_OAM0" && $9==67108864 {print $1, $10}' \
    results/osu_bw_{eessi,native}_*.csv
```

Expected key findings to verify (from the analysis so far):

- pt2pt bulk (≥ 1 MiB): `e/n ≈ 1.00` on every tier.
- pt2pt 256 B: `e/n ≈ 0.06` on `osu_bw` baseline; `≈ 1.0` after
  `osu_bw_fixed_rndv.sh`.
- collectives ≥ 128 KiB: EESSI ~15–40 % *faster* than native (the headline
  EESSI win).
- collectives 256 B: native is 12–66 × faster (bcast/alltoall); cliff
  closes after `osu_collectives_fixed_rndv.sh`.

## Known issues

- `osu_put_bw` / `osu_get_bw` on the native side may fail if Cray MPICH's
  RMA-on-GPU support isn't activated by default; if so, the .log will show
  `MPI_Win_create` errors and the CSV will be empty for `stack=native`.
  Falls back to host-buffer RMA by editing `OSU_FLAGS` to use `H H`.
- `osu_xccl_*` binaries may not exist in the EESSI OSU module if
  `--enable-xccl` wasn't a build option upstream. The script auto-skips
  missing binaries; check the .log for `[skip]` lines.
- `osu_collectives_ucc_shm.sh` depends on the UCC build shipping the `shm`
  TL. If absent, UCC silently falls back to UCP and the results match the
  baseline — verify by checking the .log for UCC TL startup messages.
