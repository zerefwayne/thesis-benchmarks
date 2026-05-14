# 5_osu_internode — inter-node EESSI vs native OSU benchmarks on LUMI

Extends 4_osu's paper-grade methodology to a 2-node (2x8 GCDs) setting.
Each benchmark runs **one stack per SLURM job** (native or EESSI), pass
the stack as the first positional arg.

## Why one stack per job?

The 4_osu "native first, EESSI second within one allocation" pattern
fails on LUMI/25.03: EESSI's CVMFS init at
`/cvmfs/software.eessi.io/versions/2025.06/init/bash` re-evaluates
`partition/G.lua` against EESSI's SitePackage, which lacks
`get_user_prefix_EasyBuild`. `module purge --force` between sections is
not enough. Confirmed in `results/diagnostics_internode_18632905.log:404`.
The 5_osu_internode scripts trade away "same-allocation, same-node"
to get reliable EESSI runs.

## Working hypothesis

EESSI's `OSU-Micro-Benchmarks/7.5-rompi-2025a` ships UCX with transports
`xpmem, rocm_copy, rocm_ipc, cma, sm, self` — **no `cxi` transport**, no
libfabric CXI provider. Inter-node EESSI traffic should fall back to
TCP-over-Slingshot and run dramatically slower than native Cray MPICH
(which the 18632905 diagnostics confirmed gets ~24 GB/s on 2-node osu_bw
at 256 MiB via CXI).

## File layout

```
common.sh                          setup_native(), setup_eessi(), node_metadata_dump
cpu_bind.sh                        GCD <-> CPU map (carried for parity)
topology.sh                        get_topology() + A1_INTERNODE_PAIRS + A2_RANK_MODES

diagnostics.sh                     Phase 1: fabric/transport probe, ONE stack
osu_bw_internode.sh                A1: bandwidth, 3 pairs, ONE stack
osu_collectives_internode.sh       A2: allreduce+alltoall at N=8/N=16, ONE stack
osu_saturation_internode.sh        A3: 1-flow vs 2-flow, ONE stack
osu_protocol_internode_eessi.sh    Phase 3: EESSI knob sweep (EESSI-only by design)

results/                           auto-created
```

## Stack-tagged file naming

Each script takes `STACK=$1` (`native` or `eessi`, required) and produces
files tagged with the stack. With job-id `<jid>`, an `sbatch X.sh native`
invocation writes:

- `results/X_native_<jid>.csv` — parsed bandwidth/latency data
- `results/X_native_<jid>.log` — per-run timing + raw OSU output
- `results/X_native_<jid>.meta` — node hardware/software snapshot
- `results/X_native_<jid>.out` — copy of SLURM stdout, made by the EXIT trap
- `results/X_<jid>.out` — original SLURM stdout (kept; `%x` was the base name)

The script also calls `scontrol update job=$SLURM_JOB_ID JobName=X_native`
so `squeue` / `sacct` show the tagged name.

## Run order

```bash
# Phase 1 — diagnose first (native, then EESSI as separate jobs)
sbatch diagnostics.sh native           # ~10 min
sbatch diagnostics.sh eessi            # ~10 min
# Inspect results/diagnostics_internode_{native,eessi}_*.log to verify:
#   (a) EESSI 'ucx_info -d' does NOT list a cxi transport
#   (b) Native 'fi_info -p cxi' reports Slingshot details
#   (c) Both osu_latency 2-node smoke tests produced numbers

# Phase 2 — baselines (4 jobs total; the eessi jobs may hang at large sizes)
sbatch osu_bw_internode.sh native             # ~1h
sbatch osu_bw_internode.sh eessi              # ~1h
sbatch osu_collectives_internode.sh native    # ~1.5h
sbatch osu_collectives_internode.sh eessi     # ~1.5h
sbatch osu_saturation_internode.sh native     # ~30min
sbatch osu_saturation_internode.sh eessi      # ~30min

# Phase 3 — close the gap (EESSI-only, no STACK arg needed)
sbatch osu_protocol_internode_eessi.sh        # ~1h
```

All scripts target `dev-g`. Phase 2 jobs are independent and can queue
in parallel.

## SDMA split fallback

If a Phase 2 job hits its walltime, resubmit with SDMA as the second arg:

```bash
sbatch osu_bw_internode.sh eessi 0    # SDMA=0 only
sbatch osu_bw_internode.sh eessi 1    # SDMA=1 only
```

The default is `"0 1"` (both values).

## Headline message sizes

OSU sweeps all powers of 2 in `-m 8:268435456`. Analysis focuses on these
five:

| Size | Regime |
|---|---|
| 8 B | Pure latency |
| 16 KiB | Eager <-> rendezvous boundary |
| 1 MiB | Typical DL gradient / mid-app message |
| 16 MiB | Bulk transfer; rendezvous-dominated |
| 256 MiB | Asymptotic bandwidth ceiling |

CSVs carry every power-of-2 in the range; the five are present without
extra runs.

## CSV schemas

| Script | Columns |
|---|---|
| `osu_bw_internode.sh` | `stack, sdma_enabled, pair_label, node_a, node_b, gcd_a, gcd_b, hop_class, num_links, run, size_bytes, bandwidth_MBps` |
| `osu_collectives_internode.sh` | `stack, sdma_enabled, benchmark, num_nodes, num_gcds, run, size_bytes, latency_us` |
| `osu_saturation_internode.sh` | `stack, sdma_enabled, config_label, num_pairs, run, size_bytes, bandwidth_MBps, msg_rate_Mps` |
| `osu_protocol_internode_eessi.sh` | `stack, sdma_enabled, config_label, ucx_tls, ompi_mca_pml, ompi_mca_btl, ucx_net_devices, fi_provider, pair_label, gcd_a, gcd_b, run, size_bytes, bandwidth_MBps` |

`stack` in {`native`, `eessi`}. `sdma_enabled` in {`0`, `1`}.
`hop_class` in {`intra_node`, `inter_node`}. `num_links` is `NA` for
inter-node rows. `msg_rate_Mps` is `NA` for 1-flow rows. Knob-unset
cells in the protocol CSV are the literal string `unset`.

`NUM_RUNS=6` for A1/A2/A3 (1 warm-up + 5 recorded); `NUM_RUNS=4` for
the protocol sweep (1 warm-up + 3 recorded). Each recorded run is
OSU `-i 100` iterations.

## Methodology notes

1. **One stack per job.** Native and EESSI are now in separate
   allocations. Node-to-node variance enters the comparison but is small
   on `--exclusive` LUMI standard-g/small-g nodes (typically <2% on
   warm caches).
2. **Every OSU invocation is wrapped in `timeout 300`.** EESSI inter-node
   bandwidth at large sizes can hang if TCP fallback misbehaves; the
   timeout ensures one bad cell can't consume the whole job's walltime.
3. **Per-rank ROCR_VISIBLE_DEVICES** via wrapper script — necessary
   because each inter-node rank sees only its local GCDs. The wrapper
   branches on `SLURM_PROCID` (native srun) / `OMPI_COMM_WORLD_RANK`
   (EESSI mpirun) / `PMIX_RANK` (fallback).
4. **`--map-by ppr:1:node`** on EESSI inter-node mpirun calls forces one
   process per node. Without it, OpenMPI may pack ranks on the first
   node and never use the second.
5. **UCC tuning** (`UCC_TLS=ucp,self` etc.) is set only inside the EESSI
   branch of `osu_collectives_internode.sh`, matching
   [4_osu/osu_collectives.sh:124-132](../4_osu/osu_collectives.sh#L124-L132).
6. **Pair selection:** GCD7-GCD7 is the NIC-adjacent best case
   (rocm-smi: KFD I/O node 11 -> GCD 7 at ~200 GB/s). GCD0-GCD0 is the
   naive default. Gap between the two reveals NIC affinity effect.

## Verification

After both diagnostics runs:

```bash
# EESSI should NOT have cxi transport (confirming hypothesis):
grep -A30 'KEY DIAGNOSTIC' results/diagnostics_internode_eessi_*.log | grep -i 'cxi'

# Native should have cxi provider:
grep -A3 'fi_info -p cxi' results/diagnostics_internode_native_*.log | head -20

# Smoke tests produced numbers:
grep -A5 'osu_latency 2-node H H' results/diagnostics_internode_native_*.log
grep -A5 'osu_latency 2-node H H' results/diagnostics_internode_eessi_*.log
```

After both `osu_bw_internode.sh` runs:

```bash
# Tagged outputs exist:
ls results/osu_bw_internode_{native,eessi}_*.csv

# 1 MiB EESSI vs native on the NIC-adjacent pair, SDMA=0:
awk -F, '$1=="native" && $2==0 && $3=="inter_node_GCD7_GCD7" && $11==1048576' \
    results/osu_bw_internode_native_*.csv
awk -F, '$1=="eessi"  && $2==0 && $3=="inter_node_GCD7_GCD7" && $11==1048576' \
    results/osu_bw_internode_eessi_*.csv
# Expect native > 10x EESSI if hypothesis holds.

# Row count: 3 pairs x 5 recorded x ~25 sizes x 2 SDMA ~= 750 + 1 hdr per CSV
wc -l results/osu_bw_internode_{native,eessi}_*.csv
```

After `osu_protocol_internode_eessi.sh`:

```bash
# All 6 configs present:
awk -F, 'NR>1 {print $3}' results/osu_protocol_internode_eessi_*.csv | sort -u

# Best EESSI config at 16 MiB:
awk -F, 'NR>1 && $13==16777216 {print $3,$14}' \
    results/osu_protocol_internode_eessi_*.csv | sort -k2 -t' ' -n
```

## Known issues / caveats

- EESSI inter-node at the largest sizes (>=64 MiB) may hit the 300 s
  per-invocation timeout. CSV rows for those sizes will be missing;
  grep the `.log` for `timeout:` to confirm.
- `cxi` / `ucx_cxi` / `ompi_ofi_btl` protocol configs are long-shots —
  the EESSI UCX build likely doesn't have CXI support compiled in, so
  these will fall back to whatever auto-detect picks. Their failure
  mode is itself data for the writeup.
- `hsn0` is the typical LUMI Slingshot interface; if the diagnostics
  output shows a different name on the compute nodes, edit
  `osu_protocol_internode_eessi.sh`'s `ucx_net_hsn0` row accordingly.
- `--constraint=eessi` selects CVMFS-mounted nodes; harmless for native
  runs but required for EESSI.
