# 7_osu_internode_thesis — paper-grade EESSI vs native OSU benchmarks on LUMI MI250X (inter-node)

The inter-node companion to [`4_osu/`](../4_osu/README.md). Mirrors that
directory's per-script structure and one-stack-per-job pattern, but every
launcher targets 2+ nodes and runs over the CXI-rebuilt EESSI stack from
[`5_osu_internode/`](../5_osu_internode/).

Reference: De Sensi et al., "Exploring GPU-to-GPU Communication: Insights into
Supercomputer Interconnects," SC'24. [arXiv:2408.14090v2](https://arxiv.org/abs/2408.14090)

## What's different from `4_osu/`

- **2-node SLURM headers everywhere** (`--nodes=2 --ntasks-per-node=N`) —
  pt2pt scripts use `--ntasks-per-node=1`, collectives / mbw_mr scale up.
- **Inter-node-only pair set** (see `topology.sh`'s `INTERNODE_PAIRS`):
  8 cross-node pairs split across two NIC-class tiers
  (4 nic_local + 4 nic_via_xgmi, one per OAM). No intra-node reference
  rows in this directory — the local-node ceiling is the responsibility
  of [`4_osu/`](../4_osu/) and [`5_osu_internode/`](../5_osu_internode/).
- **Launchers mirror the proven 5_osu_internode patterns:**
  - Native pt2pt: `srun --nodes=2 --ntasks=2 --ntasks-per-node=1 $WRAPPER …`
  - EESSI pt2pt: `mpirun -n 2 --host A:1,B:1 --map-by ppr:1:node $WRAPPER …`
  - Native collective: `srun --nodes=2 --ntasks=$N --ntasks-per-node=$((N/2)) …`
  - EESSI collective: `mpirun -n $N --map-by ppr:$((N/2)):node …`
- **EESSI side uses the CXI rebuild from 5_osu_internode.** `common.sh`
  is copied verbatim — its `setup_eessi()` exports the OFI/CXI stack
  selection (`OMPI_MCA_pml=cm`, `OMPI_MCA_mtl=ofi`, `FI_PROVIDER=cxi`,
  …) and the head-node-orted fix
  (`PRTE_MCA_ras_base_launch_orted_on_hn=1`).
- **Message-size cap at 1 MiB.** 5_osu_internode showed inter-node BW
  plateaus by ~256 KiB on the CXI path, and >= 128 MiB transfers can
  hang on the UCX CMA fallback when CXI isn't engaged. Caps applied to
  every pt2pt / RMA / protocol-sweep script.
- **Protocol sweeps differ.** `osu_protocol_eessi.sh` sweeps
  `FI_CXI_RDZV_THRESHOLD` (libfabric/CXI's analog of UCX_RNDV_THRESH);
  `osu_protocol_native.sh` sweeps `MPICH_OFI_NIC_POLICY` and
  `MPICH_GPU_IPC_THRESHOLD`.
- **Collectives at N=16 only** (2 nodes × 8 GCDs). N=32/64/128
  scale-out runs are deferred.
- **Phase 5 tuning re-runs (`osu_*_fixed_*.sh`) are deferred** — added
  once the protocol sweeps tell us which knob actually helps.

## Theoretical reference numbers

What the empirical numbers should be compared against in the writeup
(see [`plans/nested-brewing-pelican.md`](../../.claude/plans/nested-brewing-pelican.md)
for full sources):

| Metric | Value | Source |
|---|---|---|
| Slingshot-11 per-NIC unidirectional line rate | **25 GB/s** (200 Gbps) | [LUMI Network](https://docs.lumi-supercomputer.eu/hardware/network/) |
| LUMI-G aggregate per-node injection | **100 GB/s** (4 NICs) | [LUMI-G](https://docs.lumi-supercomputer.eu/hardware/lumig/) |
| Single-rank inter-node osu_bw D D (NIC-adjacent), published | **~22–25 GB/s** | SC'24, arXiv:2408.14090v2 |
| Single-rank inter-node osu_bw D D (NIC-adjacent), 5_osu_internode measured | **23.7 GB/s** @ 1 MiB | local job 18746521 |
| Inter-node D D latency, same-switch, published | **3.66 μs** | SC'24 |
| Intra-node ceiling (xGMI 4-link, @≥16 MiB) | **~150 GB/s** | 4_osu fix_rndv_reasoning.md |
| EESSI inter-node BW *before* CXI rebuild (TCP fallback) | **~2 GB/s** | local — 12× speedup over baseline |

Per-rank bandwidth published for **8 concurrent ranks per node** (mbw_mr
analog at scale) is closer to **11–11.5 GB/s** because the 4 NICs are
shared by 8 ranks. See [LUMI-G bandwidth analysis](https://hackmd.io/@mxKVWCKbQd6NvRm0h72YpQ/HJqKaZ6oa).

## Run order

```bash
# Pre-flight (run once, in 5_osu_internode/):
cd ../5_osu_internode && sbatch verify_cxi_stack.sh   # confirm 13 PASS / 0 WARN / 0 FAIL
cd -

# Phase 1: pt2pt baseline + sanity checks
sbatch osu_bw.sh         eessi    && sbatch osu_bw.sh         native
sbatch osu_bibw.sh       eessi    && sbatch osu_bibw.sh       native
sbatch osu_latency.sh    eessi    && sbatch osu_latency.sh    native
sbatch osu_bw_host.sh    eessi    && sbatch osu_bw_host.sh    native    # A3

# Phase 2: collectives + RCCL comparison
sbatch osu_collectives.sh eessi   && sbatch osu_collectives.sh native
sbatch osu_xccl.sh                                              # A6 EESSI-only

# Phase 3: deep dives (protocol thresholds + multi-pair)
sbatch osu_mbw_mr.sh     eessi    && sbatch osu_mbw_mr.sh     native    # A7
sbatch osu_protocol_eessi.sh                                            # A5 FI_CXI_RDZV sweep
sbatch osu_protocol_native.sh                                           # A5 MPICH_OFI / IPC sweep

# Phase 4: one-sided RMA
sbatch osu_put_bw.sh     eessi    && sbatch osu_put_bw.sh     native    # A8a
sbatch osu_get_bw.sh     eessi    && sbatch osu_get_bw.sh     native    # A8b

# Phase 5: tuning re-runs (DEFERRED — write after analysing the protocol sweeps)
```

## Per-script summary

| script | benchmark family | binary | pairs / N | EESSI launcher | walltime |
|---|---|---|---|---|---|
| `osu_bw.sh`         | A1 unidirectional bw | `osu_bw`         | `INTERNODE_PAIRS` (8) | mpirun --host A:1,B:1 ppr:1:node | 30 min |
| `osu_bibw.sh`       | A1 bidirectional     | `osu_bibw`       | same | same | 30 min |
| `osu_latency.sh`    | A1 ping-pong         | `osu_latency`    | same | same | 30 min |
| `osu_bw_host.sh`    | A3 H H baseline      | `osu_bw`         | `A3_HOST_PAIRS` (2)   | same | 30 min |
| `osu_put_bw.sh`     | A8a RMA put          | `osu_put_bw`     | `INTERNODE_PAIRS` (8) | same | 30 min |
| `osu_get_bw.sh`     | A8b RMA get          | `osu_get_bw`     | same | same | 30 min |
| `osu_mbw_mr.sh`     | A7 multi-pair        | `osu_mbw_mr`     | 3 ROCR configs, 8 ranks (4/node) | mpirun --host A:4,B:4 ppr:4:node | 30 min |
| `osu_collectives.sh` | A4 collectives      | `osu_{allreduce,alltoall,bcast,allgather}` | N=16 (2×8) | mpirun -n 16 ppr:8:node | 30 min |
| `osu_xccl.sh`       | A6 RCCL              | `osu_xccl_{allreduce,alltoall,broadcast,allgather}` | N=16, EESSI-only | same | 30 min |
| `osu_protocol_eessi.sh` | A5 CXI sweep      | `osu_bw`         | `inter_GCD7_GCD7`, 7 thresholds | same | 30 min |
| `osu_protocol_native.sh` | A5 MPICH sweep   | `osu_bw`         | `inter_GCD7_GCD7`, 3 NIC × 4 IPC | srun | 45 min |

## CSV schemas

All pt2pt / RMA scripts share the 12-column schema:

```
stack,sdma_enabled,pair_label,node_a,node_b,gcd_a,gcd_b,hop_class,nic_class,run,size_bytes,bandwidth_MBps|latency_us
```

(`hop_class` is always `inter_node` — column kept for cross-CSV concat with
5_osu_internode.)

Collectives:

```
stack,sdma_enabled,benchmark,num_nodes,num_gcds,run,size_bytes,latency_us
```

mbw_mr adds `msg_rate_Mps` after `bandwidth_MBps`. Protocol-sweep scripts
prepend the swept knob(s) to the schema (`fi_cxi_rdzv_threshold` for
EESSI, `mpich_ofi_nic_policy,mpich_gpu_ipc_threshold` for native).

## Sanity check after each Phase 1 run

```bash
# Expected: inter_GCD7_GCD7 at 1 MiB should be in [22000, 25000] MB/s.
awk -F, '$3 == "inter_GCD7_GCD7" && $12 == 1048576 {print $1, $3, $13}' \
    results/osu_bw_eessi_<jobid>.csv | sort -u
```

If `inter_GCD7_GCD7 @ 1 MiB ≈ 2000` MB/s → CXI fast path didn't engage;
re-check `setup_eessi`'s exports landed (look at the script's
`[setup_eessi] FI_PROVIDER=cxi …` echo line in the .log).
