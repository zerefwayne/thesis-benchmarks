# OSU Micro-Benchmarks Intra-Node Suite for LUMI standard-g

Comprehensive intra-node OSU benchmarking on a single MI250X node
(8 GCDs, 56 CPU cores, full xGMI fabric).

## Hardware assumed

LUMI `standard-g` node:
- 1× AMD EPYC 7A53 "Trento", 64 cores, 8 cores reserved for OS → 56 usable
- 8 CCDs (Core Complex Dies), 7 cores each
- 4× MI250X = 8 GCDs (Graphics Compute Dies, exposed as separate devices)
- 4× Slingshot-11 NICs

CPU↔GCD canonical binding (LUMI low-noise core per CCD):

| GCD | CCD | Cores  | Bind core |
|-----|-----|--------|-----------|
|  0  |  6  | 49-55  | 49        |
|  1  |  7  | 57-63  | 57        |
|  2  |  2  | 17-23  | 17        |
|  3  |  3  | 25-31  | 25        |
|  4  |  0  |  1-7   |  1        |
|  5  |  1  |  9-15  |  9        |
|  6  |  4  | 33-39  | 33        |
|  7  |  5  | 41-47  | 41        |

## Layout

```
osu_lumi/
├── submit_all.sh              # one-shot submission of the whole battery
├── utils/common.sh            # env setup, sourced by every script
├── scripts/
│   ├── 00_topology.sh         # rocm-smi --showtopo + UCX device list
│   ├── 01_pt2pt_pairs.sh      # all 28 GCD pairs × 4 benches × 4 placements
│   ├── 02_pt2pt_focused.sh    # quick subset hitting each topology class
│   ├── 03_collectives_8gcd.sh # full-node collectives, all 8 GCDs
│   ├── 04_collectives_scaling.sh # np=2,4,8 scaling for key collectives
│   ├── 05_onesided.sh         # put/get/acc RMA over xGMI
│   ├── 06_host_device.sh      # H<->D characterization, NUMA-local vs far
│   ├── 07_concurrent_pairs.sh # 4 simultaneous pair contention
│   └── 08_startup.sh          # MPI_Init overhead
└── results/                   # all outputs land here
```

## Quick start

```bash
cd osu_lumi
./submit_all.sh project_462000XXX
squeue -u $USER
```

Or run individual scripts:

```bash
sbatch scripts/00_topology.sh
sbatch scripts/02_pt2pt_focused.sh
```

## What each script measures

**00_topology** – Run this first. Prints `rocm-smi --showtopo` (xGMI link
count matrix), `rocminfo` HSA agents, NUMA layout, NIC list, and the
build-time UCX config. Without this you can't interpret the rest.

**01_pt2pt_pairs** – Exhaustive: every one of the 28 unordered GCD pairs,
running `osu_bw`, `osu_bibw`, `osu_latency`, `osu_mbw_mr` for D-D, H-D, D-H,
H-H. Long (~1 hour). The full xGMI bandwidth/latency map of the node.

**02_pt2pt_focused** – Faster (~20 min). One representative pair from each
topology class (intra-package, close inter-package, far inter-package,
diagonal). Use this for quick sanity checks; use 01 for the paper figure.

**03_collectives_8gcd** – Every collective in OMB 7.5 (allreduce, alltoall,
bcast, gather, scatter, neighbors, non-blocking variants) at np=8.
Compares device-buffer (D D) vs host-buffer (H H).

**04_collectives_scaling** – Same key collectives at np=2,4,8 to see how
collective performance scales with GCD count inside one node.

**05_onesided** – Put/get/acc/CAS/FOP RMA. Less commonly run, but useful
because UCX's RMA path for ROCm goes through different code than send/recv.

**06_host_device** – Studies H↔D path explicitly. Compares NUMA-local CPU
binding (canonical LUMI mapping) vs deliberately NUMA-far binding to
quantify the cost of bad CPU-to-GPU pinning.

**07_concurrent_pairs** – 4 pairs talking simultaneously via `osu_mbw_mr`
and `osu_multi_lat`. Reveals fabric contention you don't see in
isolated 2-rank tests.

**08_startup** – `osu_init` and `osu_hello`. Cheap; useful as a
ROCm-aware-MPI sanity test (large startup time often indicates
fallback to host-staging paths).

## Key environment overrides (in `utils/common.sh`)

The Option C workaround for the UCX-vs-UCX-ROCm PATH/LD_LIBRARY_PATH
collision is here:

```bash
export PATH="${EBROOTUCXMINROCM}/bin:${PATH}"
export LD_LIBRARY_PATH="${EBROOTUCXMINROCM}/lib:${LD_LIBRARY_PATH}"
```

UCX transports forced on:
```bash
export UCX_TLS=xpmem,rocm_copy,rocm_ipc,cma,sm,self
```

OpenMPI forced through UCX PML:
```bash
export OMPI_MCA_pml=ucx
```

`UCX_MEMTYPE_CACHE=n` is set because the cache occasionally misclassifies
freshly-allocated GPU buffers as host memory, especially under memory
pressure; turning it off costs a small amount of small-message latency
but produces consistent results.

## Reading the results

The headline number from your earlier sanity test was ~105 GB/s for
`osu_bw -d rocm D D` on adjacent GCDs with `rocm_ipc` enabled — that's
xGMI working as intended. Numbers to expect on LUMI MI250X intra-node:

- **D D, intra-package GCD pair**: ~150–180 GB/s for `osu_bw` at 4 MiB
- **D D, multi-link inter-package**: ~100–120 GB/s
- **D D, single-link inter-package**: ~50–70 GB/s
- **D D, no-direct-link / 2-hop**: ~25–40 GB/s
- **H D / D H, NUMA-local**: ~25–40 GB/s (PCIe gen4 limit)
- **H D / D H, NUMA-far**: noticeably lower; how much depends on the
  Infinity Fabric load
- **H H**: ~10–15 GB/s via xpmem/CMA single-copy

The exact bandwidth class for each pair is read from
`rocm-smi --showtopo` in the topology output. Cross-reference the
weight matrix against your `osu_bw` numbers to label each pair.

## Adjusting for your setup

The default `--account` in each script is `project_462000XXX`. Either:
- Pass it once via `./submit_all.sh project_462000XXX` (script rewrites
  all SBATCH lines), or
- Edit each script manually if running individually.
