# RCCL over Slingshot/cxi on LUMI: GPUDirect-RDMA bring-up and its hard floor

**System:** LUMI-G (HPE Cray EX, AMD MI250X / 8 GCDs per node), HPE Slingshot-11 (Cassini NICs),
libfabric `cxi` provider. **Stack:** hermetic EESSI (from-source `libfabric-1.22.0` shs-12.0.2 cxi +
`libcxi-1.0.2` + RCCL 2.22.3), `aws-ofi-nccl` plugin built from source. **Benchmark:**
`osu_xccl` (RCCL collectives), 2 nodes × 8 GCDs, device buffers. **Date:** 2026-05-21.

This is the RCCL counterpart to `collectives_slowdown_fix_analysis.md` (which covers the OpenMPI
MPI-collective side). Short version: **RCCL runs over the hermetic cxi stack via HMEM host-staging,
but true NIC↔GPU GPUDirect RDMA for ROCm memory does not engage** — a platform/kernel-integration
limitation below the user-space build, reproduced in every registration mode.

---

## 1. Motivation

OpenMPI MPI collectives over cxi top out at ~5× native (software algorithms; no NIC offload — see the
collectives analysis). The native-class path for GPU collectives on LUMI is **RCCL**, which talks to
the fabric through the **aws-ofi-nccl** net plugin (RCCL → libfabric → cxi). Goal: get RCCL working
over the *hermetic* cxi stack and measure it against native Cray MPICH and EESSI MPI.

## 2. Bring-up: three real bugs fixed

1. **No net plugin.** EESSI ships `RCCL/2.22.3` but not the OFI net plugin, so RCCL had no inter-node
   transport ("unhandled system error" at init, jobs 18751009/18762372). `aws-ofi-rccl` is deprecated
   → built **aws-ofi-nccl v1.19.2** from source (`build_aws_ofi_nccl.sh`, job 18763341) against the
   hermetic `libfabric-1.22.0` (cxi) + HIP 6.4.1 + hwloc → `librccl-net.so`
   (`/users/joglekar/eessi/aws-ofi-nccl-scratch/lib`). RCCL needs only the plugin at runtime
   (`NCCL_NET_PLUGIN`), not at build time. Autotools flags: `--with-libfabric`, `--with-hwloc`,
   `--with-rocm`, `--disable-tests`.
2. **No RCCL network env.** Added `NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3`, `FI_PROVIDER=cxi`,
   `NCCL_NET_GDR_LEVEL=3`, `NCCL_NET_PLUGIN=.../librccl-net.so`.
3. **Duplicate-GPU abort.** The `osu_xccl` wrapper isolated each rank to one GPU
   (`ROCR_VISIBLE_DEVICES=$local_rank`), but RCCL/OSU-xccl does its **own** `hipSetDevice(local_rank)`
   → with one visible GPU all ranks bound to the same device → "Duplicate GPU detected" abort
   (jobs 18763362/18763445). Fix: expose all 8 GCDs (`ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`).

After these, RCCL initializes cleanly: plugin loads, `NET/OFI Initializing aws-ofi-nccl 1.19.2`,
`Using Libfabric version 1.22`, `Selected provider is cxi (4 nics)`, distinct GPUs,
`GPU Direct RDMA Enabled for cxi0`.

## 3. The wall: GPUDirect RDMA hangs; host-staging is no faster than MPI

Probe `xccl_debug.sh <gdr_level> [nodmabuf] [noflush] [nccl_dmabuf] [no_cxi_dmabuf_rocr]` — allreduce
1:8192, N=16, 90 s timeout so a hang shows in seconds.

| RCCL configuration | result |
|--------------------|--------|
| `NCCL_NET_GDR_LEVEL=0` (host staging) | ✅ completes — allreduce ~43–68 µs (8 B–1 KiB) |
| GDR on, default (provider dmabuf-from-pointer) | ❌ hang (timeout) |
| GDR on + `OFI_NCCL_DISABLE_DMABUF=1` | ❌ hang |
| GDR on + `OFI_NCCL_GDR_FLUSH_DISABLE=1` | ❌ hang |
| GDR on + `NCCL_DMABUF_ENABLE=1` (app-exported dmabuf) | ❌ `fi_mr_regattr` → `-FI_EINVAL` (-22) |
| GDR on + `FI_CXI_DISABLE_DMABUF_ROCR=1` (ATS / direct cxi MAP) | ❌ hang |

- **Host-staging (GDR off) is the only thing that completes** — and at ~43–68 µs it matches OMPI's
  MPI allreduce (~44 µs) and is far off native MPICH (~6–15 µs). Staging GPU→host→NIC discards RCCL's
  entire advantage, so it is not a useful result.
- Every **true-GDR** path (NIC reads/writes GPU memory directly) either hangs (no RDMA completion) or
  has its memory registration rejected with EINVAL.

## 4. Source-level evidence — the limitation is below user space

Cloned and inspected `shs-libcxi` and `shs-libfabric` at `release/shs-12.0.2` (the tags matching LUMI's
running cxi kernel module).

- **`libcxi` has no GPU MR code.** Its `Makefile.am` places every GPU source
  (`utils/utils_gpu_hip.c`, `tests/libcxi_gpu_hip.c`, …) under `utils/` and `tests/` — **none in
  `libcxi.so`**. The easyconfig's `--without-rocm` only drops the (already-stripped) diagnostic tools.
  A `libcxi --with-rocm` rebuild adds nothing to the library. **libcxi rebuild = dead end.**
- **The cxi provider implements all GPU-RDMA registration paths**
  (`shs-libfabric/prov/cxi/src/cxip_iomm.c`): dmabuf-from-pointer (`cxip_dmabuf_hints` →
  `ofi_hmem_get_dmabuf_fd`), app-supplied dmabuf (`FI_MR_DMABUF`), and an ATS/direct-MAP fallback.
  The switch `cxip_env.disable_dmabuf_rocr` (env `FI_CXI_DISABLE_DMABUF_ROCR`) defaults **false**
  (`cxip_info.c:616`), i.e. dmabuf-for-ROCr is on by default.
- **Kernel has dmabuf** (`CONFIG_DMABUF_MOVE_NOTIFY=y`), and libfabric advertises
  `NET/OFI Support for DMA-BUF registrations: true`. So the capability is *present* end to end —
  yet every concrete registration **fails at the cxi-kernel-module ↔ amdgpu peer-memory boundary**:
  the kernel either never signals RDMA completion (hang) or rejects the MR (EINVAL).

Because a from-source, SHS-12.0.2-matched user-space stack reproduces the failure in **all** modes,
no `libcxi`/`libfabric` rebuild we control can fix it. The gap is the platform's GPU-RDMA integration
(cxi driver + amdgpu peer-mem/ATS/dmabuf) on these compute nodes.

### Decisive: host (vendor Cray) libraries hang too → host_injections is NOT the fix

To test whether the gap is *our build* vs *the node*, the RCCL ranks were forced (rank-scoped
`LD_PRELOAD`) onto LUMI's **native Cray libfabric `1.22.0.1.25.0` + host libcxi `1.5.0`** — the exact
user-space stack Cray MPICH and LUMI's supported RCCL use, and host libcxi 1.5.0 is *newer* than our
from-source 1.0.2 (a candidate for the missing GDR code). Result (job 18765090, GDR on):
**still hangs (exit 124), no errors.** So swapping in the vendor libraries does not help.

This **rules out `host_injections`** as a remedy: injecting the same host libfabric/libcxi cannot fix
what preloading them already failed to fix. The limitation is **not in the user-space libraries** —
it is at the **node/runtime layer**: the GPU-RDMA path on dev-g, and/or how RCCL's cxi endpoints
acquire a Slingshot VNI. (Caveat: the preload is a mixed stack — host libfabric over the EESSI env —
so it is strong but not a perfectly clean vendor-only run.) The one remaining genuinely-different
lever, not yet tried: launch RCCL via **`srun --mpi=pmix`** (as native does) rather than OpenMPI
`mpirun`, so SLURM provisions the Slingshot VNI for the ranks' cxi endpoints — the GDR path may need
a VNI that the mpirun-launched batch step lacks (cf. the OpenMPI `PRTE_MCA_ras_base_launch_orted_on_hn`
VNI workaround in `osu-internode-cxi-rebuild`).

## 5. Why OpenMPI's `-d rocm` "works" but RCCL's GDR doesn't

OpenMPI's OFI MTL moves device buffers through the **HMEM copy/staging** path (GPU↔host bounce); it
never requests true NIC↔GPU RDMA, so it gets full bandwidth without touching the broken path. RCCL's
value *is* true GPUDirect RDMA, which is exactly what fails here. (This also means OpenMPI's good
inter-node `-d rocm` numbers are host-staged, not GPUDirect.)

## 6. Conclusion & recommendations

**Conclusion (thesis-ready negative result):** On LUMI, from a fully hermetic from-source stack, both
EESSI MPI (OpenMPI/OFI) and RCCL (aws-ofi-nccl) **function over Slingshot/cxi via HMEM host-staging**,
but **true GPUDirect RDMA for ROCm device memory does not engage** — registration hangs (dmabuf-from-
pointer, ATS) or is rejected (`EINVAL`, app dmabuf). The transport, GPU binding, plugin, and provider
code are all correct; the limitation is the cxi-kernel/amdgpu GPU-RDMA integration, below user space.

**Recommendations:**
1. **LUMI support ticket** citing the precise failure: cxi provider rejects/hangs ROCm GPU-memory MR
   in all three modes (dmabuf-from-pointer hang; `fi_mr_regattr` EINVAL with app dmabuf; ATS/`MAP` hang),
   on shs-12.0.2-matched user space, kernel `6.4.0-150600.23.73_15.0.14-cray_shasta_c`. Ask whether
   GPUDirect RDMA over cxi is enabled for ROCm on standard-g/dev-g and what registration mode is
   expected.
2. **For native-class GPU collectives now:** use LUMI's vendor-supported RCCL + aws-ofi-rccl
   module/container, which is validated against the platform's GPU-RDMA path.
3. **Thesis framing:** report the host-staging numbers (RCCL ≈ MPI ≈ ~44 µs small allreduce; native
   ~6–15 µs) and the GPUDirect limitation as a characterized boundary of the hermetic approach.

## 7. Artifacts
- Scripts: `osu_xccl.sh` (env + plugin), `build_aws_ofi_nccl.sh` (plugin build), `xccl_debug.sh` (probe).
- Plugin: `~/eessi/aws-ofi-nccl-scratch/lib/librccl-net.so`. Source clones: `~/code/aws-ofi-nccl`
  (v1.19.2), `~/code/shs-libcxi`, `~/code/shs-libfabric` (release/shs-12.0.2).
- Jobs: 18763341 (build); 18763798 (GDR-off host-staging ✅); 18763838/18763888/18764185 (legacy GDR hang);
  18764506 (app-dmabuf EINVAL); 18764778 (ATS/MAP hang).
- Related: `collectives_slowdown_fix_analysis.md`, `IDC_LATENCY_FIX_ANALYSIS.md`, `latency_fixed_analysis.md`.

## 8. References
- libfabric `fi_cxi(7)` (FI_CXI_* env, HMEM, RX match) — https://ofiwg.github.io/libfabric/v2.1.0/man/fi_cxi.7.html
- HPE `shs-libfabric` / `shs-libcxi` (release/shs-12.0.2) — https://github.com/HewlettPackard/shs-libfabric , https://github.com/HewlettPackard/shs-libcxi
- aws-ofi-nccl (ROCm/AMD support, supersedes aws-ofi-rccl) — https://github.com/aws/aws-ofi-nccl
- ROCm/aws-ofi-rccl (deprecated) — https://github.com/ROCm/aws-ofi-rccl
- LUMI RCCL tips (NCCL_SOCKET_IFNAME, GDR, plugin) — https://klust.github.io/LUMI-tips-and-tricks/06_ROCm/06_01_RCCL/
- Shehata et al., "Bringing HPE Slingshot 11 support to Open MPI", CCPE 2024 — https://www.osti.gov/servlets/purl/2438730
