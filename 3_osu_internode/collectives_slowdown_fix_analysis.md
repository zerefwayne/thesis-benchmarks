# Inter-node collective slowdown on LUMI Slingshot — diagnosis, fixes, and the RCCL path

**System:** LUMI-G (HPE Cray EX, AMD MI250X, Slingshot-11 / Cassini, libfabric `cxi`).
**Stacks:** hermetic EESSI OpenMPI 5.0.7 / OFI MTL / cxi (+ the IDC pt2pt fix) vs native Cray MPICH
vs RCCL. **Benchmark:** `osu_collectives` / `osu_xccl`, N=16 (2 nodes × 8 GCDs), `-d rocm`.
**Date:** 2026-05-21. Companion to `IDC_LATENCY_FIX_ANALYSIS.md`, `latency_fixed_analysis.md`,
`osu_collectives_analysis.md`.

---

## 1. The three bottlenecks

After the pt2pt IDC fix (`FI_CXI_DISABLE_NON_INJECT_MSG_IDC=1`), small-message collectives improved
2–10× but stayed far off native. Three distinct problems remained:

1. **alltoall 8–256 B "plateau"** — flat **~1390 µs** (native 14–23 µs → **70–100×**), while 1 B
   (~73 µs) and ≥1024 B (≈native) are fine. A localized window, not a monotonic gap.
2. **allgather** — uniformly **8–11×** off native across all sizes.
3. **large-message (≥512 KiB)** — every collective 3–8× off native.

---

## 2. Systematic elimination (what it is NOT)

All variants via `osu_collectives_tune.sh <variant>` (EESSI-only, N=16, `-d rocm`). Each knob was
verified live with `fi_info -e`.

| hypothesis | variant / job | result on alltoall 8–256 B |
|------------|---------------|----------------------------|
| **device-buffer specific?** | `diag_hostbuf` (host `H H`) | **plateau GONE → ~20–31 µs, ≈ native.** Decisive. |
| CXI software tag-matching + big req/oflow buffers | `swmatch` (18762408) | no change (~1360–1385 µs) |
| enlarged overflow/req buffers + TX size | `bigbuf` (18762409) | no change (~1400–1420 µs) |
| hybrid preemptive matching | `hybridpre` | (not needed — sw mode already null) |
| low rendezvous threshold (force RDMA) | `rdzv0` `FI_CXI_RDZV_THRESHOLD=0` (18762738) | no change (~1380 µs) |
| flow-control / LE-exhaustion (FI_LOG warn) | `diag_log` (18762737) | **no CXI flow-control/overflow events logged** |
| HAN + Rabenseifner + RX=hardware (prior) | `osu_collectives_fixed` (18750989) | identical to baseline |

**Conclusion:** it is **not** a libfabric/CXI fabric stall — no CXI knob (match mode, buffers,
rendezvous threshold) moved it, FI_LOG was silent, and host buffers (same fabric, same algorithm,
same message count) run near native. The cause is **above** libfabric, in how OMPI handles **device
memory**.

---

## 3. Root cause (confirmed): OMPI's Bruck alltoall packs GPU memory per round

OMPI `coll/tuned` uses the **Bruck** algorithm for small-message alltoall (< ~1 KiB), which performs
**data rotation/packing (a memcpy) every log₂(P) round**. On *host* memory that pack/unpack is a
cheap CPU memcpy (so `H H` is fast); on *device* memory each pack is a **GPU copy**, and across
Bruck's rounds under 16 ranks this serializes catastrophically. At ≥1024 B OMPI switches off Bruck
(pairwise/linear, no packing) → fast. This **exactly** explains the 8–256 B window and why host
buffers escape it.

**Proof — forcing pairwise (`a2a_pairwise`, `coll_tuned_alltoall_algorithm=2`, job 18762835):**

| alltoall | baseline (Bruck) | **pairwise** | native |
|----------|-----------------:|-------------:|-------:|
| 8 B | 1398 µs | **105 µs** | 14 µs |
| 64 B | 1390 µs | **105 µs** | 19 µs |
| 256 B | 1393 µs | **109 µs** | 23 µs |
| 1024 B | 74 µs | 112 µs | 72 µs |
| 8 KiB | 100 µs | 180 µs | 156 µs |

Pairwise **collapses the plateau ~13×** (1390 → 105 µs), confirming the Bruck-device-packing
diagnosis. But it is **not a clean win**: it plateaus at ~105 µs (still **~5× native**) and slightly
*regresses* 1 B and ≥1024 B. ~105 µs is the cost of 16-rank pairwise = 15 rounds of point-to-point
on this stack — the floor of OMPI's *software* alltoall.

---

## 4. The architectural ceiling

Native Cray MPICH reaches ~14–23 µs alltoall and ~6 µs allreduce because it uses **Cassini
NIC-offloaded / GPU-aware collectives** (hardware reductions, triggered operations, GTL) that the
open OpenMPI/OFI stack **cannot access**. No MCA/FI knob closes this — confirmed by the elimination
table and corroborated by the literature (Shehata et al. CCPE 2024; HPE Cray MPI SC24; "The Big
Send-off", arXiv 2504.18658; Lavely et al. CUG 2025 on the HMEM small-message penalty). UCC is a
dead end on Slingshot (no inter-node UCX transport; hangs MPI_Init, job 18750650).

**Therefore:** OMPI MPI collectives can be made *non-pathological* (force pairwise alltoall to kill
the 70–100× Bruck cliff) but **cannot match native**. Native-class GPU collectives require a
GPU-native library: **RCCL**.

---

## 5. Recommended outcome

1. **OMPI MPI collectives (mitigation, not parity):** set
   `OMPI_MCA_coll_tuned_use_dynamic_rules=1` + `OMPI_MCA_coll_tuned_alltoall_algorithm=2` (pairwise)
   to remove the Bruck-on-device cliff (1390 → ~105 µs). Document the residual ~5× as the
   software-collective ceiling. (Do **not** make rendezvous/match changes — proven no-ops.)
2. **RCCL via aws-ofi-nccl over cxi (the native-class path):** investigated end-to-end. Outcome
   below — the plugin works and RCCL runs, but the **GPUDirect-RDMA data path hangs**, so RCCL is
   not currently a win on this hermetic stack.

### RCCL / aws-ofi-nccl bring-up (Track 2B) — what happened

- **Plugin built hermetically.** `aws-ofi-rccl` is deprecated → built **aws-ofi-nccl v1.19.2**
  (`/users/joglekar/code/aws-ofi-nccl`) via `build_aws_ofi_nccl.sh` (job 18763341) against the
  hermetic `libfabric-1.22.0` (cxi) + HIP 6.4.1 + hwloc → `librccl-net.so` in
  `/users/joglekar/eessi/aws-ofi-nccl-scratch/lib`. (RCCL not needed at build time; dlopened at run.)
- **Two `osu_xccl` config bugs found & fixed.** (a) No `NCCL_*` env → RCCL had no inter-node
  transport (job 18751009/18762372 "unhandled system error"). Added `NCCL_SOCKET_IFNAME=hsn0..3`,
  `FI_PROVIDER=cxi`, `NCCL_NET_GDR_LEVEL=3`, `NCCL_NET_PLUGIN=…/librccl-net.so`. (b) The wrapper
  isolated each rank to one GPU via `ROCR_VISIBLE_DEVICES=$local_rank`, but RCCL does its own
  `hipSetDevice(local_rank)` → all ranks landed on one GPU → "Duplicate GPU detected" abort
  (job 18763362/18763445). Fix: expose all 8 GCDs (`ROCR_VISIBLE_DEVICES=0..7`).
- **Now RCCL initializes fully:** plugin loads, "NET/OFI Initializing aws-ofi-nccl 1.19.2",
  "Selected provider is cxi (4 nics)", distinct GPUs, "GPU Direct RDMA Enabled for cxi0".
- **But the GPUDirect data path HANGS** (probe `xccl_debug.sh`, allreduce 1:8192, 90 s timeout):

  | RCCL config | result |
  |-------------|--------|
  | `NCCL_NET_GDR_LEVEL=0` (host staging) | ✅ completes — but allreduce ~43–68 µs (≈ OMPI MPI, **not** better) |
  | GDR on (dmabuf, default) | ❌ hang (124) |
  | GDR on + `OFI_NCCL_DISABLE_DMABUF=1` | ❌ hang (124) |
  | GDR on + `OFI_NCCL_GDR_FLUSH_DISABLE=1` | ❌ hang (124) |
  | GDR on + `NCCL_DMABUF_ENABLE=1` (dmabuf path) | ❌ EINVAL on MR registration (no hang) |

- **Pinpointed root cause — the cxi provider cannot register ROCm GPU memory for true RDMA.**
  RCCL has two GPUDirect registration paths and **both fail at the libfabric/cxi layer**:
  - *Legacy peer-direct* (default; "Dmabuf feature disabled without NCCL_DMABUF_ENABLE=1"): connections
    establish, "GPU Direct RDMA Enabled", then the inter-node RDMA-to-GPU **hangs** (no completion).
  - *dmabuf* (`NCCL_DMABUF_ENABLE=1`; kernel has `CONFIG_DMABUF_MOVE_NOTIFY=y`, libfabric advertises
    "DMA-BUF registrations: true"): `fi_mr_regattr` is **rejected with `-FI_EINVAL` (-22)** —
    "Couldn't register memory region with regattr … Invalid argument" (type=2 ROCm). No hang, fast fail.
- **Why OpenMPI's `-d rocm` works but RCCL's GDR doesn't:** OpenMPI's OFI MTL moves device buffers via
  the **HMEM copy/staging** path (GPU↔host bounce), never requiring true NIC↔GPU RDMA — so it gets
  full bandwidth without exercising the broken path. RCCL's GDR demands real RDMA-to-GPU, which the
  SHS 12.0.2 cxi provider in this from-source build does not support for ROCm memory.
- **Implication for the rebuild idea:** the build *already* has `--with-rocr` and *already* advertises
  dmabuf; there is **no identified missing GDR/HMEM configure flag** to add. The `EINVAL` is a cxi
  provider MR-support limitation, not a build-option gap. A plain "rebuild with GDR opts" has no clear
  lever. Realistic remaining paths: (a) a different shs-libfabric point release whose cxi provider
  implements ROCm dmabuf MR (risking kernel-UAPI mismatch — see the CXI_MAP_IOVA_ALLOC note);
  (b) the `libcxi-1.0.2` "library-only" build may omit GDR/peer-mem hooks the provider needs — revisit;
  (c) raise with LUMI support / use their supported aws-ofi-rccl stack for the GDR path;
  (d) accept as a documented limitation.
- **Net:** RCCL host-staging works but is no faster than MPI; **true GPUDirect RDMA over the hermetic
  cxi stack is non-functional for ROCm buffers**. The transport, GPU-binding, and plugin are all
  correct — the gap is GPU-memory RDMA *registration*.

### Conclusive: all three cxi GPU-registration modes fail (the libcxi/libfabric angle is exhausted)

Inspected the source (cloned `shs-libcxi`, `shs-libfabric` @ release/shs-12.0.2):
- **libcxi rebuild is a dead end.** `shs-libcxi` `Makefile.am` puts *all* GPU code
  (`utils_gpu_hip.c`, `tests/libcxi_gpu_hip.c`, …) in `utils/`+`tests/`, **none in `libcxi.so`**.
  The easyconfig's `--without-rocm` only drops the (stripped) tools; `libcxi.so` has no GPU MR code
  to gain. Rebuilding `--with-rocm` changes nothing.
- **The cxi provider implements every GPU-RDMA registration path** (`prov/cxi/src/cxip_iomm.c`:
  dmabuf-from-pointer, app-supplied dmabuf, and the ATS/direct-MAP fallback; knob
  `FI_CXI_DISABLE_DMABUF_ROCR`, default off). **All three fail on these nodes:**

  | cxi GPU-memory registration mode | how triggered | result |
  |----------------------------------|---------------|--------|
  | dmabuf-from-pointer (provider default) | GDR on, defaults | hang (124) |
  | app-exported dmabuf | `NCCL_DMABUF_ENABLE=1` | `fi_mr_regattr` → EINVAL |
  | ATS / direct cxi MAP | `FI_CXI_DISABLE_DMABUF_ROCR=1` | hang (124) |

- **Verdict.** The failure is **below our user-space build**, at the cxi-kernel-module ↔ amdgpu
  peer-memory/dmabuf/ATS integration on LUMI compute nodes: registration either stalls (no RDMA
  completion) or is rejected (EINVAL). A from-source SHS-12.0.2-matched stack reproduces it in all
  modes, so no `libcxi`/`libfabric` rebuild we control fixes it. True GPUDirect RDMA for ROCm over cxi
  is a **platform/vendor-integration limitation** — needs LUMI's supported RCCL stack or a LUMI-support
  ticket. **For the thesis: this is a clean, well-localized negative result** — EESSI MPI (OpenMPI/OFI)
  and RCCL both function over cxi via HMEM host-staging; true NIC↔GPU GPUDirect RDMA does not engage
  from a hermetic from-source stack.
  Jobs: 18763341 (plugin build), 18763798 (gdr0 host-staging ✅), 18763838/18763888/18764185 (legacy GDR
  hang), 18764506 (app-dmabuf EINVAL), 18764778 (ATS/MAP hang). Source clones: `~/code/shs-libcxi`,
  `~/code/shs-libfabric`, `~/code/aws-ofi-nccl`. Scratch plugin: `~/eessi/aws-ofi-nccl-scratch`.

---

## 6. Files & jobs
- Sweep harness: `osu_collectives_tune.sh` (variants: diag_hostbuf, diag_log, swmatch, bigbuf,
  hybridpre, rdzv0/rdzv_low/rdzv_eager0, a2a_pairwise/bruck/linsync, ag_*).
- Jobs: hostbuf 18762371; swmatch 18762408; bigbuf 18762409; diag_log 18762737; rdzv0 18762738;
  a2a_pairwise 18762835. Baselines: MPI 18750788/18761906, native 18750651, failed RCCL 18751009/18762372.

## 7. References
- fi_cxi(7) — https://ofiwg.github.io/libfabric/v2.1.0/man/fi_cxi.7.html
- OpenMPI coll-tuned (algorithm IDs) — https://docs.open-mpi.org/en/v5.0.x/tuning-apps/coll-tuned.html
- Shehata et al., "Bringing HPE Slingshot 11 support to Open MPI", CCPE 2024 — https://www.osti.gov/servlets/purl/2438730
- "The Big Send-off: High-Performance Collectives on GPU Supercomputers" — https://arxiv.org/html/2504.18658v1
- Lavely et al., CUG 2025, "MPI optimization for Slingshot" (HMEM small-msg) — https://cug.org/proceedings/cug2025_proceedings/includes/files/pap156s2-file1.pdf
- HPE Cray MPI update SC24 (NIC-offloaded collectives) — https://www.mpich.org/static/docs/slides/2024-sc-bof/hpe.pdf
- ROCm/aws-ofi-rccl — https://github.com/ROCm/aws-ofi-rccl ; LUMI RCCL tips — https://klust.github.io/LUMI-tips-and-tricks/06_ROCm/06_01_RCCL/
- OpenMPI issues #12547/#12602 (coll_tuned alltoall params), #13048 (D2D LinkX AMD)
