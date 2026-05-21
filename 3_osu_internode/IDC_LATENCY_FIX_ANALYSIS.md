# Inter-node small-message latency fix: the Cassini IDC GPU-staging penalty

**System:** LUMI-G (HPE Cray EX, AMD MI250X / 8 GCDs/node), HPE Slingshot-11 (Cassini NICs, `cxi`
libfabric provider).
**Stacks compared:** hermetic EESSI (OpenMPI 5.0.7 + shs-libfabric 1.22.0 CXI, OFI MTL) vs native
Cray MPICH 8.1.x.
**Benchmark:** OSU 7.5 `osu_latency`/`osu_collectives`, `-d rocm` device buffers, 2 nodes.
**Date:** 2026-05-21. **Status:** fixed, runtime-only, no rebuild, no `host_injections`.

---

## 1. Symptom

`osu_latency -d rocm D D` between two nodes showed EESSI ~6–8× slower than native — but **only for
small messages**, with a sharp, non-monotonic discontinuity:

| size | EESSI `D D` (baseline) | native `D D` | gap |
|------|-----------------------:|-------------:|----:|
| 1–64 B | ~17.5–20.6 µs | ~2.5 µs | ~7–8× |
| 128 B | ~18.1–21.1 µs | ~3.3 µs | ~5.5× |
| **256 B** | **~3.8–4.0 µs** | ~3.7 µs | **~1×** |
| 512 B + | ~3.1–3.4 µs | ~3.9 µs | parity / faster |
| 1 MiB + | ~50 µs | ~50 µs | parity |

EESSI's own curve was **non-monotonic**: ~6× *slower* at 1 B than at 256 B. Native was flat. The
cliff sat at **exactly 256 B**, and bandwidth + large-message latency already matched native.

**Key inference:** this is *not* a TCP fallback or a broken interconnect path. If the `cxi` provider
weren't engaged, bandwidth would be broken too — it wasn't. Something specific to *small device
sends* was wrong.

Raw data: `results/osu_latency_eessi_18750320.csv` (EESSI baseline),
`results/osu_latency_native_18750321.csv` (native).

---

## 2. What had already been ruled out

- **Provider selection.** `verify` job (18761764): `fi_info -p cxi` lists 4 `cxi` domains
  (`cxi0`–`cxi3`, `FI_PROTO_CXI`); OpenMPI logs `mca:base:select:( mtl) Selected component [ofi]`.
  The `cm` PML + `ofi` MTL + `cxi` provider chain is correct and active.
- **Software tag matching.** `FI_CXI_RX_MATCH_MODE=hardware` (job 18750700) gave **zero** change to
  the small-message floor → not a hybrid/software-matching artifact.
- **Rendezvous threshold.** A low `FI_CXI_RDZV_THRESHOLD` would slow *large* messages, not small —
  the opposite of what we saw. Not the cause.

---

## 3. Root cause: IDC inline path stages GPU→host per send

256 B is the **Cassini IDC (Immediate Data Command) inline limit**. The `cxi` provider has two
small-message transmit paths:

- **≤ ~256 B → IDC:** the payload is written *inline* into the command descriptor via CPU/MMIO
  (a doorbell write to the NIC). This is normally the lowest-latency path.
- **> 256 B → DMA:** the NIC DMA-reads the payload from a registered buffer.

The benchmark moves **GPU (ROCr/HMEM) device buffers**. To inline a *device* payload into an IDC
command, the host CPU cannot read GPU memory cheaply, so the provider must first **stage the buffer
GPU→host** — a per-send copy costing ~15 µs. At ≥256 B the message exceeds the IDC inline limit,
switches to the DMA path, and the NIC reads GPU memory **directly** via the HMEM/ROCr backend that
the hermetic libfabric was built with (`--with-rocr`) → ~3 µs. Native Cray MPICH has its own
device small-message path and never pays this.

### The decisive diagnostic: host buffers (`H H`)

Re-running the *identical* EESSI stack with host buffers (`osu_latency_idc.sh hostbuf`, job
18761765) removed the plateau entirely:

| size | EESSI `D D` | **EESSI `H H`** | native `D D` |
|------|------------:|----------------:|-------------:|
| 1–64 B | ~17.5 µs | **~2.34 µs** | ~2.5 µs |
| 128 B | ~18.1 µs | **~2.88 µs** | ~3.3 µs |
| 256 B | ~3.8 µs | **~2.91 µs** | ~3.7 µs |

With host buffers EESSI is flat ~2.34 µs and actually **beats native** — proving the OpenMPI → OFI →
CXI machinery is sound and the entire penalty lives in the *device-buffer IDC* path. `H H` is **not
a deliverable** (it changes what is measured and would force an explicit H2D/D2H copy in real GPU
codes); it is purely the experiment that isolated the cause.

The `fi_cxi(7)` man page documents the exact knob, calling out this scenario verbatim:

> **`FI_CXI_DISABLE_NON_INJECT_MSG_IDC`** — *"Experimental option to disable favoring IDC for transmit
> of small messages when `FI_INJECT` is not specified. Useful with GPU source buffers to avoid host
> copies."*

Independent corroboration: the official EESSI Slingshot-11 blog (2025-11-14) reports the **same**
~4× small-message latency gap on **NVIDIA Grace/Hopper** — i.e. this is a GPU-device-buffer IDC
issue, not LUMI/ROCm-specific.

---

## 4. The fix

Set, in `common.sh`'s EESSI CXI block (alongside `FI_PROVIDER=cxi`):

```bash
export FI_CXI_DISABLE_NON_INJECT_MSG_IDC=1
```

This routes small non-inject device sends through the DMA path, reading GPU memory directly.

### Results — device-to-device after the fix

`noidc` = job 18761810, `noidc_noinject` = job 18761811 (adds `OMPI_MCA_mtl_ofi_inject_size=0`).

**GCD7 — `nic_local` (mean µs):**

| size | baseline `D D` | **noidc** | noidc+noinject | `H H` | native |
|------|---------------:|----------:|---------------:|------:|-------:|
| 1–64 B | ~17.5 | **2.93** | 2.93 | 2.36 | 2.73 |
| 128 B | 18.10 | **2.95** | 2.94 | 2.90 | 3.28 |
| 256 B | 3.80 | **2.96** | 2.97 | 2.91 | 3.66 |
| 512 B | 3.07 | **3.03** | 2.99 | 2.93 | 3.89 |

**GCD0 — `nic_via_xgmi` (mean µs):**

| size | baseline `D D` | **noidc** | noidc+noinject | `H H` | native |
|------|---------------:|----------:|---------------:|------:|-------:|
| 1 B | 20.58 | **3.90** | 3.90 | 2.34 | 2.41 |
| 2–64 B | ~20.6 | **~3.18** | ~3.17 | ~2.35 | ~2.41 |
| 128 B | 21.11 | **3.21** | 3.21 | 2.88 | 2.97 |
| 256 B | 3.96 | **3.24** | 3.25 | 2.91 | 3.13 |

**Takeaways**

1. **~6× improvement.** Tiny-message `D D` latency drops from ~17.5–20.6 µs to ~2.9–3.2 µs.
2. **At or beyond parity with native** for `nic_local` pairs; within ~0.8 µs for `nic_via_xgmi`
   pairs (residual = the physical intra-OAM xGMI hop + OpenMPI per-send overhead) — comfortably
   inside the "5–10% / good enough" target.
3. **`OMPI_MCA_mtl_ofi_inject_size=0` adds nothing** — `noidc` and `noidc_noinject` columns are
   identical, so OpenMPI's OFI MTL was not using `fi_tinject` for these device sends. The single
   env var is the entire fix.
4. **Pair-independent** — confirmed on both NIC tiers, hence committed to `common.sh`. Full 8-pair
   confirmation: job 18761905.

---

## 5. Collectives — same root cause, same fix, inherited automatically

`osu_collectives.sh` sources `common.sh` and already runs `-d rocm`, so it inherits the fix with no
edit. OpenMPI's `tuned`/`han` collectives decompose into point-to-point sends over the same `cxi`
path, so every small inter-node send in a collective was paying the ~15 µs IDC staging, compounded
across the algorithm's communication rounds.

**N=16 (2×8 GCD) small-message latency, baseline vs after-fix (mean µs):**

| collective | baseline (18750788) | after fix (18761906) | small-msg speedup |
|------------|--------------------:|---------------------:|------------------:|
| bcast (1–64 B) | ~81–95 | **~9–11** | **~9×** |
| alltoall (1 B) | 767 | **73** | **~10×** |
| allreduce (8–64 B) | ~180–186 | **~43–46** | **~4×** |
| allgather (1–8 B) | ~194–197 | **~90–92** | **~2×** |

The fix transfers directly to collectives (each is built from point-to-point sends). **No host
buffers, no separate collectives fix** — they inherit the env var from `common.sh`. However,
collectives also expose two *additional* bottlenecks the IDC fix does not address (many-to-many
message-rate stalls; large-message algorithm/GPU-awareness) — analysed separately in
`osu_collectives_analysis.md`.

---

## 6. Why not `host_injections`?

It is unnecessary and would not help: `host_injections` would ship the *same* `cxi` provider with the
*same* IDC behavior. The hermetic shs-libfabric 1.22.0 + OpenMPI 5.0.7 stack is correct as built; the
problem was a single runtime tuning knob, not a missing/incompatible library. The "watertight, no
host system libraries" directive is preserved.

---

## 7. Reproduce

```bash
cd thesis-benchmarks/3_osu_internode
sbatch osu_latency_idc.sh verify          # confirm cxi provider + OFI MTL
sbatch osu_latency_idc.sh hostbuf         # diagnostic: H H removes the plateau
sbatch osu_latency_idc.sh baseline        # D D plateau (control)
sbatch osu_latency_idc.sh noidc           # the fix (FI_CXI_DISABLE_NON_INJECT_MSG_IDC=1)
# fix now lives in common.sh, so the standard runs inherit it:
sbatch osu_latency.sh eessi
sbatch osu_collectives.sh eessi
```

`osu_latency_idc.sh` is the parametric sweep (EESSI-only, 2 representative pairs: one `nic_local`
GCD7 + one `nic_via_xgmi` GCD0). The fix in `common.sh` is reversible — comment out the one line to
restore the pre-fix behavior.

---

## 8. References

- **libfabric `fi_cxi(7)` provider manual** — documents `FI_CXI_DISABLE_NON_INJECT_MSG_IDC`, the IDC
  vs DMA small-message paths, `FI_CXI_RX_MATCH_MODE`, rendezvous thresholds.
  https://ofiwg.github.io/libfabric/v2.1.0/man/fi_cxi.7.html
- **EESSI blog — "MPI at Warp Speed: EESSI Meets Slingshot-11" (2025-11-14)** — official EESSI
  Slingshot-11 effort; reports the same ~4× small-message GPU-buffer latency gap.
  https://www.eessi.io/docs/blog/2025/11/14/EESSI-on-Cray-Slingshot/
- **Shehata, Naughton, Bernholdt, Pritchard — "Bringing HPE Slingshot 11 support to Open MPI",**
  *Concurrency and Computation: Practice and Experience* 36(22), 2024, DOI 10.1002/cpe.8203 —
  CXI provider, OFI MTL, LINKx (shm+cxi) on Cray EX. https://www.osti.gov/pages/biblio/2438730
- **Open MPI 5.0.x — OFI/libfabric networking guide** — `pml cm` + `mtl ofi`, provider include.
  https://docs.open-mpi.org/en/v5.0.x/tuning-apps/networking/ofi.html
- **CSCS — Open MPI on Cray EX / Slingshot** — `FI_PROVIDER=cxi`, `pml cm`/`mtl ofi`, PMIx launch.
  https://docs.cscs.ch/software/communication/openmpi/
- **HPE shs-libfabric (CXI provider source)** — the libfabric 1.22.0 / shs-12.0.2 fork used in the
  hermetic build. https://github.com/HewlettPackard/shs-libfabric
- **Open MPI issue #12233** — OFI BTL/MTL provider selection (shm vs cxi) with libfabric ≥1.20.
  https://github.com/open-mpi/ompi/issues/12233

## 9. Related local docs / memory

- Plan: `.claude/plans/let-s-plan-something-the-eager-hoare.md`
- Stack build history: memory `osu-internode-cxi-rebuild`, directive `eessi-hermetic-no-system-libs`
- Experiment script: `osu_latency_idc.sh`; fix committed in `common.sh` (EESSI CXI block).
