# EESSI vs native collectives: two-front tuning strategy

Companion note for [osu_collectives_fixed_rndv.sh](osu_collectives_fixed_rndv.sh) and [osu_collectives_ucc_shm.sh](osu_collectives_ucc_shm.sh).

## Problem

Comparing `osu_collectives_eessi_18634710.csv` vs `osu_collectives_native_18634695.csv` (N=8, intra-node, all four blocking collectives) reveals **two distinct EESSI weaknesses, plus one EESSI win** at large sizes.

Per-benchmark medians of recorded runs, EESSI / native ratio:

| size band | osu_bcast | osu_allgather | osu_allreduce | osu_alltoall |
|-----------|:---------:|:-------------:|:-------------:|:------------:|
| 1 B – 128 B   | 🔥 4–6× slow | 🔥 4–5× slow | ⚠️ +75–90% slow | ≈ tied (+10–20%) |
| **256 B – 4 K** | 🔥 **4–66× slow** | 🔥 **2–5× slow** | 🔥 **6–9× slow** | 🔥 **2–13× slow** |
| 8 K – 64 K     | mixed | ≈ tied | ⚠️ +75–180% slow | ✅ EESSI 1.5× faster |
| ≥ 128 K       | ✅ EESSI ~15% faster | ✅ EESSI 16–37% faster | ✅ EESSI 25–39% faster | ≈ tied / EESSI ahead |

Two specific knees stand out:
- **256 B – 4 KiB cliff** of up to **66×** (bcast at 256 B: 37.8 µs vs 0.6 µs). Same shape as the pt2pt cliff in `osu_bw_eessi`.
- **1 B – 128 B floor** where EESSI sits 1.8–6× behind native (bcast 2.0 µs vs 0.5 µs). New finding — *below* the rndv switch, so unrelated to rendezvous.

## Two tuning fronts, two scripts

The two problems have different root causes and need different fixes. Mixing them into one script would conflate variables — we'd see "EESSI improved" but not know which knob did the work. Each script changes exactly one variable.

### Front 1 — `osu_collectives_fixed_rndv.sh`

Adds `UCX_RNDV_THRESH=1024` to the EESSI arm. Inherited directly from [fix_rndv_reasoning.md](fix_rndv_reasoning.md):

- DEFAULT UCX threshold lands at ~256 B → cliff starts there.
- 1024 keeps 256 B in eager mode → fixes the cliff at exactly 256 B.
- Higher thresholds (16 MiB) cap bulk bandwidth at ~660 MB/s due to UCX eager bounce-buffer ceiling — verified empirically.
- Strict dominance over DEFAULT confirmed in `osu_protocol_eessi_18634230`.

The collectives version of this should be **more impactful than the pt2pt version** because every collective fans out 8 messages — the 256 B per-step rendezvous overhead gets multiplied 8× by the algorithm.

**Expected effect:** the 256 B – 4 KiB cliff narrows or disappears; the 1 B – 128 B gap is unchanged (below the threshold).

### Front 2 — `osu_collectives_ucc_shm.sh`

Adds `shm` to the UCC transport-layer list — `UCC_TLS=ucp,shm,self` instead of `ucp,self`.

**Why:** With `ucp,self`, UCC routes every collective primitive through point-to-point UCX. Each step pays UCX active-message and tag-matching overhead. For 8 ranks on a single host where shared memory is the natural transport, this is two layers of indirection.

Adding `shm` exposes UCX's shared-memory transport to UCC directly. UCC's CL_BASIC (basic collective library) can then use SHM for the data-movement primitives, skipping the UCX P2P wrapper.

Cray MPICH's sub-microsecond bcast almost certainly comes from a similar SHM-direct path (or a hardware-accelerated CXI primitive on Slingshot — but we're intra-node, so that won't apply). This is the closest EESSI-side analog we can configure without rebuilding UCC.

**Expected effect:** 1 B – 128 B latencies should drop substantially (target: 2–3 µs for bcast, down from 4–6 µs in the baseline). The 256 B – 4 KiB cliff likely stays — it's above the size where SHM matters most and below where rendezvous matters.

**Risk:** If the UCC build in EESSI 2025.06 doesn't include the `shm` TL, UCC silently falls back to UCP and results are identical to the baseline. Verified by inspecting the .log for UCC startup messages (`UCC_TL_SHM` lines if available).

### What about combining both?

A third variant — `UCX_RNDV_THRESH=1024` *and* `UCC_TLS=ucp,shm,self` together — is the obvious next step once we know each works in isolation. Save it for after these two run. The plot story is cleaner if we can show "rndv knob → fixes cliff," "shm knob → fixes small-msg," and only then "both → headline number for the thesis."

## What we cannot fix

The **sub-microsecond bcast on native** (0.5 µs at 1 B) is almost certainly a vendor hardware-fast-path that Cray MPICH calls via the Cray PE. EESSI's stock OpenMPI cannot match this without RCCL routing — and routing through RCCL costs us the small-msg latency win, since RCCL has ~26 µs collective-launch overhead. There's no env-var workaround. Honest framing: "Cray's vendor primitive is unbeatable at small sizes; EESSI is competitive but not faster."

## What we expect to keep winning

**The ≥ 128 KiB EESSI advantage is already there in the baseline** and doesn't need tuning:
- allreduce 1 MiB: **197 µs (EESSI) vs 323 µs (native)** — EESSI 39 % faster.
- allgather 1 MiB: 310 vs 370 µs — EESSI 16 % faster.
- allreduce 128 KiB: 166 vs 214 µs — EESSI 22 % faster.

This is UCC's algorithm choice (Rabenseifner-style recursive halving) scaling better than Cray's at scale. Both proposed tuning changes should leave this regime untouched.

## Honest framing for the thesis

> EESSI's collective profile shows a clear crossover: native MPI is **3–6× faster at small messages** (latency-dominated regime where vendor-tuned primitives dominate), but EESSI is **15–40 % faster at large messages** (bandwidth-dominated regime where UCC's algorithm choices pay off). The 256 B – 4 KiB cliff is a protocol-default issue, solvable via `UCX_RNDV_THRESH=1024` (same fix as pt2pt). The sub-microsecond bcast gap at 1 B is a vendor-fast-path limitation that EESSI cannot match. The 1 B – 128 B small-msg gap may be partially recovered by exposing UCX shared-memory to UCC (`UCC_TLS=ucp,shm,self`); see `osu_collectives_ucc_shm.sh`.

## Reproduction

```bash
# Front 1: fix the 256B-4K cliff
sbatch osu_collectives_fixed_rndv.sh eessi
sbatch osu_collectives_fixed_rndv.sh native

# Front 2: attack the 1B-128B latency floor
sbatch osu_collectives_ucc_shm.sh eessi
sbatch osu_collectives_ucc_shm.sh native
```

CSV schema in both scripts is identical to `osu_collectives.sh`, so all six result files can be concat'd against the baseline pair for plotting.

## Source data

- [results/osu_collectives_eessi_18634710.csv](results/osu_collectives_eessi_18634710.csv) — baseline EESSI collectives
- [results/osu_collectives_native_18634695.csv](results/osu_collectives_native_18634695.csv) — baseline native collectives
- [results/osu_protocol_eessi_18634230.csv](results/osu_protocol_eessi_18634230.csv) — UCX_RNDV_THRESH sweep that justified 1024
- [fix_rndv_reasoning.md](fix_rndv_reasoning.md) — full pt2pt rendezvous-threshold reasoning
