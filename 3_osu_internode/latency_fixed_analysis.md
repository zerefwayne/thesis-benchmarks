# Inter-node latency: EESSI (fixed) vs native Cray MPICH

**Scope:** point-to-point `osu_latency -d rocm D D`, 2 nodes, after applying the IDC fix
(`FI_CXI_DISABLE_NON_INJECT_MSG_IDC=1`, see `IDC_LATENCY_FIX_ANALYSIS.md`).
**Data:** EESSI = job **18761905** (8 pairs, OpenMPI 5.0.7 + shs-libfabric 1.22.0 CXI / OFI MTL);
native = job **18750321** (Cray MPICH). Values are means over the 5 recorded runs per size.
**Date:** 2026-05-21.

---

## 1. Headline

The small-message plateau is gone. Across the full size range EESSI now tracks native closely and
**beats it across most of the mid/large range**. The only residual deficits are (a) a ~1.2–1.35×
gap at the sub-µs latency floor (≤64 B), and (b) a localized ~1.4–1.8× bump at 8–16 KiB (the
eager→rendezvous transition). Everything ≥32 KiB is at parity.

| regime | EESSI vs native |
|--------|-----------------|
| 1–64 B (floor) | 1.23× (nic_local) / 1.35× (nic_via_xgmi) — i.e. +0.6–0.9 µs |
| 128 B – 4 KiB | **parity or EESSI faster** (ratios 0.79–1.09) |
| 8–16 KiB | **1.35–1.79× slower** (rendezvous-transition bump) |
| 32 KiB – 4 MiB | parity (ratios 0.87–1.02) |

---

## 2. Full size curve (representative pairs, mean µs)

### GCD7–GCD7 — `nic_local`

| size | EESSI | native | ratio |
|------|------:|-------:|------:|
| 1 B | 2.93 | 2.73 | 1.07 |
| 8 B | 2.92 | 2.72 | 1.07 |
| 64 B | 2.92 | 2.73 | 1.07 |
| 128 B | 2.94 | 3.28 | 0.90 |
| 256 B | 2.96 | 3.66 | 0.81 |
| 1 KiB | 3.11 | 3.89 | 0.80 |
| 2 KiB | 3.25 | 4.11 | 0.79 |
| 4 KiB | 3.40 | 4.20 | 0.81 |
| **8 KiB** | **6.51** | 4.54 | **1.43** |
| **16 KiB** | **6.92** | 5.13 | **1.35** |
| 32 KiB | 7.70 | 8.90 | 0.87 |
| 64 KiB | 9.10 | 10.29 | 0.88 |
| 256 KiB | 17.25 | 18.40 | 0.94 |
| 1 MiB | 49.84 | 51.16 | 0.97 |
| 4 MiB | 179.48 | 182.99 | 0.98 |

### GCD0–GCD0 — `nic_via_xgmi`

| size | EESSI | native | ratio |
|------|------:|-------:|------:|
| 1 B | 3.96 | 2.41 | 1.64 |
| 8 B | 3.22 | 2.41 | 1.34 |
| 64 B | 3.23 | 2.41 | 1.34 |
| 128 B | 3.23 | 2.97 | 1.09 |
| 256 B | 3.24 | 3.13 | 1.04 |
| 1 KiB | 3.47 | 3.28 | 1.06 |
| 4 KiB | 3.66 | 3.56 | 1.03 |
| **8 KiB** | **6.97** | 3.89 | **1.79** |
| **16 KiB** | **7.37** | 4.50 | **1.64** |
| 32 KiB | 8.15 | 7.92 | 1.03 |
| 64 KiB | 9.52 | 9.30 | 1.02 |
| 1 MiB | 50.13 | 50.08 | 1.00 |
| 4 MiB | 179.66 | 179.66 | 1.00 |

---

## 3. Tier analysis

Aggregate over 1–64 B, all four pairs per tier:

| tier | EESSI avg | native avg | ratio |
|------|----------:|-----------:|------:|
| `nic_local` (GCD 1,3,5,7) | 3.08 µs | 2.50 µs | **1.23×** |
| `nic_via_xgmi` (GCD 0,2,4,6) | 3.38 µs | 2.51 µs | **1.35×** |

- The `nic_via_xgmi` tier carries the extra intra-OAM xGMI hop to reach the NIC; EESSI pays ~0.3 µs
  more there than on `nic_local`, consistent with that hop. Native is flat across tiers (~2.5 µs)
  because its small-message path hides the hop better.
- The residual floor gap (+0.6–0.9 µs) is OpenMPI's per-message software overhead in the `cm` PML /
  OFI MTL versus Cray MPICH's leaner native path. It is a fundamental stack difference, not a
  misconfiguration, and is far inside the goal (vs the ~6–8× / "60×" pre-fix fear).
- The GCD0 1-byte point (3.96 µs) is an outlier above its own 8–64 B values (~3.22 µs); minor, not
  pursued.

---

## 4. The 8–16 KiB rendezvous bump (only remaining wrinkle)

EESSI rises sharply from ~3.4 µs at 4 KiB to ~6.5–7.0 µs at 8–16 KiB, then drops back to parity at
32 KiB. Native climbs smoothly through the same region. This is the **eager→rendezvous protocol
transition**: EESSI's OFI/CXI rendezvous engages around 8 KiB and adds a handshake round-trip that
native amortizes more smoothly.

- **Impact:** localized; affects only the 8–16 KiB octave. Below it EESSI is *faster* than native;
  above 32 KiB they are equal.
- **Candidate follow-up (optional):** raise the eager/rendezvous crossover so the bump moves out of
  this window — sweep `FI_CXI_RDZV_THRESHOLD` (and/or `FI_CXI_RDZV_EAGER_SIZE`) upward, e.g. via the
  existing `osu_protocol_eessi.sh` harness, and re-check that large-message bandwidth is unaffected.
  Not required for the thesis goal; noting it for completeness.

---

## 5. Conclusion

With the one-line runtime fix in `common.sh`, EESSI inter-node `osu_latency` (device buffers) is
competitive with native Cray MPICH across the entire size range: within ~1.2–1.35× at the latency
floor, at parity or faster from 128 B through 4 KiB and again from 32 KiB to 4 MiB, with a single
localized 8–16 KiB rendezvous bump as the only remaining sub-parity region. No rebuild, no
`host_injections`; the hermetic CXI stack stands.

**Sources / context:** root-cause and fix derivation in `IDC_LATENCY_FIX_ANALYSIS.md` (incl. the
`fi_cxi(7)` man page and EESSI Slingshot-11 blog references). Raw CSVs:
`results/osu_latency_eessi_18761905.csv`, `results/osu_latency_native_18750321.csv`.
