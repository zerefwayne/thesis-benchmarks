"""T1, T2, T3 — CSV tables for the thesis.

T1: Tier summary — peak-size median bandwidth per (tier, stack), plus EESSI/native ratio.
T2: Portability summary — per-primitive measured value, theoretical peak (where
    defined), % of peak achieved by each stack, EESSI portability tax %.
T3: Collective cross-over — for each (collective, N), the message size at which
    EESSI catches up to native (latency_eessi <= latency_native crossing point).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from _common import (
    PAIR_TIER, REP_PAIRS, TIER_ORDER, TIER_PEAK_GBPS,
    fmt_size, load_collectives, load_pt2pt, median_runs, save_table, stack_pivot,
)


PEAK_SIZE_BW = 1 << 25  # 32 MiB
COLLECTIVES = ["osu_allreduce", "osu_alltoall", "osu_bcast", "osu_allgather"]
N_GCDS = [2, 4, 8]


def t1_tier_summary():
    df = load_pt2pt("bw")
    df = df[df["size_bytes"] == PEAK_SIZE_BW]
    if df.empty:
        print("  !! no bw data at peak size; skipping T1")
        return
    med = median_runs(df, "bandwidth_MBps",
                      ["stack", "pair_label", "tier"])
    # median across pairs within each (tier, stack)
    tier_med = (med.groupby(["tier", "stack"], as_index=False)["bandwidth_MBps"]
                   .median())
    wide = tier_med.pivot(index="tier", columns="stack",
                          values="bandwidth_MBps").reset_index()
    wide["ratio_eessi_native"] = wide["eessi"] / wide["native"]
    wide["tier"] = pd.Categorical(wide["tier"], categories=TIER_ORDER, ordered=True)
    wide = wide.sort_values("tier").reset_index(drop=True)
    wide.columns.name = None
    wide = wide.rename(columns={
        "eessi": "eessi_MBps_median",
        "native": "native_MBps_median",
    })
    wide["peak_size_bytes"] = PEAK_SIZE_BW
    cols = ["tier", "peak_size_bytes",
            "eessi_MBps_median", "native_MBps_median", "ratio_eessi_native"]
    save_table(wide[cols], "T1_tier_summary_bw_peak")


def t3_collective_crossover():
    df = load_collectives()
    med = median_runs(df, "latency_us",
                      ["stack", "benchmark", "num_gcds", "size_bytes"])
    wide = stack_pivot(med, ["benchmark", "num_gcds", "size_bytes"],
                       "latency_us")
    wide = wide.dropna(subset=["eessi", "native"])
    # advantage = native_lat / eessi_lat (>1 = EESSI faster)
    wide["adv"] = wide["native"] / wide["eessi"]

    rows = []
    for coll in COLLECTIVES:
        for n in N_GCDS:
            sub = (wide[(wide["benchmark"] == coll) & (wide["num_gcds"] == n)]
                   .sort_values("size_bytes")
                   .reset_index(drop=True))
            if sub.empty:
                rows.append({"benchmark": coll, "num_gcds": n,
                             "smallest_size_eessi_wins": np.nan,
                             "advantage_at_smallest_win": np.nan,
                             "advantage_at_min_size": np.nan,
                             "advantage_at_max_size": np.nan})
                continue
            wins = sub[sub["adv"] >= 1.0]
            first_win = int(wins["size_bytes"].iloc[0]) if not wins.empty else np.nan
            adv_first_win = (float(wins["adv"].iloc[0])
                             if not wins.empty else np.nan)
            rows.append({
                "benchmark": coll,
                "num_gcds": n,
                "smallest_size_eessi_wins": first_win,
                "advantage_at_smallest_win": adv_first_win,
                "advantage_at_min_size": float(sub["adv"].iloc[0]),
                "advantage_at_max_size": float(sub["adv"].iloc[-1]),
            })
    out = pd.DataFrame(rows)
    save_table(out, "T3_collective_crossover")


def t2_portability_summary():
    """Per-primitive table: measured native, eessi, theoretical peak, % of peak, tax."""
    rows = []

    # ---- pt2pt bandwidth at 1 MiB and 32 MiB for each REP pair ----
    bw = load_pt2pt("bw")
    bw["bw_GBps"] = bw["bandwidth_MBps"] / 1000.0
    bw_med = median_runs(bw, "bw_GBps",
                         ["stack", "pair_label", "size_bytes"])
    for pair in REP_PAIRS:
        tier = PAIR_TIER[pair]
        peak = TIER_PEAK_GBPS[tier]
        for size in (1 << 20, 1 << 25):
            sub = bw_med[(bw_med["pair_label"] == pair)
                         & (bw_med["size_bytes"] == size)]
            e = sub[sub["stack"] == "eessi"]["bw_GBps"]
            n = sub[sub["stack"] == "native"]["bw_GBps"]
            if e.empty or n.empty:
                continue
            ev, nv = float(e.iloc[0]), float(n.iloc[0])
            tax = (nv - ev) / nv * 100
            rows.append({
                "benchmark": "osu_bw",
                "condition": f"{pair} @ {fmt_size(size)}",
                "units": "GB/s",
                "native": round(nv, 2),
                "eessi": round(ev, 2),
                "theoretical_peak": peak,
                "native_pct_of_peak": round(nv / peak * 100, 1) if peak else None,
                "eessi_pct_of_peak": round(ev / peak * 100, 1) if peak else None,
                "portability_tax_pct": round(tax, 2),
            })

    # ---- bibw at 32 MiB intra-pkg ----
    bibw = load_pt2pt("bibw")
    bibw["bw_GBps"] = bibw["bandwidth_MBps"] / 1000.0
    bibw_med = median_runs(bibw, "bw_GBps",
                           ["stack", "pair_label", "size_bytes"])
    pair, size = "intra_pkg_OAM0", 1 << 25
    sub = bibw_med[(bibw_med["pair_label"] == pair)
                   & (bibw_med["size_bytes"] == size)]
    e = sub[sub["stack"] == "eessi"]["bw_GBps"]
    n = sub[sub["stack"] == "native"]["bw_GBps"]
    if not e.empty and not n.empty:
        ev, nv = float(e.iloc[0]), float(n.iloc[0])
        # Bidirectional: nominal peak is 2x the unidirectional IF peak.
        peak = TIER_PEAK_GBPS["intra_pkg"] * 2 if TIER_PEAK_GBPS["intra_pkg"] else None
        tax = (nv - ev) / nv * 100
        rows.append({
            "benchmark": "osu_bibw",
            "condition": f"{pair} @ {fmt_size(size)}",
            "units": "GB/s",
            "native": round(nv, 2),
            "eessi": round(ev, 2),
            "theoretical_peak": peak,
            "native_pct_of_peak": round(nv / peak * 100, 1) if peak else None,
            "eessi_pct_of_peak": round(ev / peak * 100, 1) if peak else None,
            "portability_tax_pct": round(tax, 2),
        })

    # ---- latency at 8 B and 1 KiB intra-pkg ----
    lat = load_pt2pt("latency")
    lat_med = median_runs(lat, "latency_us",
                          ["stack", "pair_label", "size_bytes"])
    for size in (8, 1024):
        sub = lat_med[(lat_med["pair_label"] == "intra_pkg_OAM0")
                      & (lat_med["size_bytes"] == size)]
        e = sub[sub["stack"] == "eessi"]["latency_us"]
        n = sub[sub["stack"] == "native"]["latency_us"]
        if e.empty or n.empty:
            continue
        ev, nv = float(e.iloc[0]), float(n.iloc[0])
        tax = (ev - nv) / nv * 100
        rows.append({
            "benchmark": "osu_latency",
            "condition": f"intra_pkg_OAM0 @ {fmt_size(size)}",
            "units": "us",
            "native": round(nv, 3),
            "eessi": round(ev, 3),
            "theoretical_peak": None,
            "native_pct_of_peak": None,
            "eessi_pct_of_peak": None,
            "portability_tax_pct": round(tax, 2),
        })

    # ---- collectives at N=8, three sizes ----
    coll = load_collectives()
    coll_med = median_runs(coll, "latency_us",
                           ["stack", "benchmark", "num_gcds", "size_bytes"])
    for bm in COLLECTIVES:
        for sz in (4, 65536, 1 << 20):
            sub = coll_med[(coll_med["benchmark"] == bm)
                           & (coll_med["num_gcds"] == 8)
                           & (coll_med["size_bytes"] == sz)]
            e = sub[sub["stack"] == "eessi"]["latency_us"]
            n = sub[sub["stack"] == "native"]["latency_us"]
            if e.empty or n.empty:
                continue
            ev, nv = float(e.iloc[0]), float(n.iloc[0])
            tax = (ev - nv) / nv * 100
            rows.append({
                "benchmark": bm,
                "condition": f"N=8 @ {fmt_size(sz)}",
                "units": "us",
                "native": round(nv, 3),
                "eessi": round(ev, 3),
                "theoretical_peak": None,
                "native_pct_of_peak": None,
                "eessi_pct_of_peak": None,
                "portability_tax_pct": round(tax, 2),
            })

    out = pd.DataFrame(rows)
    save_table(out, "T2_portability_summary")


def main():
    print("[tables] T1 — tier summary @ 32 MiB")
    t1_tier_summary()
    print("[tables] T2 — portability summary across primitives")
    t2_portability_summary()
    print("[tables] T3 — collective EESSI/native crossover")
    t3_collective_crossover()


if __name__ == "__main__":
    main()
