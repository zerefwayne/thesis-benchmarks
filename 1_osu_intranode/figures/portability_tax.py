"""P3 — single-figure thesis bullet: EESSI portability tax across primitives.

For each (benchmark, condition) representative, compute the % gap of EESSI
vs Native, sign-corrected so positive always means EESSI is slower (the
"portability tax"). One horizontal bar per primitive, sorted ascending so
EESSI's wins float to the top. Plot title reports median, p90, and worst.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _common import (
    fmt_size, load_collectives, load_pt2pt, median_runs, save,
)


POS_COLOR = "#d62728"  # EESSI slower (tax)
NEG_COLOR = "#2ca02c"  # EESSI faster

PT2PT_ROWS = [
    # (name, value_col, lower_is_better, pair, size, size_label, short_pair)
    ("bw",      "bandwidth_MBps", False, "intra_pkg_OAM0",      1 << 20, "1 MiB",  "intra-pkg"),
    ("bw",      "bandwidth_MBps", False, "intra_pkg_OAM0",      1 << 25, "32 MiB", "intra-pkg"),
    ("bw",      "bandwidth_MBps", False, "inter_pkg_2link_06",  1 << 25, "32 MiB", "inter-2L"),
    ("bw",      "bandwidth_MBps", False, "inter_pkg_1link_02",  1 << 25, "32 MiB", "inter-1L"),
    ("bw",      "bandwidth_MBps", False, "routed_07",           1 << 25, "32 MiB", "routed"),
    ("bibw",    "bandwidth_MBps", False, "intra_pkg_OAM0",      1 << 25, "32 MiB", "intra-pkg"),
    ("latency", "latency_us",     True,  "intra_pkg_OAM0",      8,        "8 B",    "intra-pkg"),
    ("latency", "latency_us",     True,  "intra_pkg_OAM0",      1024,     "1 KiB",  "intra-pkg"),
]

COLL_BENCHMARKS = ["osu_allreduce", "osu_alltoall", "osu_bcast", "osu_allgather"]
COLL_SIZES = [4, 65536, 1 << 20]   # 4 B, 64 KiB, 1 MiB
COLL_N = 8


def _pt2pt_tax(name, value_col, lower_is_better, pair, size):
    df = load_pt2pt(name)
    df = df[(df["pair_label"] == pair) & (df["size_bytes"] == size)]
    if df.empty:
        return None
    med = median_runs(df, value_col, ["stack"])
    e = med[med["stack"] == "eessi"][value_col]
    n = med[med["stack"] == "native"][value_col]
    if e.empty or n.empty:
        return None
    ev, nv = float(e.iloc[0]), float(n.iloc[0])
    if lower_is_better:
        return (ev - nv) / nv * 100   # latency: EESSI slower if EESSI > Native
    return (nv - ev) / nv * 100       # bandwidth: EESSI slower if EESSI < Native


def _coll_tax(benchmark, n_gcds, size):
    df = load_collectives()
    df = df[(df["benchmark"] == benchmark)
            & (df["num_gcds"] == n_gcds)
            & (df["size_bytes"] == size)]
    if df.empty:
        return None
    med = median_runs(df, "latency_us", ["stack"])
    e = med[med["stack"] == "eessi"]["latency_us"]
    n = med[med["stack"] == "native"]["latency_us"]
    if e.empty or n.empty:
        return None
    ev, nv = float(e.iloc[0]), float(n.iloc[0])
    return (ev - nv) / nv * 100


def main():
    print("[portability_tax] P3 — thesis-bullet portability tax bar chart")
    rows = []

    for name, vcol, lib, pair, size, size_label, short_pair in PT2PT_ROWS:
        tax = _pt2pt_tax(name, vcol, lib, pair, size)
        if tax is None:
            continue
        label = f"{name} @ {size_label}, {short_pair}"
        rows.append((label, tax))

    for bm in COLL_BENCHMARKS:
        for sz in COLL_SIZES:
            tax = _coll_tax(bm, COLL_N, sz)
            if tax is None:
                continue
            short_bm = bm.replace("osu_", "")
            rows.append((f"{short_bm} @ N={COLL_N}, {fmt_size(sz)}", tax))

    if not rows:
        print("  !! no rows produced; skipping P3")
        return
    df = pd.DataFrame(rows, columns=["label", "tax"]).sort_values("tax").reset_index(drop=True)

    colors = [POS_COLOR if t >= 0 else NEG_COLOR for t in df["tax"]]
    n = len(df)

    fig, ax = plt.subplots(figsize=(11, max(5.5, 0.36 * n)))
    bars = ax.barh(range(n), df["tax"], color=colors,
                   edgecolor="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=1.0)
    ax.set_yticks(range(n))
    ax.set_yticklabels(df["label"])

    xmax = max(abs(df["tax"].min()), abs(df["tax"].max()))
    pad = 0.18 * xmax if xmax > 0 else 1.0
    ax.set_xlim(-xmax - pad, xmax + pad)

    for bar, t in zip(bars, df["tax"]):
        x = bar.get_width()
        ha = "left" if x >= 0 else "right"
        offset = pad * 0.10 if x >= 0 else -pad * 0.10
        ax.text(x + offset, bar.get_y() + bar.get_height() / 2,
                f"{t:+.1f}%", va="center", ha=ha, fontsize=9)

    median = df["tax"].median()
    p90 = df["tax"].quantile(0.90)
    worst = df["tax"].max()
    n_within_5 = int(((df["tax"] >= -5) & (df["tax"] <= 5)).sum())

    ax.set_title(
        f"EESSI portability tax across {n} primitives\n"
        f"median {median:+.1f}%   p90 {p90:+.1f}%   worst {worst:+.1f}%   "
        f"({n_within_5}/{n} within ±5% of Native)",
        fontsize=11, pad=14,
    )
    ax.set_xlabel("EESSI portability tax (%) — positive: EESSI slower; negative: EESSI faster")
    ax.grid(True, axis="x", linestyle=":", alpha=0.4)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=NEG_COLOR, label="EESSI faster (negative tax)"),
        plt.Rectangle((0, 0), 1, 1, color=POS_COLOR, label="EESSI slower (positive tax)"),
    ]
    ax.legend(handles=handles, frameon=True, loc="lower right")
    save(fig, "P3_portability_tax")


if __name__ == "__main__":
    main()
