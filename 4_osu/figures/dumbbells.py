"""D3 — dumbbell / slope plots.

For each tier representative pair, a horizontal "dumbbell" — Native marker
and EESSI marker connected by a thick gray segment. The visual length of
the segment is the absolute gap; the delta is annotated on the right.

Reads as: "EESSI is X units away from Native for this pair." Strong
because both the absolute values and the gap are visible at once.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from _common import (
    PAIR_TIER, REP_PAIRS, STACK_COLORS, STACK_LABELS,
    TIER_COLORS,
    load_pt2pt, median_runs, save,
)


def dumbbell(name, value_col, unit, divisor, fixed_size, size_label,
             fname, lower_is_better=False):
    df = load_pt2pt(name)
    df = df[(df["size_bytes"] == fixed_size)
            & (df["pair_label"].isin(REP_PAIRS))]
    if df.empty:
        print(f"  !! no {name} data at {size_label}; skipping")
        return
    med = median_runs(df, value_col, ["stack", "pair_label"])

    fig, ax = plt.subplots(figsize=(11, 5.2))
    y = np.arange(len(REP_PAIRS))[::-1]

    legend_done = False
    for yi, pair in zip(y, REP_PAIRS):
        e = med[(med["stack"] == "eessi") & (med["pair_label"] == pair)][value_col]
        n = med[(med["stack"] == "native") & (med["pair_label"] == pair)][value_col]
        if e.empty or n.empty:
            continue
        ev = float(e.iloc[0]) / divisor
        nv = float(n.iloc[0]) / divisor

        ax.plot([min(ev, nv), max(ev, nv)], [yi, yi],
                color="gray", linewidth=5, alpha=0.5, zorder=1,
                solid_capstyle="round")
        ax.scatter([nv], [yi], s=160, color=STACK_COLORS["native"],
                   edgecolor="black", linewidth=0.6, zorder=2,
                   label=STACK_LABELS["native"] if not legend_done else None)
        ax.scatter([ev], [yi], s=160, color=STACK_COLORS["eessi"],
                   edgecolor="black", linewidth=0.6, zorder=2,
                   label=STACK_LABELS["eessi"] if not legend_done else None)
        legend_done = True

        if lower_is_better:
            delta = nv - ev
        else:
            delta = ev - nv
        delta_pct = delta / nv * 100
        sign = "+" if delta >= 0 else ""
        rmost = max(ev, nv)
        ax.text(
            rmost, yi,
            f"   Δ = {sign}{delta:.2f} {unit}  ({sign}{delta_pct:.1f}%)",
            va="center", ha="left", fontsize=10,
            color=("#2ca02c" if delta >= 0 else "#d62728"),
            fontweight="bold",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(REP_PAIRS)
    for tick, p in zip(ax.get_yticklabels(), REP_PAIRS):
        tick.set_color(TIER_COLORS[PAIR_TIER[p]])
    ax.set_xlabel(f"{name} @ {size_label} ({unit})")
    ax.set_title(
        f"{name} @ {size_label} — head-to-head dumbbell "
        "(green Δ = EESSI faster, red Δ = Native faster)"
    )
    ax.grid(True, axis="x", linestyle=":", alpha=0.4)
    ax.legend(frameon=True, loc="lower right")

    xlim = ax.get_xlim()
    ax.set_xlim(xlim[0], xlim[1] + 0.45 * (xlim[1] - xlim[0]))
    save(fig, fname)


def main():
    print("[dumbbells] D3a — bw @ 1 MiB")
    dumbbell("bw", "bandwidth_MBps", "GB/s", 1000.0, 1 << 20, "1 MiB",
             "D3a_dumbbell_bw_1MiB", lower_is_better=False)
    print("[dumbbells] D3b — bw @ 32 MiB")
    dumbbell("bw", "bandwidth_MBps", "GB/s", 1000.0, 1 << 25, "32 MiB",
             "D3b_dumbbell_bw_32MiB", lower_is_better=False)
    print("[dumbbells] D3c — latency @ 8 B")
    dumbbell("latency", "latency_us", "us", 1.0, 8, "8 B",
             "D3c_dumbbell_latency_8B", lower_is_better=True)
    print("[dumbbells] D3d — latency @ 1 KiB")
    dumbbell("latency", "latency_us", "us", 1.0, 1024, "1 KiB",
             "D3d_dumbbell_latency_1KiB", lower_is_better=True)


if __name__ == "__main__":
    main()
