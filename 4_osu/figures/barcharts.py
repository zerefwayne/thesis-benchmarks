"""B1, B2, B4 — head-to-head EESSI-vs-Native grouped bar charts.

B1: per-pair bandwidth @ 1 MiB.
B2: per-pair latency  @ 8 B.
B4: tier-aggregated bandwidth at 3 representative sizes.

All bars: median across the 5 recorded runs; error bars = min/max.
Pairs are ordered by tier (intra_pkg, inter_2link, inter_1link, routed),
then by GCD ids. Tier color shows up as a faint x-axis background band
and as the x-tick-label color.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from _common import (
    PAIR_TIER, REP_PAIRS, STACK_COLORS, STACK_LABELS,
    TIER_COLORS, TIER_LABELS, TIER_ORDER,
    fmt_size, load_pt2pt, save,
)


def _agg_runs(df, value_col, group_cols):
    return (df.groupby(group_cols, as_index=False)
              .agg(median=(value_col, "median"),
                   lo=(value_col, "min"),
                   hi=(value_col, "max")))


def _draw_tier_bands(ax, pairs):
    for tier in TIER_ORDER:
        idx = [i for i, p in enumerate(pairs) if PAIR_TIER[p] == tier]
        if not idx:
            continue
        x0, x1 = min(idx) - 0.5, max(idx) + 0.5
        ax.axvspan(x0, x1, color=TIER_COLORS[tier], alpha=0.10, zorder=0)
        ax.text((x0 + x1) / 2, 1.02, TIER_LABELS[tier],
                transform=ax.get_xaxis_transform(),
                ha="center", va="bottom",
                color=TIER_COLORS[tier], fontsize=10, fontweight="bold")


def _color_pair_ticks(ax, pairs):
    for tick, p in zip(ax.get_xticklabels(), pairs):
        tick.set_color(TIER_COLORS[PAIR_TIER[p]])


def per_pair_bars(name, value_col, ylabel, divisor, fixed_size, size_label, fname):
    df = load_pt2pt(name)
    df = df[(df["size_bytes"] == fixed_size) & (df["pair_label"].isin(REP_PAIRS))].copy()
    if df.empty:
        print(f"  !! no {name} data at {size_label}; skipping")
        return
    df[value_col] = df[value_col] / divisor
    agg = _agg_runs(df, value_col, ["stack", "pair_label"])

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    _draw_tier_bands(ax, REP_PAIRS)

    width = 0.34
    x = np.arange(len(REP_PAIRS))
    for i, stack in enumerate(("eessi", "native")):
        s = (agg[agg["stack"] == stack]
                .set_index("pair_label")
                .reindex(REP_PAIRS))
        offset = (i - 0.5) * width
        med = s["median"].to_numpy()
        lo_err = (s["median"] - s["lo"]).to_numpy()
        hi_err = (s["hi"] - s["median"]).to_numpy()
        ax.bar(x + offset, med, width=width,
               color=STACK_COLORS[stack],
               label=STACK_LABELS[stack],
               edgecolor="black", linewidth=0.5,
               yerr=[lo_err, hi_err], capsize=4,
               error_kw={"linewidth": 0.9, "ecolor": "black"})

    ax.set_xticks(x)
    ax.set_xticklabels(REP_PAIRS, rotation=20, ha="right", fontsize=10)
    _color_pair_ticks(ax, REP_PAIRS)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{name} @ {size_label} — EESSI vs Native "
                 f"(median across 5 runs; error bars = min/max)",
                 pad=22)
    ax.legend(frameon=True, loc="best")
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    ax.margins(x=0.04)
    save(fig, fname)


def b4_tier_aggregated():
    df = load_pt2pt("bw")
    sizes = [1024, 1 << 20, 1 << 25]  # 1 KiB, 1 MiB, 32 MiB
    sub = df[df["size_bytes"].isin(sizes)].copy()
    sub["bw_GBps"] = sub["bandwidth_MBps"] / 1000.0

    # per-pair median across 5 runs first, then aggregate across pairs in tier.
    pair_med = (sub.groupby(["stack", "pair_label", "tier", "size_bytes"],
                            as_index=False)["bw_GBps"].median())
    agg = (pair_med.groupby(["stack", "tier", "size_bytes"], as_index=False)
                   .agg(median=("bw_GBps", "median"),
                        lo=("bw_GBps", "min"),
                        hi=("bw_GBps", "max")))

    fig, axes = plt.subplots(1, len(sizes), figsize=(14, 4.6), sharey=False)
    width = 0.38
    x = np.arange(len(TIER_ORDER))
    for ax, size in zip(axes, sizes):
        for i, stack in enumerate(("eessi", "native")):
            s = (agg[(agg["stack"] == stack) & (agg["size_bytes"] == size)]
                     .set_index("tier")
                     .reindex(TIER_ORDER))
            offset = (i - 0.5) * width
            med = s["median"].to_numpy()
            lo_err = (s["median"] - s["lo"]).to_numpy()
            hi_err = (s["hi"] - s["median"]).to_numpy()
            ax.bar(x + offset, med, width=width,
                   color=STACK_COLORS[stack],
                   label=STACK_LABELS[stack] if ax is axes[0] else None,
                   edgecolor="black", linewidth=0.4,
                   yerr=[lo_err, hi_err], capsize=3,
                   error_kw={"linewidth": 0.8, "ecolor": "black"})
        ax.set_xticks(x)
        ax.set_xticklabels([TIER_LABELS[t] for t in TIER_ORDER],
                           rotation=20, ha="right", fontsize=9)
        for tick, t in zip(ax.get_xticklabels(), TIER_ORDER):
            tick.set_color(TIER_COLORS[t])
        ax.set_title(f"@ {fmt_size(size)}")
        if ax is axes[0]:
            ax.set_ylabel("bandwidth (GB/s, median across pairs in tier)")
            ax.legend(frameon=True, loc="best")
        ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    fig.suptitle("Bandwidth by topology tier — EESSI vs Native "
                 "(error bars span the pair range)",
                 y=1.02, fontsize=13)
    save(fig, "B4_tier_aggregated_bw")


def main():
    print("[barcharts] B1 — per-pair bandwidth @ 1 MiB")
    per_pair_bars("bw", "bandwidth_MBps",
                  ylabel="bandwidth (GB/s)", divisor=1000.0,
                  fixed_size=1 << 20, size_label="1 MiB",
                  fname="B1_perpair_bw_1MiB")
    print("[barcharts] B2 — per-pair latency @ 8 B")
    per_pair_bars("latency", "latency_us",
                  ylabel="latency (us)", divisor=1.0,
                  fixed_size=8, size_label="8 B",
                  fname="B2_perpair_latency_8B")
    print("[barcharts] B4 — tier-aggregated bandwidth")
    b4_tier_aggregated()


if __name__ == "__main__":
    main()
