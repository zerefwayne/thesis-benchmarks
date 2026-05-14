"""D2 — sorted signed % delta bar charts (tornado-style).

For each (metric, fixed size), one horizontal bar chart of all 12 pairs
sorted by delta. Bar color: green if EESSI is faster, red if slower.
Pair-tick labels colored by tier so the reader can see whether the gap
correlates with topology.

Convention: positive = EESSI advantage.
"""
from __future__ import annotations

import matplotlib.pyplot as plt

from _common import (
    PAIR_ORDER, PAIR_TIER, TIER_COLORS,
    load_pt2pt, median_runs, save, stack_pivot,
)


POS_COLOR = "#2ca02c"  # EESSI faster
NEG_COLOR = "#d62728"  # EESSI slower


def tornado(name, value_col, lower_is_better, fixed_size, size_label, fname):
    df = load_pt2pt(name)
    df = df[(df["size_bytes"] == fixed_size)
            & (df["pair_label"].isin(PAIR_ORDER))]
    if df.empty:
        print(f"  !! no {name} data at {size_label}; skipping")
        return
    med = median_runs(df, value_col, ["stack", "pair_label"])
    wide = stack_pivot(med, ["pair_label"], value_col)
    wide = wide.dropna(subset=["eessi", "native"])
    wide = wide[wide["native"] > 0]
    if lower_is_better:
        wide["delta_pct"] = (wide["native"] - wide["eessi"]) / wide["native"] * 100
    else:
        wide["delta_pct"] = (wide["eessi"] - wide["native"]) / wide["native"] * 100
    wide = wide.sort_values("delta_pct").reset_index(drop=True)

    colors = [POS_COLOR if d >= 0 else NEG_COLOR for d in wide["delta_pct"]]
    n = len(wide)

    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.4 * n)))
    bars = ax.barh(range(n), wide["delta_pct"], color=colors,
                   edgecolor="black", linewidth=0.4)
    ax.axvline(0, color="black", linewidth=1.0)
    ax.set_yticks(range(n))
    ax.set_yticklabels(wide["pair_label"])
    for tick, p in zip(ax.get_yticklabels(), wide["pair_label"]):
        tick.set_color(TIER_COLORS[PAIR_TIER[p]])

    xmax = max(abs(wide["delta_pct"].min()), abs(wide["delta_pct"].max()))
    pad = 0.16 * xmax if xmax > 0 else 1.0
    ax.set_xlim(-xmax - pad, xmax + pad)

    for bar, d in zip(bars, wide["delta_pct"]):
        x = bar.get_width()
        ha = "left" if x >= 0 else "right"
        offset = pad * 0.10 if x >= 0 else -pad * 0.10
        ax.text(x + offset, bar.get_y() + bar.get_height() / 2,
                f"{d:+.1f}%", va="center", ha=ha, fontsize=9)

    ax.set_xlabel("EESSI advantage (%) — positive: EESSI faster")
    ax.set_title(f"{name} @ {size_label} — signed % delta, sorted ({n} pairs)")
    ax.grid(True, axis="x", linestyle=":", alpha=0.4)

    # Legend (color = sign)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=POS_COLOR, label="EESSI faster"),
        plt.Rectangle((0, 0), 1, 1, color=NEG_COLOR, label="Native faster"),
    ]
    ax.legend(handles=handles, frameon=True, loc="lower right")
    save(fig, fname)


def main():
    print("[delta_bars] D2a — bw % delta @ 1 MiB")
    tornado("bw", "bandwidth_MBps", False, 1 << 20, "1 MiB",
            "D2a_tornado_bw_1MiB")
    print("[delta_bars] D2b — bw % delta @ 32 MiB")
    tornado("bw", "bandwidth_MBps", False, 1 << 25, "32 MiB",
            "D2b_tornado_bw_32MiB")
    print("[delta_bars] D2c — latency % delta @ 8 B")
    tornado("latency", "latency_us", True, 8, "8 B",
            "D2c_tornado_latency_8B")
    print("[delta_bars] D2d — latency % delta @ 1 KiB")
    tornado("latency", "latency_us", True, 1024, "1 KiB",
            "D2d_tornado_latency_1KiB")


if __name__ == "__main__":
    main()
