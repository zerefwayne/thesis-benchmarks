"""P2 — per-pair peak-bandwidth bars with theoretical IF peak references.

Paper-Fig-4 style. All 12 pairs on the x-axis, two bars per pair (EESSI,
Native) at the peak measured msg size (32 MiB). Above each pair group, a
short tier-colored dashed segment marks that tier's nominal IF peak. Bar
text annotates measured GB/s. Tier color shows up as a faint x-axis
background band and as the x-tick label color.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from _common import (
    PAIR_ORDER, PAIR_TIER, STACK_COLORS, STACK_LABELS,
    TIER_COLORS, TIER_LABELS, TIER_ORDER, TIER_PEAK_GBPS,
    fmt_size, load_pt2pt, save,
)


PEAK_SIZE = 1 << 25  # 32 MiB


def main():
    print("[peak_efficiency_bars] P2 — per-pair bandwidth + theoretical peak")
    df = load_pt2pt("bw")
    df = df[df["size_bytes"] == PEAK_SIZE].copy()
    if df.empty:
        print(f"  !! no bw data at {fmt_size(PEAK_SIZE)}; skipping")
        return
    df["bw_GBps"] = df["bandwidth_MBps"] / 1000.0
    agg = (df.groupby(["stack", "pair_label"], as_index=False)
              .agg(median=("bw_GBps", "median"),
                   lo=("bw_GBps", "min"),
                   hi=("bw_GBps", "max")))

    fig, ax = plt.subplots(figsize=(15, 6.5))

    # Tier background bands + headers
    for tier in TIER_ORDER:
        idx = [i for i, p in enumerate(PAIR_ORDER) if PAIR_TIER[p] == tier]
        if not idx:
            continue
        x0, x1 = min(idx) - 0.5, max(idx) + 0.5
        ax.axvspan(x0, x1, color=TIER_COLORS[tier], alpha=0.10, zorder=0)
        ax.text((x0 + x1) / 2, 1.02, TIER_LABELS[tier],
                transform=ax.get_xaxis_transform(),
                ha="center", va="bottom",
                color=TIER_COLORS[tier], fontsize=10, fontweight="bold")

    width = 0.36
    x = np.arange(len(PAIR_ORDER))
    for i, stack in enumerate(("eessi", "native")):
        s = (agg[agg["stack"] == stack]
                .set_index("pair_label")
                .reindex(PAIR_ORDER))
        offset = (i - 0.5) * width
        med = s["median"].to_numpy()
        lo_err = (s["median"] - s["lo"]).to_numpy()
        hi_err = (s["hi"] - s["median"]).to_numpy()
        ax.bar(x + offset, med, width=width,
               color=STACK_COLORS[stack],
               edgecolor="black", linewidth=0.5,
               yerr=[lo_err, hi_err], capsize=2.5,
               error_kw={"linewidth": 0.7, "ecolor": "black"},
               zorder=2)

    # Per-pair theoretical peak: short tier-colored dashed segment above each pair
    for j, pair in enumerate(PAIR_ORDER):
        tier = PAIR_TIER[pair]
        peak = TIER_PEAK_GBPS[tier]
        if peak is None:
            ax.text(j, 5, "no peak\n(routed)",
                    ha="center", va="bottom", fontsize=7,
                    style="italic", color="gray", zorder=4)
            continue
        ax.hlines(peak, j - width * 1.1, j + width * 1.1,
                  color=TIER_COLORS[tier], linestyle="--", linewidth=1.8,
                  zorder=4)

    # Bar value annotations
    for i, stack in enumerate(("eessi", "native")):
        s = (agg[agg["stack"] == stack]
                .set_index("pair_label")
                .reindex(PAIR_ORDER))
        offset = (i - 0.5) * width
        for j, v in enumerate(s["median"].to_numpy()):
            if np.isnan(v):
                continue
            ax.text(j + offset, v + 1.0, f"{v:.0f}",
                    ha="center", va="bottom",
                    fontsize=7, color="black", rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels(PAIR_ORDER, rotation=45, ha="right", fontsize=8)
    for tick, p in zip(ax.get_xticklabels(), PAIR_ORDER):
        tick.set_color(TIER_COLORS[PAIR_TIER[p]])
    ax.set_ylabel("bandwidth (GB/s)")
    ax.set_title(f"osu_bw @ {fmt_size(PEAK_SIZE)} — measured vs nominal IF peak per pair "
                 "(dashed segment = nominal peak for that tier)",
                 pad=24)

    handles = [
        Patch(facecolor=STACK_COLORS["eessi"], edgecolor="black",
              label=STACK_LABELS["eessi"]),
        Patch(facecolor=STACK_COLORS["native"], edgecolor="black",
              label=STACK_LABELS["native"]),
        plt.Line2D([], [], color="black", linestyle="--", linewidth=1.8,
                   label="nominal IF peak (per tier)"),
    ]
    ax.legend(handles=handles, frameon=True, loc="upper right")
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    ax.margins(x=0.01)

    # Pad y-axis so the dashed peak markers are visible above the tallest tier
    ymax = max(p for p in TIER_PEAK_GBPS.values() if p is not None)
    ax.set_ylim(0, ymax * 1.10)

    save(fig, "P2_peak_efficiency_bars")


if __name__ == "__main__":
    main()
