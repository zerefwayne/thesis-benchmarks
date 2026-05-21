"""P1 — per-tier bandwidth curves with theoretical peak reference.

Paper-Fig-3 style. 1x4 panels (one per tier representative pair). Each panel:
  * x = log message size, y = bandwidth (GB/s)
  * 2 lines (EESSI, Native) overlaid on the same axes
  * dashed horizontal at the IF nominal peak (for tiers with a defined peak)
  * inset axis showing latency for small messages (<= 1 KiB)
  * annotation reporting each stack's % of nominal at the peak msg size
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from _common import (
    PAIR_TIER, REP_PAIRS, STACK_COLORS, STACK_LABELS,
    TIER_COLORS, TIER_PEAK_GBPS,
    fmt_size, load_pt2pt, median_runs, save, setup_log_axes,
)


PEAK_SIZE = 1 << 25       # 32 MiB
SMALL_LATENCY_MAX = 1024  # inset shows sizes <= 1 KiB


def main():
    print("[efficiency_curves] P1 — per-tier bandwidth curves vs theoretical peak")
    bw = load_pt2pt("bw")
    bw_med = median_runs(bw, "bandwidth_MBps",
                         ["stack", "pair_label", "size_bytes"])
    bw_med["bw_GBps"] = bw_med["bandwidth_MBps"] / 1000.0

    lat = load_pt2pt("latency")
    lat_med = median_runs(lat, "latency_us",
                          ["stack", "pair_label", "size_bytes"])

    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    for ax, pair in zip(axes.flat, REP_PAIRS):
        tier = PAIR_TIER[pair]
        peak = TIER_PEAK_GBPS[tier]

        # Bandwidth curves
        for stack in ("eessi", "native"):
            s = (bw_med[(bw_med["stack"] == stack)
                        & (bw_med["pair_label"] == pair)]
                    .sort_values("size_bytes"))
            if s.empty:
                continue
            ax.plot(s["size_bytes"], s["bw_GBps"],
                    color=STACK_COLORS[stack], linewidth=1.8,
                    marker="o", markersize=3.5,
                    label=STACK_LABELS[stack], zorder=3)

        # Theoretical peak reference + % of peak annotation
        if peak is not None:
            ax.axhline(peak, color="black", linestyle="--",
                       linewidth=1.3, alpha=0.75,
                       label=f"IF peak = {peak:.0f} GB/s", zorder=2)
            row_e = bw_med[(bw_med["stack"] == "eessi")
                           & (bw_med["pair_label"] == pair)
                           & (bw_med["size_bytes"] == PEAK_SIZE)]
            row_n = bw_med[(bw_med["stack"] == "native")
                           & (bw_med["pair_label"] == pair)
                           & (bw_med["size_bytes"] == PEAK_SIZE)]
            if not row_e.empty and not row_n.empty:
                pct_e = float(row_e["bw_GBps"].iloc[0]) / peak * 100
                pct_n = float(row_n["bw_GBps"].iloc[0]) / peak * 100
                ax.text(0.98, 0.97,
                        f"@ {fmt_size(PEAK_SIZE)}:\n"
                        f"EESSI {pct_e:.0f}% of peak\n"
                        f"Native {pct_n:.0f}% of peak",
                        transform=ax.transAxes, va="top", ha="right",
                        fontsize=9,
                        bbox=dict(facecolor="white", alpha=0.92,
                                  edgecolor="gray",
                                  boxstyle="round,pad=0.35"))
        else:
            ax.text(0.98, 0.97, "routed:\npath-dependent\npeak",
                    transform=ax.transAxes, va="top", ha="right",
                    fontsize=9, style="italic", color="gray",
                    bbox=dict(facecolor="white", alpha=0.92,
                              edgecolor="gray",
                              boxstyle="round,pad=0.35"))

        setup_log_axes(ax, x=True, y=False)
        for spine in ax.spines.values():
            spine.set_color(TIER_COLORS[tier])
            spine.set_linewidth(1.8)
        ax.set_title(pair, color=TIER_COLORS[tier], fontsize=11)
        ax.set_xlabel("message size (bytes)")
        ax.set_ylabel("bandwidth (GB/s)")
        ax.legend(fontsize=8, loc="lower right", frameon=True)

        # Latency inset (upper-left of each panel)
        ins = inset_axes(ax, width="38%", height="32%",
                         loc="upper left", borderpad=1.2)
        for stack in ("eessi", "native"):
            s = (lat_med[(lat_med["stack"] == stack)
                         & (lat_med["pair_label"] == pair)
                         & (lat_med["size_bytes"] <= SMALL_LATENCY_MAX)]
                    .sort_values("size_bytes"))
            if s.empty:
                continue
            ins.plot(s["size_bytes"], s["latency_us"],
                     color=STACK_COLORS[stack],
                     linewidth=1.2, marker=".", markersize=3)
        ins.set_xscale("log", base=2)
        ins.set_xlabel("size", fontsize=7)
        ins.set_ylabel("lat (us)", fontsize=7)
        ins.tick_params(labelsize=7)
        ins.grid(True, linestyle=":", alpha=0.4)
        ins.set_title("small-msg latency", fontsize=7, pad=2)

    fig.suptitle(
        "Bandwidth vs message size — both stacks reach for the same hardware ceiling "
        f"(IF peaks marked; % of peak reported at {fmt_size(PEAK_SIZE)})",
        fontsize=12, y=1.02,
    )
    save(fig, "P1_efficiency_curves")


if __name__ == "__main__":
    main()
