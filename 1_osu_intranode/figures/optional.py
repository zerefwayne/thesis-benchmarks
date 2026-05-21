"""I1, I2 — optional / advanced figures.

I1: run-variability plot for one representative pair (intra_pkg_OAM0):
    5 individual runs as faint lines + median bold, EESSI vs Native.
I2: latency vs bandwidth Pareto: small-msg latency vs peak bw,
    one marker per (pair, stack), tier as marker shape.
"""
from __future__ import annotations

import matplotlib.pyplot as plt

from _common import (
    PAIR_TIER, REP_PAIRS, STACK_COLORS, STACK_LABELS,
    TIER_COLORS, TIER_LABELS, TIER_ORDER,
    fmt_size, load_pt2pt, median_runs, save, setup_log_axes,
)


REP_PAIR = "intra_pkg_OAM0"
SMALL_SIZE = 8       # 8 B for small-msg latency
PEAK_SIZE = 1 << 25  # 32 MiB for peak bandwidth

TIER_MARKERS = {
    "intra_pkg":       "o",
    "inter_pkg_2link": "s",
    "inter_pkg_1link": "^",
    "routed":          "D",
}


def i1_variability():
    df = load_pt2pt("bw")
    sub = df[df["pair_label"] == REP_PAIR]
    if sub.empty:
        print(f"  !! {REP_PAIR} missing; skipping I1")
        return
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for stack in ("eessi", "native"):
        s = sub[sub["stack"] == stack]
        # individual runs
        for run, g in s.groupby("run"):
            g = g.sort_values("size_bytes")
            ax.plot(g["size_bytes"], g["bandwidth_MBps"],
                    color=STACK_COLORS[stack], alpha=0.25, linewidth=0.9)
        # median
        med = (s.groupby("size_bytes", as_index=False)["bandwidth_MBps"]
                .median()
                .sort_values("size_bytes"))
        ax.plot(med["size_bytes"], med["bandwidth_MBps"],
                color=STACK_COLORS[stack], linewidth=2.0,
                label=f"{STACK_LABELS[stack]} (median)")
    setup_log_axes(ax, x=True, y=True)
    ax.set_xlabel("message size (bytes)")
    ax.set_ylabel("bandwidth (MB/s)")
    ax.set_title(f"Run variability — osu_bw on {REP_PAIR} (5 recorded runs + median)")
    ax.legend(frameon=True)
    save(fig, "I1_variability_intra_pkg_OAM0")


def i2_pareto():
    bw = load_pt2pt("bw")
    lat = load_pt2pt("latency")

    bw_med = median_runs(bw, "bandwidth_MBps",
                         ["stack", "pair_label", "size_bytes"])
    lat_med = median_runs(lat, "latency_us",
                          ["stack", "pair_label", "size_bytes"])

    bw_peak = bw_med[bw_med["size_bytes"] == PEAK_SIZE]
    lat_small = lat_med[lat_med["size_bytes"] == SMALL_SIZE]

    fig, ax = plt.subplots(figsize=(8.5, 6))
    seen_legend = set()
    for pair in REP_PAIRS:
        tier = PAIR_TIER[pair]
        for stack in ("eessi", "native"):
            x = lat_small[(lat_small["pair_label"] == pair)
                          & (lat_small["stack"] == stack)]
            y = bw_peak[(bw_peak["pair_label"] == pair)
                        & (bw_peak["stack"] == stack)]
            if x.empty or y.empty:
                continue
            ax.scatter(x["latency_us"].iloc[0], y["bandwidth_MBps"].iloc[0],
                       marker=TIER_MARKERS[tier],
                       color=STACK_COLORS[stack],
                       s=110, alpha=0.85, edgecolor="black", linewidth=0.6)
            tag = (stack, tier)
            if tag not in seen_legend:
                seen_legend.add(tag)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"small-msg latency @ {fmt_size(SMALL_SIZE)} (us, log)")
    ax.set_ylabel(f"peak bandwidth @ {fmt_size(PEAK_SIZE)} (MB/s, log)")
    ax.set_title("Intranode pt2pt characteristics — latency vs bandwidth Pareto")
    ax.grid(True, which="both", linestyle=":", alpha=0.4)

    # Combined legend: stack color + tier shape
    handles = [plt.Line2D([], [], marker="o", linestyle="",
                          color=STACK_COLORS[s], markersize=9,
                          label=STACK_LABELS[s]) for s in ("eessi", "native")]
    for t in TIER_ORDER:
        handles.append(plt.Line2D([], [], marker=TIER_MARKERS[t], linestyle="",
                                  color="gray", markersize=9,
                                  label=TIER_LABELS[t]))
    ax.legend(handles=handles, loc="lower left", fontsize=9, frameon=True,
              ncol=2)
    save(fig, "I2_latency_bw_pareto")


def main():
    print("[optional] I1 — run variability")
    i1_variability()
    print("[optional] I2 — latency vs bandwidth Pareto")
    i2_pareto()


if __name__ == "__main__":
    main()
