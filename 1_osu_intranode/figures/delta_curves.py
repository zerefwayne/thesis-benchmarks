"""D1 — signed % delta across message sizes.

For each metric (bw, bibw, latency), one panel: x = log message size,
y = signed % delta. Convention: positive = EESSI advantage.

  bw:      delta% = (EESSI_MBps - Native_MBps) / Native_MBps * 100
  bibw:    same
  latency: delta% = (Native_us  - EESSI_us)   / Native_us   * 100
           (inverted so positive still means EESSI advantage)

One line per tier representative pair (4 lines per panel). The panel is
shaded green above 0 (EESSI advantage zone) and red below 0 (Native
advantage zone) so the eye registers sign before reading numbers.
"""
from __future__ import annotations

import matplotlib.pyplot as plt

from _common import (
    PAIR_TIER, REP_PAIRS, TIER_COLORS, TIER_LABELS, TIER_ORDER,
    load_pt2pt, median_runs, save, setup_log_axes, stack_pivot,
)


SPECS = [
    # (name, value_col, lower_is_better, title, y_clip)
    # y_clip: (ymin, ymax) for the panel. Latency clipped because the
    # small-msg cliff drives EESSI advantage to ~-1700% which crushes the
    # rest of the curve into a flat line.
    ("bw",      "bandwidth_MBps", False, "osu_bw",      None),
    ("bibw",    "bandwidth_MBps", False, "osu_bibw",    None),
    ("latency", "latency_us",     True,  "osu_latency", (-200, 60)),
]


def _delta_pct(name, value_col, lower_is_better):
    df = load_pt2pt(name)
    df = df[df["pair_label"].isin(REP_PAIRS)]
    med = median_runs(df, value_col, ["stack", "pair_label", "size_bytes"])
    wide = stack_pivot(med, ["pair_label", "size_bytes"], value_col)
    wide = wide.dropna(subset=["eessi", "native"])
    wide = wide[wide["native"] > 0]
    if lower_is_better:
        wide["delta_pct"] = (wide["native"] - wide["eessi"]) / wide["native"] * 100
    else:
        wide["delta_pct"] = (wide["eessi"] - wide["native"]) / wide["native"] * 100
    return wide


def _shade_signs(ax):
    ymin, ymax = ax.get_ylim()
    if ymax > 0:
        ax.axhspan(0, ymax, color="#2ca02c", alpha=0.06, zorder=0)
    if ymin < 0:
        ax.axhspan(ymin, 0, color="#d62728", alpha=0.06, zorder=0)
    ax.set_ylim(ymin, ymax)


def main():
    print("[delta_curves] D1 — signed % delta vs message size")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=False)
    for ax, (name, vcol, lib, title, y_clip) in zip(axes, SPECS):
        wide = _delta_pct(name, vcol, lib)
        for pair in REP_PAIRS:
            sub = wide[wide["pair_label"] == pair].sort_values("size_bytes")
            if sub.empty:
                continue
            tier = PAIR_TIER[pair]
            ax.plot(sub["size_bytes"], sub["delta_pct"],
                    color=TIER_COLORS[tier], linewidth=1.8,
                    marker="o", markersize=3.5,
                    label=pair)
        ax.axhline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
        setup_log_axes(ax, x=True, y=False)
        if y_clip is not None:
            ax.set_ylim(*y_clip)
            ax.text(0.98, 0.04,
                    f"y-axis clipped to [{y_clip[0]}, {y_clip[1]}]%\n"
                    "(small-msg cliff dives below)",
                    transform=ax.transAxes, va="bottom", ha="right",
                    fontsize=8, style="italic", color="gray",
                    bbox=dict(facecolor="white", alpha=0.85,
                              edgecolor="none", boxstyle="round,pad=0.25"))
        _shade_signs(ax)
        ax.set_xlabel("message size (bytes)")
        ax.set_ylabel("EESSI advantage (%)" if ax is axes[0] else "")
        ax.set_title(title)

    handles = [plt.Line2D([], [], color=TIER_COLORS[t], lw=2.2, label=TIER_LABELS[t])
               for t in TIER_ORDER]
    handles.append(plt.Line2D([], [], color="black", lw=1, linestyle="--",
                              label="parity (delta = 0)"))
    fig.legend(handles=handles, loc="upper center", ncol=5,
               bbox_to_anchor=(0.5, 1.05), frameon=False, fontsize=10)
    fig.suptitle(
        "Signed % delta across message sizes — positive bands favor EESSI, negative bands favor Native",
        fontsize=13, y=1.13,
    )
    save(fig, "D1_delta_pct_curves")


if __name__ == "__main__":
    main()
