"""B3 — per-pair small-multiples line plot, both stacks overlaid.

2x2 grid of 4 panels — one representative pair per topology tier. Each
panel: log-log message size vs metric, EESSI and Native lines on the
same axes. Panel border colored by tier.

Run for bw, bibw, latency.
"""
from __future__ import annotations

import matplotlib.pyplot as plt

from _common import (
    PAIR_TIER, REP_PAIRS, STACK_COLORS, STACK_LABELS,
    TIER_COLORS,
    load_pt2pt, median_runs, save, setup_log_axes,
)


NROWS, NCOLS = 2, 2


def small_multiples(name, value_col, ylabel, fname, log_y=True):
    df = load_pt2pt(name)
    med = median_runs(df, value_col,
                      ["stack", "pair_label", "size_bytes"])

    fig, axes = plt.subplots(NROWS, NCOLS, figsize=(11, 8),
                             sharex=True, sharey=False)
    for r in range(NROWS):
        for c in range(NCOLS):
            ax = axes[r, c]
            i = r * NCOLS + c
            if i >= len(REP_PAIRS):
                ax.axis("off")
                continue
            pair = REP_PAIRS[i]
            tier = PAIR_TIER[pair]
            for stack in ("eessi", "native"):
                s = (med[(med["stack"] == stack)
                         & (med["pair_label"] == pair)]
                        .sort_values("size_bytes"))
                if s.empty:
                    continue
                ax.plot(s["size_bytes"], s[value_col],
                        color=STACK_COLORS[stack],
                        marker="o", markersize=3.5, linewidth=1.6,
                        label=STACK_LABELS[stack])
            setup_log_axes(ax, x=True, y=log_y)
            for spine in ax.spines.values():
                spine.set_color(TIER_COLORS[tier])
                spine.set_linewidth(1.8)
            ax.set_title(pair, color=TIER_COLORS[tier], fontsize=11)
            if r == NROWS - 1:
                ax.set_xlabel("message size (bytes)")
            if c == 0:
                ax.set_ylabel(ylabel)
            if r == 0 and c == 0:
                ax.legend(fontsize=10, loc="best", frameon=True)

    fig.suptitle(f"{name} — EESSI vs Native, one pair per tier",
                 fontsize=14, y=1.00)
    fig.tight_layout()
    save(fig, fname)


def main():
    print("[per_pair_curves] B3a — bw small multiples")
    small_multiples("bw", "bandwidth_MBps",
                    "bandwidth (MB/s)", "B3a_perpair_bw_curves")
    print("[per_pair_curves] B3b — bibw small multiples")
    small_multiples("bibw", "bandwidth_MBps",
                    "bandwidth (MB/s)", "B3b_perpair_bibw_curves")
    print("[per_pair_curves] B3c — latency small multiples")
    small_multiples("latency", "latency_us",
                    "latency (us)", "B3c_perpair_latency_curves")


if __name__ == "__main__":
    main()
