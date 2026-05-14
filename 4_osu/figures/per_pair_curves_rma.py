"""B3e, B3f - per-pair small-multiples osu_put_bw / osu_get_bw curves.

Same 2x2 layout as B3a but for the MPI one-sided RMA primitives:
  osu_put_bw - origin rank issues MPI_Put (writes into target's window)
  osu_get_bw - origin rank issues MPI_Get (reads from target's window)

Both stacks overlaid per panel; one panel per tier representative pair.
Compare against B3a (two-sided osu_bw) to see whether the stack treats
RMA paths comparably to send/recv on the same hardware.
"""
from __future__ import annotations

import matplotlib.pyplot as plt

from _common import (
    PAIR_TIER, REP_PAIRS, STACK_COLORS, STACK_LABELS,
    TIER_COLORS,
    load_pt2pt, median_runs, save, setup_log_axes,
)


NROWS, NCOLS = 2, 2


def small_multiples(name: str, fname: str, title_prefix: str):
    df = load_pt2pt(name)
    med = median_runs(df, "bandwidth_MBps",
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
                ax.plot(s["size_bytes"], s["bandwidth_MBps"],
                        color=STACK_COLORS[stack],
                        marker="o", markersize=3.5, linewidth=1.6,
                        label=STACK_LABELS[stack])
            setup_log_axes(ax, x=True, y=True)
            for spine in ax.spines.values():
                spine.set_color(TIER_COLORS[tier])
                spine.set_linewidth(1.8)
            ax.set_title(pair, color=TIER_COLORS[tier], fontsize=11)
            if r == NROWS - 1:
                ax.set_xlabel("message size (bytes)")
            if c == 0:
                ax.set_ylabel("bandwidth (MB/s)")
            if r == 0 and c == 0:
                ax.legend(fontsize=10, loc="best", frameon=True)

    fig.suptitle(
        f"{title_prefix} - EESSI vs Native, one pair per tier",
        fontsize=13, y=1.00,
    )
    fig.tight_layout()
    save(fig, fname)


def main():
    print("[per_pair_curves_rma] B3e - osu_put_bw per-pair curves")
    small_multiples(
        "put_bw",
        "B3e_perpair_put_bw_curves",
        "osu_put_bw (MPI_Put one-sided write)",
    )
    print("[per_pair_curves_rma] B3f - osu_get_bw per-pair curves")
    small_multiples(
        "get_bw",
        "B3f_perpair_get_bw_curves",
        "osu_get_bw (MPI_Get one-sided read)",
    )


if __name__ == "__main__":
    main()
