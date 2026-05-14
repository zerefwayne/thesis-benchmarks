"""B3d - per-pair small-multiples osu_bw curves, fixed_rndv variant.

Same 2x2 layout as B3a (per_pair_curves.py), but reads the fixed_rndv runs:
  EESSI side: UCX_RNDV_THRESH=1024
  Native side: matched fixed-threshold run for parity

Companion to B3a. Compare side-by-side to see what the threshold change
actually did. **Note**: looking at intra_pkg_OAM0, the small-msg cliff
that appears at 256 B in baseline B3a is NOT closed by raising the RNDV
threshold to 1024 - it shifts to 512 B (baseline 256 B = 30 MB/s -> 512 B
in fixed_rndv = 43 MB/s). The underlying UCX small-msg pathology persists;
only the size that falls into the bad regime moves.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from _common import (
    PAIR_TIER, REP_PAIRS, RESULTS, STACK_COLORS, STACK_LABELS,
    TIER_COLORS,
    median_runs, save, setup_log_axes,
)


NROWS, NCOLS = 2, 2


def _load_bw_fixed_rndv() -> pd.DataFrame:
    frames = []
    for stack in ("eessi", "native"):
        cands = sorted(RESULTS.glob(f"osu_bw_fixed_rndv_{stack}_*.csv"))
        if not cands:
            raise FileNotFoundError(
                f"No osu_bw_fixed_rndv_{stack}_*.csv in {RESULTS}"
            )
        f = cands[-1]
        df = pd.read_csv(f)
        df["stack"] = stack
        df["_source"] = f.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main():
    print("[per_pair_curves_fixed_rndv] B3d - osu_bw fixed_rndv per-pair curves")
    df = _load_bw_fixed_rndv()
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
        "osu_bw with fixed RNDV threshold (EESSI: UCX_RNDV_THRESH=1024) "
        "- EESSI vs Native, one pair per tier",
        fontsize=12, y=1.00,
    )
    fig.tight_layout()
    save(fig, "B3d_perpair_bw_curves_fixed_rndv")


if __name__ == "__main__":
    main()
