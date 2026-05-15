"""B3g - osu_bw on intra_pkg_OAM0: baseline vs UCX_RNDV_THRESH=128.

Single panel, 4 lines, log-log axes:
  EESSI baseline            (solid blue)
  EESSI UCX_RNDV_THRESH=128 (dashed blue)
  Native baseline           (solid red)
  Native UCX_RNDV_THRESH=128 (dashed red)

The threshold=128 run shows two effects compared to baseline B3a (one
pair, intra_pkg_OAM0): a NEW cliff at 128 B (the threshold-transition
cliff), AND the original 256 B-1 KiB slow zone is still present. Tuning
the threshold below the natural slow zone is strictly worse.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from _common import (
    RESULTS, STACK_COLORS, STACK_LABELS,
    median_runs, save, setup_log_axes,
)


PAIR = "intra_pkg_OAM0"

SERIES = [
    # (stack, variant, glob, linestyle, linewidth)
    ("eessi",  "baseline", "osu_bw_eessi_*.csv",         "-",  2.0),
    ("eessi",  "rndv_128", "osu_bw_rndv_128_eessi_*.csv", "--", 1.8),
    ("native", "baseline", "osu_bw_native_*.csv",         "-",  2.0),
    ("native", "rndv_128", "osu_bw_rndv_128_native_*.csv", "--", 1.8),
]


def _latest(pattern: str) -> pd.DataFrame:
    cands = sorted(RESULTS.glob(pattern))
    if not cands:
        raise FileNotFoundError(f"No match for {pattern} in {RESULTS}")
    return pd.read_csv(cands[-1])


def main():
    print("[rndv_128_compare] B3g - baseline vs RNDV_THRESH=128 on intra_pkg_OAM0")

    fig, ax = plt.subplots(figsize=(10, 6.5))

    for stack, variant, glob, ls, lw in SERIES:
        df = _latest(glob)
        df = df[df["pair_label"] == PAIR]
        med = (median_runs(df, "bandwidth_MBps", ["size_bytes"])
                  .sort_values("size_bytes"))
        ax.plot(med["size_bytes"], med["bandwidth_MBps"],
                color=STACK_COLORS[stack], linestyle=ls, linewidth=lw,
                marker="o", markersize=3.5,
                label=f"{STACK_LABELS[stack]} {variant}")

    setup_log_axes(ax, x=True, y=True)
    ax.set_xlabel("message size (bytes)")
    ax.set_ylabel("bandwidth (MB/s)")
    ax.set_title(
        f"osu_bw on {PAIR} - baseline vs UCX_RNDV_THRESH=128\n"
        "lowering the threshold adds a new cliff at 128 B without fixing "
        "the 256-1024 B slow zone"
    )
    ax.legend(fontsize=10, loc="lower right", frameon=True)

    # After autoscale: mark threshold + persistent slow zone
    ax.axvline(128, color="purple", linestyle=":", linewidth=1.2,
               alpha=0.7, zorder=1)
    ax.text(128, 0.02, "  THRESH=128",
            transform=ax.get_xaxis_transform(),
            color="purple", fontsize=9, ha="left", va="bottom",
            rotation=90, alpha=0.85)
    ax.axvspan(256, 1024, color="orange", alpha=0.10, zorder=0)
    ax.text((256 * 1024) ** 0.5, 0.92,
            "persistent slow zone\n(survives every threshold)",
            transform=ax.get_xaxis_transform(),
            color="#aa5500", fontsize=9, ha="center", va="top",
            style="italic")

    save(fig, "B3g_rndv_128_vs_baseline_OAM0")


if __name__ == "__main__":
    main()
