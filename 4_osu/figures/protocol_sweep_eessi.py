"""E1 - UCX_RNDV_THRESH sweep on intra_pkg_OAM0.

Single panel: osu_bw bandwidth vs message size, one line per swept threshold
(DEFAULT, 1024, 2048, 4096, 8192, 16384 bytes). Inset zooms into the small-
to-mid-message region where the threshold choice matters.

The story: UCX has a fixed small-message pathology in its eager path. Raising
UCX_RNDV_THRESH expands the size range where eager is used, so higher
thresholds hit MORE sizes with that pathology. THRESH=1024 minimizes the
affected size range; THRESH > 1024 widens the bad zone.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from _common import RESULTS, save, setup_log_axes


PAIR = "intra_pkg_OAM0"

# Threshold -> (color, linestyle, linewidth, label)
THRESH_STYLE = {
    "DEFAULT": ("#444444", "--", 1.6, "DEFAULT (UCX auto)"),
    "1024":    ("#1f77b4", "-",  2.4, "1024 (recommended)"),
    "2048":    ("#ff7f0e", "-",  1.4, "2048"),
    "4096":    ("#2ca02c", "-",  1.4, "4096"),
    "8192":    ("#d62728", "-",  1.4, "8192"),
    "16384":   ("#9467bd", "-",  1.4, "16384"),
}
THRESH_ORDER = ["DEFAULT", "1024", "2048", "4096", "8192", "16384"]


def _load():
    cands = sorted(RESULTS.glob("osu_protocol_eessi_*.csv"))
    if not cands:
        raise FileNotFoundError(f"No osu_protocol_eessi_*.csv in {RESULTS}")
    f = cands[-1]
    print(f"  reading {f.name}")
    return pd.read_csv(f)


def main():
    print("[protocol_sweep_eessi] E1 - UCX_RNDV_THRESH sweep")
    df = _load()
    df = df[df["pair_label"] == PAIR]
    df["ucx_rndv_thresh"] = df["ucx_rndv_thresh"].astype(str)
    med = (df.groupby(["ucx_rndv_thresh", "size_bytes"], as_index=False)
              ["bandwidth_MBps"].median())

    fig, ax = plt.subplots(figsize=(11, 6.5))

    for thr in THRESH_ORDER:
        sub = (med[med["ucx_rndv_thresh"] == thr]
                  .sort_values("size_bytes"))
        if sub.empty:
            continue
        color, ls, lw, label = THRESH_STYLE[thr]
        ax.plot(sub["size_bytes"], sub["bandwidth_MBps"],
                color=color, linestyle=ls, linewidth=lw,
                marker="o", markersize=3.5, label=label)

    setup_log_axes(ax, x=True, y=True)
    ax.set_xlabel("message size (bytes)")
    ax.set_ylabel("bandwidth (MB/s)")
    ax.set_title(
        f"osu_bw on {PAIR}, EESSI - UCX_RNDV_THRESH sweep\n"
        "raising the threshold widens the size range hit by UCX's eager-path cliff"
    )
    ax.legend(title="UCX_RNDV_THRESH", fontsize=10, frameon=True,
              loc="lower right", title_fontsize=10)

    # Shade the "danger zone" 256 B -> 32 KiB where threshold choice matters most
    ax.axvspan(256, 32768, color="orange", alpha=0.07, zorder=0)
    ax.text(0.5, 0.97,
            "shaded band: range where eager-path cliff degrades bandwidth\n"
            "higher threshold = wider band of degraded sizes",
            transform=ax.transAxes, va="top", ha="center",
            fontsize=10, style="italic", color="#555555",
            bbox=dict(facecolor="white", alpha=0.92, edgecolor="gray",
                      boxstyle="round,pad=0.4"))

    save(fig, "E1_protocol_sweep_eessi")


if __name__ == "__main__":
    main()
