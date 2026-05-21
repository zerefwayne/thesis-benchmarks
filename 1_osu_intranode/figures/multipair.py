"""G1, G2, G3 — multi-pair (osu_mbw_mr) plots.

Each `config_label` is a different concurrent-pair scenario (4 pairs each):
  cfg_intra_pkg     — 4 intra-package pairs in parallel (ideal)
  cfg_mixed_1link   — 4 inter-package 1-link pairs in parallel
  cfg_routed_split  — 4 routed pairs in parallel (worst case)

G1: aggregate bandwidth vs message size, line per config, panel per stack.
G2: message rate vs message size, same layout.
G3: peak aggregate bandwidth bar chart (config x stack).
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from _common import (
    STACK_COLORS, STACK_LABELS,
    fmt_size, load_mbw_mr, median_runs, save, setup_log_axes,
)


CONFIG_ORDER = ["cfg_intra_pkg", "cfg_mixed_1link", "cfg_routed_split"]
CONFIG_LABELS = {
    "cfg_intra_pkg":    "intra-pkg x4",
    "cfg_mixed_1link":  "inter-pkg 1-link x4",
    "cfg_routed_split": "routed x4",
}
CONFIG_COLORS = {
    "cfg_intra_pkg":    "#1f77b4",
    "cfg_mixed_1link":  "#ff7f0e",
    "cfg_routed_split": "#d62728",
}


def _curves(value_col, ylabel, log_y, title, fname):
    df = load_mbw_mr()
    med = median_runs(df, value_col,
                      ["stack", "config_label", "size_bytes"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, stack in zip(axes, ("eessi", "native")):
        s = med[med["stack"] == stack]
        for cfg in CONFIG_ORDER:
            ss = s[s["config_label"] == cfg].sort_values("size_bytes")
            if ss.empty:
                continue
            ax.plot(ss["size_bytes"], ss[value_col],
                    marker="o", markersize=2.5, linewidth=1.4,
                    color=CONFIG_COLORS[cfg],
                    label=CONFIG_LABELS[cfg])
        setup_log_axes(ax, x=True, y=log_y)
        ax.set_title(STACK_LABELS[stack])
        ax.set_xlabel("message size (bytes)")
        ax.legend(fontsize=9, frameon=True, loc="best")
        if ax is axes[0]:
            ax.set_ylabel(ylabel)
    fig.suptitle(title, fontsize=13, y=1.02)
    save(fig, fname)


def g3_peak_bars():
    df = load_mbw_mr()
    df["bw_GBps"] = df["bandwidth_MBps"] / 1000.0
    med = median_runs(df, "bw_GBps",
                      ["stack", "config_label", "size_bytes"])
    # Use the largest message size present as "peak".
    peak_size = int(med["size_bytes"].max())
    sub = med[med["size_bytes"] == peak_size]

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    width = 0.36
    x = np.arange(len(CONFIG_ORDER))
    for i, stack in enumerate(("eessi", "native")):
        vals = [
            sub[(sub["stack"] == stack) & (sub["config_label"] == c)]["bw_GBps"]
                .iloc[0] if not sub[(sub["stack"] == stack) & (sub["config_label"] == c)].empty else np.nan
            for c in CONFIG_ORDER
        ]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, vals, width=width,
                      color=STACK_COLORS[stack],
                      label=STACK_LABELS[stack],
                      edgecolor="black", linewidth=0.5)
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v,
                        f"{v:.1f} GB/s",
                        ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([CONFIG_LABELS[c] for c in CONFIG_ORDER])
    ax.set_ylabel("aggregate bandwidth (GB/s)")
    ax.set_title(f"osu_mbw_mr — aggregate bandwidth @ {fmt_size(peak_size)} (4 concurrent pairs)")
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    ax.legend(frameon=True)
    save(fig, "G3_mbw_mr_peak_bars")


def main():
    print("[multipair] G1 — aggregate bandwidth vs size")
    _curves("bandwidth_MBps", "aggregate bandwidth (MB/s)", True,
            "osu_mbw_mr — aggregate bandwidth (4 concurrent pairs)",
            "G1_mbw_mr_bandwidth_curves")
    print("[multipair] G2 — message rate vs size")
    _curves("msg_rate_Mps", "message rate (msgs/s)", True,
            "osu_mbw_mr — message rate (4 concurrent pairs)",
            "G2_mbw_mr_message_rate_curves")
    print("[multipair] G3 — peak aggregate bandwidth bars")
    g3_peak_bars()


if __name__ == "__main__":
    main()
