"""B5, B6 — per-run distribution plots ("is the noise small?").

For each (pair, stack), plot all 5 recorded runs as dots with horizontal
jitter, plus a horizontal median line. Side-by-side EESSI/Native at each
pair lets the reader judge whether the head-to-head difference is real or
falls inside run-to-run noise.

B5: bandwidth @ 1 MiB.
B6: latency  @ 8 B.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from _common import (
    PAIR_TIER, REP_PAIRS, STACK_COLORS, STACK_LABELS,
    TIER_COLORS, TIER_LABELS, TIER_ORDER,
    load_pt2pt, save,
)


def _draw_tier_bands(ax, pairs):
    for tier in TIER_ORDER:
        idx = [i for i, p in enumerate(pairs) if PAIR_TIER[p] == tier]
        if not idx:
            continue
        x0, x1 = min(idx) - 0.5, max(idx) + 0.5
        ax.axvspan(x0, x1, color=TIER_COLORS[tier], alpha=0.10, zorder=0)
        ax.text((x0 + x1) / 2, 1.02, TIER_LABELS[tier],
                transform=ax.get_xaxis_transform(),
                ha="center", va="bottom",
                color=TIER_COLORS[tier], fontsize=10, fontweight="bold")


def _color_pair_ticks(ax, pairs):
    for tick, p in zip(ax.get_xticklabels(), pairs):
        tick.set_color(TIER_COLORS[PAIR_TIER[p]])


def strip_per_pair(name, value_col, ylabel, divisor, fixed_size, size_label, fname):
    df = load_pt2pt(name)
    df = df[(df["size_bytes"] == fixed_size) & (df["pair_label"].isin(REP_PAIRS))].copy()
    if df.empty:
        print(f"  !! no {name} data at {size_label}; skipping")
        return
    df["v"] = df[value_col] / divisor

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    _draw_tier_bands(ax, REP_PAIRS)

    width = 0.36
    x = np.arange(len(REP_PAIRS))
    rng = np.random.default_rng(42)

    legend_seen = set()
    for i, stack in enumerate(("eessi", "native")):
        offset = (i - 0.5) * width
        for j, pair in enumerate(REP_PAIRS):
            sub = df[(df["stack"] == stack) & (df["pair_label"] == pair)]
            if sub.empty:
                continue
            xpos = x[j] + offset
            jitter = rng.uniform(-0.07, 0.07, size=len(sub))
            label = STACK_LABELS[stack] if stack not in legend_seen else None
            ax.scatter(np.full(len(sub), xpos) + jitter, sub["v"],
                       color=STACK_COLORS[stack], alpha=0.8, s=42,
                       edgecolor="black", linewidth=0.5,
                       label=label, zorder=2)
            ax.hlines(sub["v"].median(),
                      xpos - 0.13, xpos + 0.13,
                      color=STACK_COLORS[stack], linewidth=2.6, zorder=3)
            legend_seen.add(stack)

    ax.set_xticks(x)
    ax.set_xticklabels(REP_PAIRS, rotation=20, ha="right", fontsize=10)
    _color_pair_ticks(ax, REP_PAIRS)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{name} @ {size_label} — per-run spread "
                 f"(5 dots per pair-stack; bar = median)",
                 pad=22)
    ax.legend(frameon=True, loc="best")
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    ax.margins(x=0.04)
    save(fig, fname)


def main():
    print("[distributions] B5 — bandwidth strip @ 1 MiB")
    strip_per_pair("bw", "bandwidth_MBps",
                   "bandwidth (GB/s)", divisor=1000.0,
                   fixed_size=1 << 20, size_label="1 MiB",
                   fname="B5_strip_bw_1MiB")
    print("[distributions] B6 — latency strip @ 8 B")
    strip_per_pair("latency", "latency_us",
                   "latency (us)", divisor=1.0,
                   fixed_size=8, size_label="8 B",
                   fname="B6_strip_latency_8B")


if __name__ == "__main__":
    main()
