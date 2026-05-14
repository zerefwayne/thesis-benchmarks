"""F1, F2, F3 — collectives plots.

F1: per-collective latency curves (4 panels x 6 lines: stack x N).
F2: per-collective head-to-head bars at small/mid/large message sizes (N=8).
F3: latency vs N at fixed mid-size, one subplot per collective.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from _common import (
    STACK_COLORS, STACK_LABELS,
    fmt_size, load_collectives, median_runs, save, setup_log_axes,
)


COLLECTIVES = ["osu_allreduce", "osu_alltoall", "osu_bcast", "osu_allgather"]
N_GCDS = [2, 4, 8]
NSTYLE = {2: "-", 4: "--", 8: ":"}
MID_SIZE = 65536  # 64 KiB for F3
F2_SIZES = [4, 1024, 65536, 1 << 20]  # 4 B, 1 KiB, 64 KiB, 1 MiB
F2_FIXED_N = 8


# ============================================================================
# F1 — per-collective latency curves
# ============================================================================
def f1_per_collective():
    df = load_collectives()
    med = median_runs(df, "latency_us",
                      ["stack", "benchmark", "num_gcds", "size_bytes"])

    fig, axes = plt.subplots(2, 2, figsize=(13, 9),
                             gridspec_kw={"hspace": 0.32, "wspace": 0.22})
    for ax, coll in zip(axes.flat, COLLECTIVES):
        sub = med[med["benchmark"] == coll]
        if sub.empty:
            ax.set_title(f"{coll} — no data")
            continue
        for stack in ("eessi", "native"):
            for n in N_GCDS:
                s = sub[(sub["stack"] == stack) & (sub["num_gcds"] == n)]
                s = s.sort_values("size_bytes")
                if s.empty:
                    continue
                ax.plot(s["size_bytes"], s["latency_us"],
                        color=STACK_COLORS[stack],
                        linestyle=NSTYLE[n],
                        marker="o", markersize=2.5, linewidth=1.4,
                        label=f"{STACK_LABELS[stack]}, N={n}")
        setup_log_axes(ax, x=True, y=True)
        ax.set_title(coll)
        ax.set_xlabel("message size (bytes)")
        ax.set_ylabel("latency (us)")
        if ax is axes.flat[0]:
            ax.legend(fontsize=8, loc="upper left", ncol=2, frameon=True)
    fig.suptitle("Collectives latency — EESSI vs Native, N in {2, 4, 8}",
                 fontsize=14, y=0.995)
    save(fig, "F1_collectives_latency_curves")


# ============================================================================
# F2 — per-collective head-to-head bars at small/mid/large message sizes
# ============================================================================
def f2_size_bars():
    df = load_collectives()
    sub = df[(df["num_gcds"] == F2_FIXED_N)
             & (df["size_bytes"].isin(F2_SIZES))].copy()
    if sub.empty:
        print(f"  !! no collectives data at N={F2_FIXED_N}; skipping F2")
        return
    agg = (sub.groupby(["stack", "benchmark", "size_bytes"], as_index=False)
              .agg(median=("latency_us", "median"),
                   lo=("latency_us", "min"),
                   hi=("latency_us", "max")))

    fig, axes = plt.subplots(1, len(COLLECTIVES),
                             figsize=(4 * len(COLLECTIVES), 4.6),
                             sharey=False)
    width = 0.38
    x = np.arange(len(F2_SIZES))
    for ax, coll in zip(axes, COLLECTIVES):
        for i, stack in enumerate(("eessi", "native")):
            s = (agg[(agg["stack"] == stack) & (agg["benchmark"] == coll)]
                     .set_index("size_bytes")
                     .reindex(F2_SIZES))
            offset = (i - 0.5) * width
            med = s["median"].to_numpy()
            lo_err = (s["median"] - s["lo"]).to_numpy()
            hi_err = (s["hi"] - s["median"]).to_numpy()
            ax.bar(x + offset, med, width=width,
                   color=STACK_COLORS[stack],
                   label=STACK_LABELS[stack] if ax is axes[0] else None,
                   edgecolor="black", linewidth=0.4,
                   yerr=[lo_err, hi_err], capsize=3,
                   error_kw={"linewidth": 0.8, "ecolor": "black"})
        ax.set_xticks(x)
        ax.set_xticklabels([fmt_size(sz) for sz in F2_SIZES],
                           rotation=20, ha="right", fontsize=9)
        ax.set_yscale("log")
        ax.set_title(coll.replace("osu_", ""))
        if ax is axes[0]:
            ax.set_ylabel("latency (us, log)")
            ax.legend(frameon=True, fontsize=9, loc="best")
        ax.grid(True, axis="y", which="both", linestyle=":", alpha=0.4)

    fig.suptitle(f"Collectives @ N={F2_FIXED_N} — EESSI vs Native at small/mid/large messages "
                 "(error bars = min/max over 5 runs)",
                 y=1.04, fontsize=13)
    save(fig, "F2_collectives_size_bars")


# ============================================================================
# F3 — strong-scaling-style: latency at fixed size vs N
# ============================================================================
def f3_strong_scaling():
    df = load_collectives()
    med = median_runs(df, "latency_us",
                      ["stack", "benchmark", "num_gcds", "size_bytes"])
    sub = med[med["size_bytes"] == MID_SIZE]

    fig, axes = plt.subplots(1, len(COLLECTIVES),
                             figsize=(4 * len(COLLECTIVES), 4.2),
                             sharey=False)
    for ax, coll in zip(axes, COLLECTIVES):
        s = sub[sub["benchmark"] == coll]
        if s.empty:
            ax.set_title(f"{coll} — no data")
            continue
        for stack in ("eessi", "native"):
            ss = s[s["stack"] == stack].sort_values("num_gcds")
            ax.plot(ss["num_gcds"], ss["latency_us"],
                    marker="o", markersize=6, linewidth=1.6,
                    color=STACK_COLORS[stack],
                    label=STACK_LABELS[stack])
        ax.set_xticks(N_GCDS)
        ax.set_title(coll)
        ax.set_xlabel("num GCDs")
        ax.set_ylabel(f"latency (us) @ {fmt_size(MID_SIZE)}")
        ax.grid(True, linestyle=":", alpha=0.4)
        if ax is axes[0]:
            ax.legend(fontsize=9, frameon=True)
    fig.suptitle(f"Collectives — latency at {fmt_size(MID_SIZE)} vs participating GCDs",
                 fontsize=13, y=1.04)
    save(fig, "F3_collectives_scaling")


def main():
    print("[collectives] F1 — per-collective latency curves")
    f1_per_collective()
    print("[collectives] F2 — head-to-head bars at small/mid/large sizes")
    f2_size_bars()
    print("[collectives] F3 — latency vs N @ 64 KiB")
    f3_strong_scaling()


if __name__ == "__main__":
    main()
