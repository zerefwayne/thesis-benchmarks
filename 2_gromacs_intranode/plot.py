#!/usr/bin/env python3
"""Plot EESSI vs native GROMACS performance from results/perf.csv.

Two PNGs in this directory:
  perf_bars.png   - grouped bars, one group per benchmark, two bars per stack,
                    error bars over the per-run sample sd.
  perf_pairs.png  - paired dot plot, each benchmark's eessi/native means
                    connected by a line for easy visual delta-reading.

Per the plotting preferences in CLAUDE.md / feedback memory: no heatmaps,
no pure-ratio panels. Show the actual numbers.
"""
from __future__ import annotations

import csv
import statistics as stats
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
PERF = HERE / "results" / "perf.csv"

STACKS = ("eessi", "native")
BENCH_ORDER = ("crambin", "hEGFRDimer", "stmv", "hEGFRDimerPair")
STACK_COLORS = {"eessi": "#1f77b4", "native": "#d62728"}


def load() -> dict[tuple[str, str], list[float]]:
    by: dict[tuple[str, str], list[float]] = {}
    with PERF.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                v = float(r["perf_ns_per_day"])
            except (TypeError, ValueError):
                continue
            by.setdefault((r["benchmark"], r["stack"]), []).append(v)
    return by


def order_benchmarks(by: dict[tuple[str, str], list[float]]) -> list[str]:
    present = {b for (b, _s) in by}
    ordered = [b for b in BENCH_ORDER if b in present]
    extras = sorted(present - set(ordered))
    return ordered + extras


def grouped_bars(by, benches, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(benches)), 4.5))
    x = list(range(len(benches)))
    width = 0.36

    for i, stack in enumerate(STACKS):
        means, sds = [], []
        for b in benches:
            vs = by.get((b, stack), [])
            means.append(stats.fmean(vs) if vs else 0.0)
            sds.append(stats.stdev(vs) if len(vs) > 1 else 0.0)
        offset = (i - 0.5) * width
        bars = ax.bar(
            [xi + offset for xi in x], means, width,
            yerr=sds, capsize=4, label=stack,
            color=STACK_COLORS[stack], edgecolor="black", linewidth=0.5,
        )
        for xi, m in zip(x, means):
            if m > 0:
                ax.text(xi + offset, m, f" {m:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(benches, rotation=0)
    ax.set_ylabel("Performance (ns/day)")
    ax.set_title("GROMACS 2025.1  —  EESSI (rfoss-SYCL) vs Native (cpeAMD-VkFFT-rocm)\nLUMI-G, 1 node, 8 GCDs")
    ax.legend(title="stack", loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def paired_dots(by, benches, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(5, 1.2 * len(benches)), 4.5))

    for xi, b in enumerate(benches):
        means = {}
        for stack in STACKS:
            vs = by.get((b, stack), [])
            if vs:
                means[stack] = stats.fmean(vs)
                ax.scatter([xi], [means[stack]],
                           color=STACK_COLORS[stack], s=80, zorder=3,
                           label=stack if xi == 0 else None)
                for v in vs:
                    ax.scatter([xi], [v], color=STACK_COLORS[stack],
                               alpha=0.35, s=24, zorder=2)
        if "eessi" in means and "native" in means:
            ax.plot([xi, xi], [means["eessi"], means["native"]],
                    color="gray", linewidth=1, zorder=1)
            delta_pct = 100.0 * (means["native"] - means["eessi"]) / means["eessi"]
            y_text = max(means.values()) * 1.02
            ax.text(xi, y_text, f"{delta_pct:+.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(list(range(len(benches))))
    ax.set_xticklabels(benches)
    ax.set_ylabel("Performance (ns/day)")
    ax.set_title("Paired dot plot — native delta vs EESSI (single 1-node MI250X allocation)")
    ax.legend(title="stack", loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def main() -> int:
    if not PERF.exists():
        print(f"ERROR: {PERF} does not exist — run parse_results.py first", file=sys.stderr)
        return 1
    by = load()
    if not by:
        print("ERROR: no valid rows in perf.csv", file=sys.stderr)
        return 1
    benches = order_benchmarks(by)
    grouped_bars(by, benches, HERE / "perf_bars.png")
    paired_dots(by, benches, HERE / "perf_pairs.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
