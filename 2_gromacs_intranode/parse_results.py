#!/usr/bin/env python3
"""Aggregate every per-job CSV in results/ into a single perf.csv.

Each job writes one CSV with columns:
  benchmark,stack,jobid,run,perf_ns_per_day,wall_s,core_s,ntmpi,toolchain

This script concatenates them, drops rows with perf=NA, and prints a tidy
summary (mean / median / sd / CV per benchmark x stack). Output:
  results/perf.csv  (all per-run rows)
  stdout            (summary table)

Kept compatible with the stock LUMI login-node Python 3.6 — no f-string
type syntax, no statistics.fmean, no PEP 604/585 annotations.
"""

import csv
import math
import statistics as stats
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT = RESULTS / "perf.csv"

FIELDS = [
    "benchmark", "stack", "jobid", "run",
    "perf_ns_per_day", "wall_s", "core_s", "ntmpi", "toolchain",
]


def load_rows():
    rows = []
    for csv_path in sorted(RESULTS.glob("*_*_*.csv")):
        if csv_path.name == OUT.name:
            continue
        with csv_path.open() as f:
            for r in csv.DictReader(f):
                rows.append(r)
    return rows


def to_float(s):
    if s is None or s == "" or s == "NA":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def summarize(rows):
    by_bs = {}
    for r in rows:
        perf = to_float(r["perf_ns_per_day"])
        if perf is None:
            continue
        by_bs.setdefault((r["benchmark"], r["stack"]), []).append(perf)

    if not by_bs:
        print("(no valid perf rows yet)", file=sys.stderr)
        return

    header = "{:<18}{:<10}{:>4}  {:>13}  {:>13}  {:>9}  {:>7}  {:>9}  {:>9}".format(
        "benchmark", "stack", "n", "mean (ns/day)", "median", "sd", "CV %", "min", "max")
    print(header)
    print("-" * len(header))
    for key in sorted(by_bs):
        bench, stack = key
        vs = by_bs[key]
        n = len(vs)
        mean = stats.mean(vs)
        median = stats.median(vs)
        sd = stats.stdev(vs) if n > 1 else math.nan
        cv = 100.0 * sd / mean if (n > 1 and mean) else math.nan
        print("{:<18}{:<10}{:>4}  {:>13.3f}  {:>13.3f}  {:>9.3f}  {:>7.2f}  {:>9.3f}  {:>9.3f}".format(
            bench, stack, n, mean, median, sd, cv, min(vs), max(vs)))


def main():
    rows = load_rows()
    if not rows:
        print("No per-job CSVs found in {}/".format(RESULTS), file=sys.stderr)
        return 1

    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})

    print("wrote {}  ({} rows)".format(OUT, len(rows)))
    print()
    summarize(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
