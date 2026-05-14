"""Shared helpers for 4_osu/figures plotting scripts."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FIG_DIR = Path(__file__).resolve().parent
ROOT = FIG_DIR.parent
RESULTS = ROOT / "results"
PNG_DIR = FIG_DIR / "pngs"
TABLE_DIR = FIG_DIR / "tables"
PNG_DIR.mkdir(exist_ok=True)
TABLE_DIR.mkdir(exist_ok=True)

# --- topology ----------------------------------------------------------------
INTRA_PKG = {(0, 1), (2, 3), (4, 5), (6, 7)}
INTER_2LINK = {(0, 6), (2, 4)}
INTER_1LINK = {(0, 2), (1, 3), (1, 5), (3, 7), (4, 6), (5, 7)}


def topology(g0: int, g1: int) -> tuple[str, int]:
    a, b = sorted((g0, g1))
    if (a, b) in INTRA_PKG:
        return "intra_pkg", 4
    if (a, b) in INTER_2LINK:
        return "inter_pkg_2link", 2
    if (a, b) in INTER_1LINK:
        return "inter_pkg_1link", 1
    return "routed", 0


TIER_ORDER = ["intra_pkg", "inter_pkg_2link", "inter_pkg_1link", "routed"]
TIER_LABELS = {
    "intra_pkg": "intra-pkg (4 link)",
    "inter_pkg_2link": "inter-pkg (2 link)",
    "inter_pkg_1link": "inter-pkg (1 link)",
    "routed": "routed",
}
TIER_COLORS = {
    "intra_pkg": "#1f77b4",
    "inter_pkg_2link": "#2ca02c",
    "inter_pkg_1link": "#ff7f0e",
    "routed": "#d62728",
}
STACK_COLORS = {"eessi": "#1f77b4", "native": "#d62728"}
STACK_LABELS = {"eessi": "EESSI", "native": "Native (Cray)"}

# Canonical 12 measured pairs (matches topology.sh A1_PAIRS).
A1_PAIRS = [
    ("intra_pkg_OAM0",     0, 1, "intra_pkg",       4),
    ("intra_pkg_OAM1",     2, 3, "intra_pkg",       4),
    ("intra_pkg_OAM2",     4, 5, "intra_pkg",       4),
    ("inter_pkg_2link_06", 0, 6, "inter_pkg_2link", 2),
    ("inter_pkg_2link_24", 2, 4, "inter_pkg_2link", 2),
    ("inter_pkg_1link_02", 0, 2, "inter_pkg_1link", 1),
    ("inter_pkg_1link_15", 1, 5, "inter_pkg_1link", 1),
    ("inter_pkg_1link_37", 3, 7, "inter_pkg_1link", 1),
    ("inter_pkg_1link_57", 5, 7, "inter_pkg_1link", 1),
    ("routed_07",          0, 7, "routed",          0),
    ("routed_03",          0, 3, "routed",          0),
    ("routed_17",          1, 7, "routed",          0),
]
PAIR_ORDER = [p[0] for p in A1_PAIRS]
PAIR_TIER = {p[0]: p[3] for p in A1_PAIRS}
PAIR_GCDS = {p[0]: (p[1], p[2]) for p in A1_PAIRS}

# One representative pair per tier (matches A3_PAIRS in topology.sh).
# Used by per-pair display plots; tier aggregation still uses all 12.
REP_PAIRS = [
    "intra_pkg_OAM0",
    "inter_pkg_2link_06",
    "inter_pkg_1link_02",
    "routed_07",
]

# LUMI MI250X Infinity Fabric peak unidirectional bandwidth per tier (GB/s).
# Source: De Sensi et al. arXiv:2408.14090 Section II-C; 1 IF link = 400 Gb/s = 50 GB/s.
TIER_PEAK_GBPS = {
    "intra_pkg":       200.0,  # 4 IF links
    "inter_pkg_2link": 100.0,  # 2 IF links
    "inter_pkg_1link":  50.0,  # 1 IF link
    "routed":          None,   # path-dependent; intentionally omitted from plots
}


# --- loaders -----------------------------------------------------------------
def latest(pattern: str) -> Path:
    cands = sorted(RESULTS.glob(pattern))
    if not cands:
        raise FileNotFoundError(f"No match for {pattern} in {RESULTS}")
    return cands[-1]


def _load_two_stacks(filename_glob_for):
    frames = []
    for stack in ("eessi", "native"):
        f = latest(filename_glob_for(stack))
        df = pd.read_csv(f)
        df["stack"] = stack
        df["_source"] = f.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_pt2pt(name: str) -> pd.DataFrame:
    """name in {'bw', 'bibw', 'latency'}; concat eessi+native baseline runs."""
    return _load_two_stacks(lambda s: f"osu_{name}_{s}_*.csv")


def load_collectives(variant: str = "") -> pd.DataFrame:
    """variant '' = baseline; 'fixed_rndv' = tuned variant."""
    suffix = f"_{variant}" if variant else ""
    return _load_two_stacks(lambda s: f"osu_collectives{suffix}_{s}_*.csv")


def load_mbw_mr() -> pd.DataFrame:
    """mbw_mr CSV has unquoted commas in pairing_desc — custom parser."""
    cols = [
        "stack", "sdma_enabled", "config_label", "pairing_desc",
        "num_pairs", "run", "size_bytes", "bandwidth_MBps", "msg_rate_Mps",
    ]
    n_lead, n_tail = 3, 5
    frames = []
    for stack in ("eessi", "native"):
        f = latest(f"osu_mbw_mr_{stack}_*.csv")
        rows = []
        with open(f) as fh:
            next(fh)
            for line in fh:
                parts = line.rstrip("\n").split(",")
                lead = parts[:n_lead]
                tail = parts[-n_tail:]
                middle = ",".join(parts[n_lead:len(parts) - n_tail])
                rows.append(lead + [middle] + tail)
        df = pd.DataFrame(rows, columns=cols)
        for c in ("sdma_enabled", "num_pairs", "run", "size_bytes"):
            df[c] = df[c].astype(int)
        for c in ("bandwidth_MBps", "msg_rate_Mps"):
            df[c] = df[c].astype(float)
        df["stack"] = stack
        df["_source"] = f.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# --- aggregation -------------------------------------------------------------
def median_runs(df: pd.DataFrame, value_col: str, group_cols: list[str]) -> pd.DataFrame:
    """Median across `run` per group. CSVs already exclude warm-up (run starts at 2)."""
    return (
        df.groupby(group_cols, as_index=False)[value_col]
          .median()
    )


def stack_pivot(med: pd.DataFrame, index_cols: list[str], value_col: str) -> pd.DataFrame:
    """Pivot wide: stack as columns, returns df with eessi/native cols + ratio."""
    wide = med.pivot_table(index=index_cols, columns="stack", values=value_col).reset_index()
    if "eessi" in wide.columns and "native" in wide.columns:
        wide["ratio_eessi_native"] = wide["eessi"] / wide["native"]
    return wide


# --- plotting helpers --------------------------------------------------------
def save(fig, name: str) -> Path:
    out = PNG_DIR / f"{name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out.relative_to(ROOT)}")
    return out


def save_table(df: pd.DataFrame, name: str) -> Path:
    out = TABLE_DIR / f"{name}.csv"
    df.to_csv(out, index=False)
    print(f"  -> {out.relative_to(ROOT)}")
    return out


def fmt_size(b: int | float) -> str:
    units = ["B", "KiB", "MiB", "GiB"]
    v, i = float(b), 0
    while v >= 1024 and i < len(units) - 1:
        v /= 1024.0
        i += 1
    s = f"{v:.0f}" if v == int(v) else f"{v:.1f}"
    return f"{s} {units[i]}"


def setup_log_axes(ax, x: bool = True, y: bool = False) -> None:
    if x:
        ax.set_xscale("log", base=2)
    if y:
        ax.set_yscale("log")
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
