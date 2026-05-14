"""H1, H2 — reference figures.

H1: xGMI topology schematic (8 GCDs in 4 OAM packages, links by tier).
H2: 8x8 pair-coverage matrix showing which pairs are measured + tier color.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import ListedColormap, BoundaryNorm

from _common import (
    A1_PAIRS, INTRA_PKG, INTER_2LINK, INTER_1LINK,
    TIER_COLORS, TIER_LABELS, TIER_ORDER,
    save, topology,
)


# Layout positions for the 8 GCDs (paired by OAM package).
GCD_POS = {
    0: (0.0, 1.0), 1: (1.0, 1.0),   # OAM0 (top-left)
    2: (3.0, 1.0), 3: (4.0, 1.0),   # OAM1 (top-right)
    4: (0.0, 0.0), 5: (1.0, 0.0),   # OAM2 (bottom-left)
    6: (3.0, 0.0), 7: (4.0, 0.0),   # OAM3 (bottom-right)
}
PKG_GROUPS = {0: [0, 1], 1: [2, 3], 2: [4, 5], 3: [6, 7]}


def h1_topology_schematic():
    fig, ax = plt.subplots(figsize=(11, 6.5))

    # Draw OAM package boxes
    for pkg_idx, gcds in PKG_GROUPS.items():
        xs = [GCD_POS[g][0] for g in gcds]
        ys = [GCD_POS[g][1] for g in gcds]
        x0, x1 = min(xs) - 0.45, max(xs) + 0.45
        y0, y1 = min(ys) - 0.4, max(ys) + 0.4
        box = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                             boxstyle="round,pad=0.05",
                             linewidth=1.2, edgecolor="gray",
                             facecolor="#f5f5f5", zorder=1)
        ax.add_patch(box)
        ax.text((x0 + x1) / 2, y1 - 0.05, f"OAM{pkg_idx}",
                ha="center", va="bottom", fontsize=10, color="gray",
                style="italic")

    # Draw all xGMI links (intra_pkg, inter_2link, inter_1link)
    drawn = set()
    for tier_set, tier in [(INTRA_PKG, "intra_pkg"),
                           (INTER_2LINK, "inter_pkg_2link"),
                           (INTER_1LINK, "inter_pkg_1link")]:
        for (a, b) in tier_set:
            if (a, b) in drawn:
                continue
            drawn.add((a, b))
            xa, ya = GCD_POS[a]
            xb, yb = GCD_POS[b]
            num_links = {"intra_pkg": 4, "inter_pkg_2link": 2,
                         "inter_pkg_1link": 1}[tier]
            ax.plot([xa, xb], [ya, yb],
                    color=TIER_COLORS[tier], linewidth=num_links * 1.2,
                    alpha=0.75, zorder=2,
                    solid_capstyle="round")

    # Draw GCD nodes on top
    for g, (x, y) in GCD_POS.items():
        ax.scatter([x], [y], s=900, color="#222", zorder=3, edgecolor="white",
                   linewidth=2)
        ax.text(x, y, f"GCD\n{g}", ha="center", va="center",
                color="white", fontsize=9, fontweight="bold", zorder=4)

    # Legend
    handles = []
    for tier in ("intra_pkg", "inter_pkg_2link", "inter_pkg_1link"):
        nl = {"intra_pkg": 4, "inter_pkg_2link": 2, "inter_pkg_1link": 1}[tier]
        handles.append(plt.Line2D([], [], color=TIER_COLORS[tier],
                                  linewidth=nl * 1.2,
                                  label=f"{TIER_LABELS[tier]}"))
    handles.append(plt.Line2D([], [], color="lightgray", linewidth=1,
                              linestyle=":",
                              label=f"{TIER_LABELS['routed']} (no direct link)"))
    ax.legend(handles=handles, loc="lower center",
              bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False, fontsize=10)

    ax.set_xlim(-0.9, 4.9)
    ax.set_ylim(-0.7, 1.7)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("LUMI MI250X intranode xGMI topology — 4 OAM packages, 8 GCDs",
                 fontsize=13)
    save(fig, "H1_xgmi_topology")


def h2_pair_coverage():
    """8x8 grid: tier color + dot in cells that are in the measured A1 set."""
    measured = {tuple(sorted((a, b))) for _, a, b, _, _ in A1_PAIRS}

    # Build tier-id matrix (0..3 + nan for self)
    tier_to_id = {t: i for i, t in enumerate(TIER_ORDER)}
    M = np.full((8, 8), np.nan)
    for i in range(8):
        for j in range(8):
            if i == j:
                continue
            tier, _ = topology(i, j)
            M[i, j] = tier_to_id[tier]

    cmap = ListedColormap([TIER_COLORS[t] for t in TIER_ORDER])
    bounds = np.arange(-0.5, len(TIER_ORDER) + 0.5)
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(7.8, 6.5))
    im = ax.imshow(M, cmap=cmap, norm=norm)
    # Mark measured cells with bold dot + label
    for i in range(8):
        for j in range(8):
            if (min(i, j), max(i, j)) in measured and i != j:
                ax.scatter([j], [i], s=80, marker="o",
                           edgecolor="white", linewidth=1.4,
                           facecolor="black", zorder=3)
            elif i == j:
                ax.scatter([j], [i], s=180, marker="x",
                           color="lightgray", linewidth=1.5)

    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    ax.set_xticklabels([f"GCD{i}" for i in range(8)], rotation=45, ha="right")
    ax.set_yticklabels([f"GCD{i}" for i in range(8)])
    ax.set_xlim(-0.5, 7.5)
    ax.set_ylim(7.5, -0.5)
    ax.set_title("Pair coverage matrix (12 measured pairs marked with bullets)")

    handles = [plt.Line2D([], [], marker="s", color="w",
                          markerfacecolor=TIER_COLORS[t], markersize=12,
                          label=TIER_LABELS[t]) for t in TIER_ORDER]
    handles.append(plt.Line2D([], [], marker="o", color="w",
                              markerfacecolor="black", markeredgecolor="white",
                              markersize=10, label="measured pair"))
    handles.append(plt.Line2D([], [], marker="x", color="lightgray",
                              markersize=10, linestyle="",
                              label="self (not measured)"))
    ax.legend(handles=handles, loc="center left",
              bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=9)
    save(fig, "H2_pair_coverage_matrix")


def main():
    print("[reference] H1 — xGMI topology schematic")
    h1_topology_schematic()
    print("[reference] H2 — pair coverage matrix")
    h2_pair_coverage()


if __name__ == "__main__":
    main()
