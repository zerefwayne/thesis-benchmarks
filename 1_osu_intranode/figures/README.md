# `4_osu/figures/`

Plotting code for the intranode EESSI-vs-Native chapter. Reads CSVs from
[../results/](../results/), writes PNGs to [pngs/](pngs/) and CSV summary
tables to [tables/](tables/).

Every plot puts both stacks **on the same axes** — grouped bars side by
side, lines overlaid in each panel, or paired strip plots. No heatmaps; no
pure-ratio panels.

## What's here

| Script | Figures | What it shows |
|---|---|---|
| [_common.py](_common.py)               | (library) | tier map, pair ordering, palettes, CSV loaders, save helpers |
| [reference.py](reference.py)           | H1, H2 | xGMI topology schematic; pair coverage matrix |
| [barcharts.py](barcharts.py)           | B1, B2, B4 | per-pair grouped bars (bw @ 1 MiB, lat @ 8 B) and tier-aggregated bars at 3 sizes |
| [per_pair_curves.py](per_pair_curves.py) | B3a, B3b, B3c | 4x3 small multiples — bw, bibw, latency curves with EESSI+Native overlaid |
| [per_pair_curves_fixed_rndv.py](per_pair_curves_fixed_rndv.py) | B3d | same 2x2 layout as B3a but uses the fixed_rndv osu_bw runs (EESSI UCX_RNDV_THRESH=1024) |
| [per_pair_curves_rma.py](per_pair_curves_rma.py) | B3e, B3f | same 2x2 layout for the MPI one-sided RMA primitives — osu_put_bw and osu_get_bw |
| [rndv_128_compare.py](rndv_128_compare.py) | B3g | single panel on intra_pkg_OAM0: baseline vs UCX_RNDV_THRESH=128 for both stacks |
| [protocol_sweep_eessi.py](protocol_sweep_eessi.py) | E1 | UCX_RNDV_THRESH sweep on intra_pkg_OAM0 — shows that thresholds above 1024 widen the eager-path cliff zone |
| [distributions.py](distributions.py)   | B5, B6 | per-run strip plots — 5 dots per pair-stack, median bar; "is the noise small?" |
| [delta_curves.py](delta_curves.py)     | D1 | signed % delta vs message size for bw, bibw, latency (positive = EESSI advantage) |
| [delta_bars.py](delta_bars.py)         | D2a-d | sorted % delta tornado bars across all 12 pairs at fixed sizes |
| [dumbbells.py](dumbbells.py)           | D3a-d | head-to-head dumbbell plots — Native and EESSI markers connected, Δ annotated |
| [efficiency_curves.py](efficiency_curves.py) | P1 | per-tier bandwidth curves with **theoretical IF peak** dashed line + small-msg latency inset (paper Fig 3 style) |
| [peak_efficiency_bars.py](peak_efficiency_bars.py) | P2 | per-pair peak bandwidth bars with per-pair nominal peak markers (paper Fig 4 style) |
| [portability_tax.py](portability_tax.py) | P3 | single-figure thesis bullet — sorted EESSI portability-tax bars across ~20 primitives |
| [collectives.py](collectives.py)       | F1, F2, F3 | per-collective curves (overlaid stacks); head-to-head bars at small/mid/large; latency vs N |
| [multipair.py](multipair.py)           | G1, G2, G3 | osu_mbw_mr aggregate bandwidth, message rate, peak-size bar chart |
| [tables.py](tables.py)                 | T1, T3 | tier-summary CSV; collective cross-over CSV |
| [optional.py](optional.py)             | I1, I2 | run-variability single-pair plot; latency vs bandwidth Pareto |

The full plan with figure descriptions lives at
`/users/joglekar/.claude/plans/okay-i-want-you-gleaming-key.md`.

## Running

These scripts need `pandas`, `numpy`, `matplotlib`. **Not available on the
LUMI login node** — use the local `plotenv/` venv (created already), or
activate one yourself, or load EESSI's SciPy-bundle.

```bash
cd thesis-benchmarks/4_osu/figures/
./run_all.sh
```

`run_all.sh` activates `plotenv/` if present, then runs each script in
reading order. Re-running a single script only regenerates its own outputs.

## Run order

There are no dependencies between scripts; any order works. The default
`run_all.sh` ordering matches a sensible thesis read:

1. `reference.py` — topology + coverage (sets up everything that follows)
2. `barcharts.py` — head-to-head bandwidth/latency at fixed sizes (the main story)
3. `per_pair_curves.py` — full size sweep, both stacks overlaid per pair
4. `distributions.py` — per-run spread (noise check)
5. `collectives.py` — collectives suite
6. `multipair.py` — multi-pair aggregate bandwidth
7. `tables.py` — CSV summary tables
8. `optional.py` — variability + Pareto

## Outputs

After a clean run:

```
pngs/
  H1_xgmi_topology.png
  H2_pair_coverage_matrix.png
  B1_perpair_bw_1MiB.png
  B2_perpair_latency_8B.png
  B3a_perpair_bw_curves.png
  B3b_perpair_bibw_curves.png
  B3c_perpair_latency_curves.png
  B3d_perpair_bw_curves_fixed_rndv.png
  B3e_perpair_put_bw_curves.png
  B3f_perpair_get_bw_curves.png
  B3g_rndv_128_vs_baseline_OAM0.png
  E1_protocol_sweep_eessi.png
  B4_tier_aggregated_bw.png
  B5_strip_bw_1MiB.png
  B6_strip_latency_8B.png
  D1_delta_pct_curves.png
  D2a_tornado_bw_1MiB.png
  D2b_tornado_bw_32MiB.png
  D2c_tornado_latency_8B.png
  D2d_tornado_latency_1KiB.png
  D3a_dumbbell_bw_1MiB.png
  D3b_dumbbell_bw_32MiB.png
  D3c_dumbbell_latency_8B.png
  D3d_dumbbell_latency_1KiB.png
  P1_efficiency_curves.png
  P2_peak_efficiency_bars.png
  P3_portability_tax.png
  F1_collectives_latency_curves.png
  F2_collectives_size_bars.png
  F3_collectives_scaling.png
  G1_mbw_mr_bandwidth_curves.png
  G2_mbw_mr_message_rate_curves.png
  G3_mbw_mr_peak_bars.png
  I1_variability_intra_pkg_OAM0.png
  I2_latency_bw_pareto.png
tables/
  T1_tier_summary_bw_peak.csv
  T2_portability_summary.csv
  T3_collective_crossover.csv
```

## What's intentionally not here

- Per-pair line plots for `osu_bw`, `osu_bibw`, `osu_latency` with one
  stack per panel (those live under
  [../../3_osu_eessi_vs_native/](../../3_osu_eessi_vs_native/)). Here the
  per-pair small multiples in [per_pair_curves.py](per_pair_curves.py)
  always overlay both stacks on the same axes.
- One-sided RMA (`osu_put_bw`, `osu_get_bw`) and host-buffer (`osu_bw_host`).
- Tuning impact (`fixed_rndv`, `ucc_shm`).
- Protocol-threshold sweeps (UCX_RNDV_THRESH, MPICH_GPU_IPC_THRESHOLD,
  NCCL/RCCL parameters).
- Heatmaps and pure-ratio panels — replaced by direct head-to-head primitives.

These are deferred per the plan in
`/users/joglekar/.claude/plans/okay-i-want-you-gleaming-key.md`.
