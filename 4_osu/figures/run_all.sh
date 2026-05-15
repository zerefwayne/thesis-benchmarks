#!/usr/bin/env bash
# Generate every figure and CSV table in 4_osu/figures/.
# Activates ./plotenv if present; otherwise assumes python+pandas+matplotlib are on PATH.

set -euo pipefail

cd "$(dirname "$0")"

if [[ -f plotenv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source plotenv/bin/activate
fi

SCRIPTS=(
    reference.py        # H1 topology, H2 coverage
    barcharts.py        # B1, B2, B4 (per-pair + tier-aggregated bars)
    per_pair_curves.py  # B3 (small multiples: bw, bibw, latency)
    per_pair_curves_fixed_rndv.py  # B3d (bw small multiples, fixed_rndv)
    per_pair_curves_rma.py  # B3e/B3f (osu_put_bw, osu_get_bw small multiples)
    rndv_128_compare.py     # B3g (baseline vs RNDV_THRESH=128 on intra_pkg_OAM0)
    protocol_sweep_eessi.py # E1 (UCX_RNDV_THRESH sweep)
    distributions.py    # B5, B6 (per-run strip plots)
    delta_curves.py     # D1 (signed % delta vs message size)
    delta_bars.py       # D2 (sorted % delta tornado bars)
    dumbbells.py        # D3 (head-to-head slope/dumbbell plots)
    efficiency_curves.py    # P1 (per-tier bw curves vs theoretical IF peak)
    peak_efficiency_bars.py # P2 (per-pair bw bars + nominal peak references)
    portability_tax.py      # P3 (single-figure portability-tax bar chart)
    collectives.py      # F1, F2, F3
    multipair.py        # G1, G2, G3
    tables.py           # T1, T3
    optional.py         # I1, I2
)

for s in "${SCRIPTS[@]}"; do
    echo "===== $s ====="
    python "$s"
done

echo
echo "Done. Outputs:"
echo "  PNGs:   $(ls pngs/ | wc -l) files in pngs/"
echo "  Tables: $(ls tables/ | wc -l) files in tables/"
