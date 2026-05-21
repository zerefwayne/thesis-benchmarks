#!/bin/bash
# Reconstruct GROMACS_Benchmark_Suite/ from upstream sources. The .tpr inputs
# are large binary blobs (STMV ~134 MB, hEGFRtetramerPair ~74 MB) that blow
# past GitHub's file-size limits, so they are NOT committed to this repo —
# run this script once on a login node to rebuild the folder.
#
# Two steps:
#   1. HECBioSim + MPSBench suite  <- git clone eth-cscs/GROMACS_Benchmark_Suite
#      (Crambin, Glutamine-Binding-Protein, hEGFRDimer[Pair|SmallerPL],
#       hEGFRtetramerPair, MPSBench/RNAseCubic)
#   2. STMV (~1M atoms)            -> GROMACS_Benchmark_Suite/STMV/benchmark.tpr
#
# We prefer the AMD InfinityHub-CI STMV tarball — that is the exact input
# behind the AMD ROCm Blog LUMI guide's Tables 5-8 (the headline reference
# numbers we want to compare against). If GitHub is unreachable we fall
# back to Zenodo 3893789 (Kutzner et al. JCP 2019 benchmark archive),
# which contains a stmv/topol.tpr.
#
# Run this on a login node — compute nodes don't have outbound HTTPS.
# Idempotent: re-running with the .tpr already in place is a no-op.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

# ----------------------------------------------------------------------
# Step 1 — HECBioSim + MPSBench suite (eth-cscs/GROMACS_Benchmark_Suite).
# Plain (non-LFS) git blobs; largest is hEGFRtetramerPair at ~74 MB. We
# clone to a temp dir and copy the trees in, stripping the embedded .git
# so GROMACS_Benchmark_Suite never becomes a nested repo inside
# thesis-benchmarks. Idempotent: skipped if hEGFRDimer is already present.
# ----------------------------------------------------------------------
SUITE_DIR="$HERE/GROMACS_Benchmark_Suite"
SUITE_URL="https://github.com/eth-cscs/GROMACS_Benchmark_Suite.git"
if [[ -f "$SUITE_DIR/HECBioSim/hEGFRDimer/benchmark.tpr" ]]; then
    echo "HECBioSim/MPSBench suite already present in $SUITE_DIR — skipping clone."
else
    echo "Cloning $SUITE_URL ..."
    SUITE_TMP="$(mktemp -d -t gbs-fetch.XXXXXX)"
    if git clone --depth 1 "$SUITE_URL" "$SUITE_TMP/repo"; then
        mkdir -p "$SUITE_DIR"
        for sub in HECBioSim MPSBench; do
            if [[ -d "$SUITE_TMP/repo/$sub" ]]; then
                cp -a "$SUITE_TMP/repo/$sub" "$SUITE_DIR/"
                echo "  installed $sub/"
            fi
        done
        [[ -f "$SUITE_TMP/repo/README.md" ]] && cp -a "$SUITE_TMP/repo/README.md" "$SUITE_DIR/"
        rm -rf "$SUITE_TMP"
        echo "Suite ready: $SUITE_DIR (HECBioSim + MPSBench)"
    else
        rm -rf "$SUITE_TMP"
        echo "ERROR: failed to clone $SUITE_URL" >&2
        echo "Manual fallback: git clone $SUITE_URL, then copy its HECBioSim/" >&2
        echo "and MPSBench/ trees into $SUITE_DIR/ (drop the .git directory)." >&2
        exit 1
    fi
fi

# ----------------------------------------------------------------------
# Step 2 — STMV. Not in the eth-cscs repo; fetched from AMD / Zenodo below.
# ----------------------------------------------------------------------
TARGET_DIR="$SUITE_DIR/STMV"
TARGET_TPR="$TARGET_DIR/benchmark.tpr"

if [[ -f "$TARGET_TPR" ]]; then
    echo "STMV TPR already present at $TARGET_TPR — skipping download."
    exit 0
fi

mkdir -p "$TARGET_DIR"
TMPDIR="$(mktemp -d -t stmv-fetch.XXXXXX)"
trap 'rm -rf "$TMPDIR"' EXIT

AMD_URL="https://raw.githubusercontent.com/amd/InfinityHub-CI/main/gromacs/docker/benchmark/stmv/stmv.tar.gz"
ZENODO_URL="https://zenodo.org/record/3893789/files/GROMACS_heterogeneous_parallelization_benchmark_info_and_systems_JCP.tar.gz"

# ----------------------------------------------------------------------
# Source A — AMD InfinityHub-CI (preferred; matches AMD ROCm Blog numbers)
# ----------------------------------------------------------------------
echo "Trying AMD InfinityHub-CI STMV tarball..."
echo "  $AMD_URL"
if curl -L --fail --retry 3 --connect-timeout 30 \
       -o "$TMPDIR/amd_stmv.tar.gz" "$AMD_URL"; then
    echo "Extracting AMD STMV..."
    tar -xzf "$TMPDIR/amd_stmv.tar.gz" -C "$TMPDIR"
    AMD_TPR=$(find "$TMPDIR" -type f -name 'topol.tpr' | head -1)
    if [[ -n "$AMD_TPR" ]]; then
        cp -v "$AMD_TPR" "$TARGET_TPR"
        echo "STMV ready (source: AMD InfinityHub-CI): $TARGET_TPR"
        echo "Note: this is the exact input behind the AMD ROCm Blog LUMI" \
             "guide Tables 5-8 — direct ns/day comparator."
        exit 0
    fi
    echo "WARN: AMD tarball downloaded but no topol.tpr inside; falling back." >&2
else
    echo "WARN: AMD InfinityHub-CI download failed; falling back to Zenodo." >&2
fi

# ----------------------------------------------------------------------
# Source B — Zenodo 3893789 (Kutzner et al. JCP 2019)
# ----------------------------------------------------------------------
echo
echo "Trying Zenodo 3893789..."
echo "  $ZENODO_URL"
if ! curl -L --fail --retry 3 --connect-timeout 30 \
         -o "$TMPDIR/zenodo.tar.gz" "$ZENODO_URL"; then
    cat >&2 <<EOF
ERROR: both sources failed.

Manual fallback:
  1. Obtain an STMV .tpr from one of:
       - https://github.com/amd/InfinityHub-CI/tree/main/gromacs/docker/benchmark/stmv
       - https://zenodo.org/record/3893789  (path: .../stmv/topol.tpr)
       - https://www.bioexcel.eu/benchmark-suite/
  2. Place it at: $TARGET_TPR
EOF
    exit 1
fi

echo "Extracting Zenodo archive..."
tar -xzf "$TMPDIR/zenodo.tar.gz" -C "$TMPDIR"

# The Zenodo layout is .../stmv/topol.tpr — match by parent dir name,
# not by filename (the file is "topol.tpr", not "*stmv*.tpr").
ZENODO_TPR=$(find "$TMPDIR" -type f -name 'topol.tpr' -path '*/stmv/*' | head -1)
if [[ -z "$ZENODO_TPR" ]]; then
    echo "ERROR: no stmv/topol.tpr found inside Zenodo archive." >&2
    echo "Tempdir preserved for inspection: $TMPDIR" >&2
    trap - EXIT
    exit 1
fi

cp -v "$ZENODO_TPR" "$TARGET_TPR"
echo "STMV ready (source: Zenodo 3893789): $TARGET_TPR"
echo "Note: this differs slightly from the AMD InfinityHub-CI STMV used" \
     "in the AMD ROCm Blog Tables 5-8. ns/day numbers are still comparable" \
     "in order of magnitude but not exactly."

# Optional sanity check.
if command -v gmx_mpi >/dev/null 2>&1; then
    echo "--- gmx_mpi check ---"
    gmx_mpi check -s "$TARGET_TPR" 2>&1 | grep -E '(atoms|topology|step)' | head -5 || true
fi
