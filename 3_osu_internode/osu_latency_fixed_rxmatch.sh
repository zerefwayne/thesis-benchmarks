#!/bin/bash
#SBATCH --job-name=osu_latency_fixed_rxmatch
#SBATCH --account=project_462000226
#SBATCH --partition=dev-g
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --cpus-per-task=7
#SBATCH --time=00:30:00
#SBATCH --output=results/%x_%j.out
#SBATCH --constraint=eessi
#
# osu_latency_fixed_rxmatch.sh — inter-node ping-pong latency with CXI hardware-
# matched receives (instead of common.sh's default hybrid mode).
#
# Companion to osu_latency.sh. Only difference: EESSI arm overrides
# FI_CXI_RX_MATCH_MODE=hardware after setup_eessi. Native arm runs unchanged.
#
# Why: osu_latency_eessi_18750320 showed an 18.7 μs floor for messages 1–128 B,
# vs ~2.5 μs on native — a 7.5× small-msg latency penalty. The protocol cliff
# at 128→256 B (19.2 μs → 3.5 μs) suggests EESSI's small-msg path is going
# through CXI's *software* match list (the hybrid default falls back to SW
# matching when HW resources are tight). Forcing match_mode=hardware should
# either:
#   (a) drop the floor to ~3–4 μs (matching native) — confirms SW-matching was
#       the culprit; we then keep `hardware` as the recommended setting; or
#   (b) leave the floor unchanged — rules out SW matching and the penalty is
#       elsewhere (OFI MTL `cm` PML, OpenMPI matching layer, etc.).
#
# Schema mirrors osu_latency.sh exactly so CSVs concat / diff against baseline.
#
# Submit:
#   sbatch osu_latency_fixed_rxmatch.sh eessi
#   sbatch osu_latency_fixed_rxmatch.sh native    # baseline re-take, env irrelevant

STACK="${1:?usage: sbatch $0 <eessi|native>}"
[[ "$STACK" == "eessi" || "$STACK" == "native" ]] \
    || { echo "ERROR: stack must be 'eessi' or 'native', got '$STACK'" >&2; exit 1; }

source common.sh
source topology.sh

FILE_BASE="osu_latency_fixed_rxmatch_${STACK}"
CSV_FILE="${RESULT_DIR}/${FILE_BASE}_${SLURM_JOB_ID}.csv"
LOG_FILE="${RESULT_DIR}/${FILE_BASE}_${SLURM_JOB_ID}.log"
META_FILE="${RESULT_DIR}/${FILE_BASE}_${SLURM_JOB_ID}.meta"

node_metadata_dump "$META_FILE"

NODES=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
[[ ${#NODES[@]} -ge 2 ]] || { echo "ERROR: need 2 nodes" | tee -a "$LOG_FILE"; exit 1; }
NODE_A=${NODES[0]}; NODE_B=${NODES[1]}
echo "STACK=$STACK  NODE_A=$NODE_A  NODE_B=$NODE_B" | tee -a "$LOG_FILE"

echo "stack,sdma_enabled,pair_label,node_a,node_b,gcd_a,gcd_b,hop_class,nic_class,run,size_bytes,latency_us" \
    > "$CSV_FILE"

NUM_RUNS=6; WARMUP_RUN=1; RUN_TIMEOUT=300
OSU_FLAGS="-i 100 -d rocm D D"   # OSU default 1 MiB cap

WRAPPER="${RESULT_DIR}/wrap_pt2pt_${SLURM_JOB_ID}.sh"
cat > "$WRAPPER" <<'EOF'
#!/bin/bash
GCD_A=$1; GCD_B=$2; shift 2
rank=${SLURM_PROCID:-${OMPI_COMM_WORLD_RANK:-${PMIX_RANK:-0}}}
case "$rank" in
    0) export ROCR_VISIBLE_DEVICES=$GCD_A ;;
    *) export ROCR_VISIBLE_DEVICES=$GCD_B ;;
esac
exec "$@"
EOF
chmod +x "$WRAPPER"

parse_lat() {
    local stack=$1 sdma=$2 lab=$3 na=$4 nb=$5 g0=$6 g1=$7 nic=$8 run=$9
    awk -v st="$stack" -v sdma="$sdma" -v lab="$lab" -v na="$na" -v nb="$nb" \
        -v g0="$g0" -v g1="$g1" -v nic="$nic" -v run="$run" '
        /^#/ { next }
        NF == 2 && $1 ~ /^[0-9]+$/ {
            printf "%s,%d,%s,%s,%s,%d,%d,inter_node,%s,%d,%d,%.2f\n",
                   st, sdma, lab, na, nb, g0, g1, nic, run, $1, $2
        }'
}

run_pair() {
    local sdma=$1 g0=$2 g1=$3 lab=$4 nic=$5 bin=$6
    {
        echo
        echo "################################################################"
        echo "# [$STACK sdma=$sdma] $lab  GCD($g0,$g1)  nic=$nic"
        echo "#   FI_CXI_RX_MATCH_MODE=${FI_CXI_RX_MATCH_MODE:-unset}"
        echo "################################################################"
    } | tee -a "$LOG_FILE"

    for run in $(seq 1 $NUM_RUNS); do
        local t0=$SECONDS tag raw
        tag=$( (( run == WARMUP_RUN )) && echo "warm-up" || echo "recording" )
        echo "  [run $run/$NUM_RUNS] $tag - $(date '+%H:%M:%S')" | tee -a "$LOG_FILE"

        if [[ "$STACK" == "native" ]]; then
            raw=$(timeout $RUN_TIMEOUT srun --nodes=2 --ntasks=2 --ntasks-per-node=1 \
                "$WRAPPER" "$g0" "$g1" "$bin" $OSU_FLAGS 2>>"$LOG_FILE")
        else
            raw=$(timeout $RUN_TIMEOUT mpirun -n 2 --host "$NODE_A:1,$NODE_B:1" \
                --map-by ppr:1:node \
                "$WRAPPER" "$g0" "$g1" "$bin" $OSU_FLAGS 2>>"$LOG_FILE")
        fi
        echo "  [run $run/$NUM_RUNS] done in $((SECONDS - t0))s" | tee -a "$LOG_FILE"
        echo "$raw" >> "$LOG_FILE"
        if (( run != WARMUP_RUN )); then
            echo "$raw" | parse_lat "$STACK" "$sdma" "$lab" "$NODE_A" "$NODE_B" \
                "$g0" "$g1" "$nic" "$run" >> "$CSV_FILE"
        fi
    done
}

# ============================================================================
export HSA_ENABLE_SDMA=0
if [[ "$STACK" == "native" ]]; then
    echo "################# NATIVE STACK (HSA_ENABLE_SDMA=0) #################" | tee -a "$LOG_FILE"
    setup_native || { echo "ERROR: setup_native failed" >&2; exit 1; }
    BIN="${OSU_NATIVE_PT2PT}/osu_latency"
else
    echo "################# EESSI STACK (HSA_ENABLE_SDMA=0) ##################" | tee -a "$LOG_FILE"
    setup_eessi || { echo "ERROR: setup_eessi failed" >&2; exit 1; }
    BIN="${OSU_PT2PT}/osu_latency"

    # *** THE TUNING OVERRIDE ***
    # setup_eessi exports FI_CXI_RX_MATCH_MODE=hybrid in common.sh; flip to
    # 'hardware' to force the on-NIC match list and skip the SW-matching
    # fallback. If this drops the 1B latency from ~18.7 μs to ~3-4 μs, the
    # CXI hardware-match path was the bottleneck.
    export FI_CXI_RX_MATCH_MODE=hardware
    echo "[fixed_rxmatch] FI_CXI_RX_MATCH_MODE=$FI_CXI_RX_MATCH_MODE" | tee -a "$LOG_FILE"
fi

for entry in "${INTERNODE_PAIRS[@]}"; do
    read -r g0 g1 lab nic <<< "$entry"
    run_pair 0 "$g0" "$g1" "$lab" "$nic" "$BIN"
done

rm -f "$WRAPPER"
echo
echo "ALL BENCHMARKS COMPLETE  CSV=$CSV_FILE" | tee -a "$LOG_FILE"

OLD_OUT="${RESULT_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
NEW_OUT="${RESULT_DIR}/${FILE_BASE}_${SLURM_JOB_ID}.out"
[[ -f "$OLD_OUT" && "$OLD_OUT" != "$NEW_OUT" ]] && mv "$OLD_OUT" "$NEW_OUT" 2>/dev/null || true
