#!/bin/bash
#SBATCH --job-name=osu_latency
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
# osu_latency.sh — A1 inter-node ping-pong latency.
# Same pair sweep / launchers as osu_bw.sh. osu_latency's OSU default cap
# (1 MiB) matches the explicit cap used elsewhere in this directory; we leave
# -m unset. CSV's final column is latency_us.

STACK="${1:?usage: sbatch $0 <eessi|native>}"
[[ "$STACK" == "eessi" || "$STACK" == "native" ]] \
    || { echo "ERROR: stack must be 'eessi' or 'native', got '$STACK'" >&2; exit 1; }

source common.sh
source topology.sh

FILE_BASE="osu_latency_${STACK}"
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
