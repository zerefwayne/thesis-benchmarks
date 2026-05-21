#!/bin/bash
#SBATCH --job-name=osu_bw
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
# osu_bw.sh — A1 inter-node bandwidth (8 cross-node pairs across 2 NIC tiers).
#
# Submit twice — once per stack — to get stack-tagged outputs:
#   sbatch osu_bw.sh eessi
#   sbatch osu_bw.sh native
#
# Each pair runs NUM_RUNS=6 (1 warm-up discarded + 5 recorded). Native uses
# srun --nodes=2 --ntasks-per-node=1; EESSI uses mpirun --host A:1,B:1
# --map-by ppr:1:node. HSA_ENABLE_SDMA hardcoded 0 per 4_osu.
#
# CSV schema (12 columns):
#   stack, sdma_enabled, pair_label, node_a, node_b, gcd_a, gcd_b,
#   hop_class, nic_class, run, size_bytes, bandwidth_MBps
#   (hop_class is always "inter_node" — column kept for cross-CSV stability.)

STACK="${1:?usage: sbatch $0 <eessi|native>}"
if [[ "$STACK" != "eessi" && "$STACK" != "native" ]]; then
    echo "ERROR: stack must be 'eessi' or 'native', got '$STACK'" >&2
    exit 1
fi

source common.sh
source topology.sh

FILE_BASE="osu_bw_${STACK}"
CSV_FILE="${RESULT_DIR}/${FILE_BASE}_${SLURM_JOB_ID}.csv"
LOG_FILE="${RESULT_DIR}/${FILE_BASE}_${SLURM_JOB_ID}.log"
META_FILE="${RESULT_DIR}/${FILE_BASE}_${SLURM_JOB_ID}.meta"

node_metadata_dump "$META_FILE"

NODES=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
if [[ ${#NODES[@]} -lt 2 ]]; then
    echo "ERROR: need 2 nodes, got ${#NODES[@]} (${NODES[*]})" | tee -a "$LOG_FILE"
    exit 1
fi
NODE_A=${NODES[0]}; NODE_B=${NODES[1]}
echo "STACK=$STACK  NODE_A=$NODE_A  NODE_B=$NODE_B" | tee -a "$LOG_FILE"

echo "stack,sdma_enabled,pair_label,node_a,node_b,gcd_a,gcd_b,hop_class,nic_class,run,size_bytes,bandwidth_MBps" \
    > "$CSV_FILE"

NUM_RUNS=6
WARMUP_RUN=1
RUN_TIMEOUT=300
# 1 MiB cap — inter-node BW plateaus by ~256 KiB; going to 64 MiB just spends
# walltime confirming the plateau (and 5_osu_internode showed >=128 MiB can hang
# on the UCX CMA fallback when CXI isn't engaged).
OSU_FLAGS="-m 1:1048576 -i 100 -d rocm D D"

# Per-rank wrapper — same idiom as 5_osu_internode/osu_bw_internode.sh:73-82.
# SLURM_PROCID is set by srun; OMPI_COMM_WORLD_RANK / PMIX_RANK cover the
# mpirun launchers.
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

parse_pt2pt() {
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
            echo "$raw" | parse_pt2pt "$STACK" "$sdma" "$lab" "$NODE_A" "$NODE_B" \
                "$g0" "$g1" "$nic" "$run" >> "$CSV_FILE"
        fi
    done
}

# ============================================================================
export HSA_ENABLE_SDMA=0
if [[ "$STACK" == "native" ]]; then
    echo "################# NATIVE STACK (HSA_ENABLE_SDMA=0) #################" | tee -a "$LOG_FILE"
    setup_native || { echo "ERROR: setup_native failed" >&2; exit 1; }
    BIN="${OSU_NATIVE_PT2PT}/osu_bw"
else
    echo "################# EESSI STACK (HSA_ENABLE_SDMA=0) ##################" | tee -a "$LOG_FILE"
    setup_eessi || { echo "ERROR: setup_eessi failed" >&2; exit 1; }
    BIN="${OSU_PT2PT}/osu_bw"
fi

for entry in "${INTERNODE_PAIRS[@]}"; do
    read -r g0 g1 lab nic <<< "$entry"
    run_pair 0 "$g0" "$g1" "$lab" "$nic" "$BIN"
done

rm -f "$WRAPPER"
echo
echo "================================================================" | tee -a "$LOG_FILE"
echo "ALL BENCHMARKS COMPLETE"                                            | tee -a "$LOG_FILE"
echo "  CSV : $CSV_FILE"                                                  | tee -a "$LOG_FILE"
echo "  Log : $LOG_FILE  Meta: $META_FILE"                                | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"

# Rename SLURM's .out (named after --job-name) to include the stack.
OLD_OUT="${RESULT_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
NEW_OUT="${RESULT_DIR}/${FILE_BASE}_${SLURM_JOB_ID}.out"
[[ -f "$OLD_OUT" && "$OLD_OUT" != "$NEW_OUT" ]] && mv "$OLD_OUT" "$NEW_OUT" 2>/dev/null || true
