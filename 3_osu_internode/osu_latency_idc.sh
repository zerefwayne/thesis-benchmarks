#!/bin/bash
#SBATCH --job-name=osu_latency_idc
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
# osu_latency_idc.sh — EESSI-only small-message latency tuning sweep.
#
# Targets the ~18 us plateau on inter-node osu_latency for messages <=128 B that
# vanishes at exactly 256 B (the Cassini IDC inline limit). Hypothesis: with GPU
# device buffers (-d rocm D D), the CXI IDC inline path stages payload GPU->host
# per send; the >=256 B DMA path reads GPU memory directly via ROCr/HMEM.
# See .claude/plans/let-s-plan-something-the-eager-hoare.md.
#
# Always uses the EESSI stack (the fix is EESSI-specific). Native baseline for
# comparison already exists at results/osu_latency_native_18750321.csv.
#
# Usage: sbatch osu_latency_idc.sh <variant>
#   baseline        D D, no overrides   (reproduce the plateau in this harness)
#   hostbuf         H H, no overrides   (decisive: GPU-specific?  expect ~3 us flat)
#   noidc           D D + FI_CXI_DISABLE_NON_INJECT_MSG_IDC=1            (primary)
#   noidc_noinject  noidc + OMPI_MCA_mtl_ofi_inject_size=0              (likely fix)
#   optmrs          D D + FI_CXI_OPTIMIZED_MRS=0                        (secondary)
#   llring          D D + FI_CXI_LLRING_MODE=always                    (secondary)
#   rdzvgetmin      D D + FI_CXI_RDZV_GET_MIN=256                       (secondary)
#   verify          print fi_info -p cxi + one mtl-verbose run, then exit

VARIANT="${1:?usage: sbatch $0 <baseline|hostbuf|noidc|noidc_noinject|optmrs|llring|rdzvgetmin|verify>}"

source common.sh
source topology.sh

FILE_BASE="osu_latency_${VARIANT}_eessi"
CSV_FILE="${RESULT_DIR}/${FILE_BASE}_${SLURM_JOB_ID}.csv"
LOG_FILE="${RESULT_DIR}/${FILE_BASE}_${SLURM_JOB_ID}.log"
META_FILE="${RESULT_DIR}/${FILE_BASE}_${SLURM_JOB_ID}.meta"

node_metadata_dump "$META_FILE"

NODES=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
[[ ${#NODES[@]} -ge 2 ]] || { echo "ERROR: need 2 nodes" | tee -a "$LOG_FILE"; exit 1; }
NODE_A=${NODES[0]}; NODE_B=${NODES[1]}
echo "VARIANT=$VARIANT  NODE_A=$NODE_A  NODE_B=$NODE_B" | tee -a "$LOG_FILE"

# --- EESSI stack + baseline CXI selection (from common.sh) ------------------
export HSA_ENABLE_SDMA=0
echo "############### EESSI STACK  variant=$VARIANT  (SDMA=0) ###############" | tee -a "$LOG_FILE"
setup_eessi || { echo "ERROR: setup_eessi failed" >&2; exit 1; }
BIN="${OSU_PT2PT}/osu_latency"

# --- per-variant overrides --------------------------------------------------
# Default: GPU device buffers. hostbuf flips to host buffers.
OSU_MEM="D D"
case "$VARIANT" in
    baseline)        : ;;                                                   # plateau control
    hostbuf)         OSU_MEM="H H" ;;                                       # GPU-specific test
    noidc)           export FI_CXI_DISABLE_NON_INJECT_MSG_IDC=1 ;;
    noidc_noinject)  export FI_CXI_DISABLE_NON_INJECT_MSG_IDC=1
                     export OMPI_MCA_mtl_ofi_inject_size=0 ;;
    optmrs)          export FI_CXI_OPTIMIZED_MRS=0 ;;
    llring)          export FI_CXI_LLRING_MODE=always ;;
    rdzvgetmin)      export FI_CXI_RDZV_GET_MIN=256 ;;
    verify)          : ;;
    *) echo "ERROR: unknown variant '$VARIANT'" >&2; exit 1 ;;
esac
echo "[variant=$VARIANT] OSU_MEM='$OSU_MEM'  IDC_DISABLE=${FI_CXI_DISABLE_NON_INJECT_MSG_IDC:-unset}  INJECT=${OMPI_MCA_mtl_ofi_inject_size:-default}" | tee -a "$LOG_FILE"

# ============================================================================
# verify: confirm cxi is the selected provider + OFI MTL is active, then exit.
if [[ "$VARIANT" == "verify" ]]; then
    {
        echo "=== fi_info -p cxi ==="
        fi_info -p cxi 2>&1 | head -40
        echo "=== one verbose run (mtl_base_verbose=100, FI_LOG warn/cxi) ==="
    } | tee -a "$LOG_FILE"
    timeout 120 mpirun -n 2 --host "$NODE_A:1,$NODE_B:1" --map-by ppr:1:node \
        --mca mtl_base_verbose 100 \
        -x FI_LOG_LEVEL=warn -x FI_LOG_PROV=cxi \
        "$BIN" -i 10 -m 1:256 -d rocm D D 2>&1 | tee -a "$LOG_FILE"
    echo "VERIFY COMPLETE  LOG=$LOG_FILE" | tee -a "$LOG_FILE"
    exit 0
fi

echo "stack,sdma_enabled,pair_label,node_a,node_b,gcd_a,gcd_b,hop_class,nic_class,run,size_bytes,latency_us" \
    > "$CSV_FILE"

NUM_RUNS=6; WARMUP_RUN=1; RUN_TIMEOUT=300
OSU_FLAGS="-i 100 -d rocm $OSU_MEM"

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
    local g0=$1 g1=$2 lab=$3 nic=$4
    {
        echo
        echo "################################################################"
        echo "# [eessi variant=$VARIANT] $lab  GCD($g0,$g1)  nic=$nic"
        echo "################################################################"
    } | tee -a "$LOG_FILE"

    for run in $(seq 1 $NUM_RUNS); do
        local t0=$SECONDS tag raw
        tag=$( (( run == WARMUP_RUN )) && echo "warm-up" || echo "recording" )
        echo "  [run $run/$NUM_RUNS] $tag - $(date '+%H:%M:%S')" | tee -a "$LOG_FILE"
        raw=$(timeout $RUN_TIMEOUT mpirun -n 2 --host "$NODE_A:1,$NODE_B:1" \
            --map-by ppr:1:node \
            "$WRAPPER" "$g0" "$g1" "$BIN" $OSU_FLAGS 2>>"$LOG_FILE")
        echo "  [run $run/$NUM_RUNS] done in $((SECONDS - t0))s" | tee -a "$LOG_FILE"
        echo "$raw" >> "$LOG_FILE"
        if (( run != WARMUP_RUN )); then
            echo "$raw" | parse_lat "eessi" 0 "$lab" "$NODE_A" "$NODE_B" \
                "$g0" "$g1" "$nic" "$run" >> "$CSV_FILE"
        fi
    done
}

# Two representative pairs: one nic_via_xgmi (even GCD) + one nic_local (odd GCD).
EXP_PAIRS=(
    "0 0 inter_GCD0_GCD0 nic_via_xgmi"
    "7 7 inter_GCD7_GCD7 nic_local"
)
for entry in "${EXP_PAIRS[@]}"; do
    read -r g0 g1 lab nic <<< "$entry"
    run_pair "$g0" "$g1" "$lab" "$nic"
done

rm -f "$WRAPPER"
echo
echo "VARIANT=$VARIANT COMPLETE  CSV=$CSV_FILE" | tee -a "$LOG_FILE"

OLD_OUT="${RESULT_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
NEW_OUT="${RESULT_DIR}/${FILE_BASE}_${SLURM_JOB_ID}.out"
[[ -f "$OLD_OUT" && "$OLD_OUT" != "$NEW_OUT" ]] && mv "$OLD_OUT" "$NEW_OUT" 2>/dev/null || true
