#!/bin/bash
#SBATCH --job-name=osu_allreduce_diag
#SBATCH --account=project_462000226
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --exclusive
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=7
#SBATCH --time=00:30:00
#SBATCH --output=results/%x_%j.out
#SBATCH --constraint=eessi
#
# Goal: isolate the source of the ~25us small-message allreduce floor on
# EESSI vs ~0.4us on Cray. We hold the workload fixed (osu_allreduce, N=2,
# devices 0,1 = intra-package 4-link xGMI pair, message sizes 1B..1KiB) and
# vary the transport stack across a small number of configurations. Each
# section logs:
#   - which UCC TLs and UCC CLs were actually available,
#   - the OSU MPI-ROCM latency table for that config,
#   - mpirun stderr (so we don't silently lose UCC/UCX init failures).
#
# After this completes, the "SUMMARY" block at the bottom of the .out file
# lists the 4-byte latency per config. The contrast between configs tells us
# where the floor lives (RCCL launch, UCC framework, UCX rocm transport, OMPI
# accelerator probe, or the OSU benchmark itself).

source common.sh

RESULT_DIR="results"
mkdir -p "$RESULT_DIR"
LOG_FILE="$RESULT_DIR/osu_allreduce_diag_${SLURM_JOB_ID}.log"
SUMMARY_FILE="$RESULT_DIR/osu_allreduce_diag_${SLURM_JOB_ID}.summary"

# Pin to GCDs 0,1 (intra-package, 4-link xGMI) and to the CPU cores LUMI's
# standard-g layout assigns those GCDs.
export ROCR_VISIBLE_DEVICES=0,1
DEVICES=0,1
CPU_MASK=49,57

# Small message range only — that's where the floor sits and where the
# native stack is sub-microsecond.
MSG_RANGE="1:1024"

# Iterations: keep them low so verbose logs stay readable. OSU defaults are
# already fine for stable averages at small sizes.

OSU_BIN="${OSU_COLL}/osu_allreduce"
OSU_XCCL_DIR="${EBROOTOSUMINMICROMINBENCHMARKS}/libexec/osu-micro-benchmarks/xccl/collective"
OSU_XCCL_BIN="${OSU_XCCL_DIR}/osu_xccl_allreduce"

# ---------------------------------------------------------------------------
# One-time environment snapshot: print what UCX/UCC actually offer on this
# node. If rocm_ipc is missing here, every config that relies on peer GPU
# memory access is going to fall back to host-bounced rocm_copy, and no amount
# of MCA tuning will close the gap.
# ---------------------------------------------------------------------------
{
    echo "================================================================"
    echo "Diagnostic run: $(date)"
    echo "Hostname:        $(hostname)"
    echo "ROCR_VISIBLE_DEVICES: $ROCR_VISIBLE_DEVICES"
    echo "OSU MPI binary:  $OSU_BIN"
    echo "OSU XCCL binary: $OSU_XCCL_BIN"
    echo
    echo "--- UCX transports advertised on this node ---"
    ucx_info -d 2>/dev/null | awk '
        /^# Transport:/ { tr=$3 }
        /Device:/ && tr ~ /rocm|sm|self|cma|xpmem/ { printf "  %-12s %s\n", tr, $0 }
    '
    echo
    echo "--- UCC components/TLs/CLs built ---"
    ucc_info -c 2>/dev/null | grep -iE "(TLS|CLS|rocm|rccl|ucp)" | head -40
    echo
    echo "--- coll/ucc MCA parameters seen by ompi_info ---"
    ompi_info --param coll ucc --level 9 2>/dev/null | head -60
    echo "================================================================"
    echo
} | tee -a "$LOG_FILE"

# ---------------------------------------------------------------------------
# Helper: run one config. We bracket the run with banners, capture stderr
# (the previous bug — silently lost UCC init errors), and append the OSU
# table verbatim. The 4-byte latency line is also extracted into the summary
# file so we can compare configs at a glance.
# ---------------------------------------------------------------------------
run_config() {
    local label=$1; shift   # remaining args are KEY=VAL env assignments

    {
        echo
        echo "################################################################"
        echo "# CONFIG: $label"
        echo "#   env: $*"
        echo "################################################################"
    } | tee -a "$LOG_FILE"

    # Warm-up (silent) — keeps the first-call init costs out of the recorded
    # numbers without hiding errors: stderr still goes to the log.
    env "$@" \
        mpirun --report-bindings -n 2 --cpu-set "$CPU_MASK" \
        "$OSU_BIN" -d rocm -m "$MSG_RANGE" >> "$LOG_FILE" 2>&1 \
        || echo "  WARN: warm-up exited non-zero" | tee -a "$LOG_FILE"

    # Recorded run — capture stdout into raw so we can both append it to
    # the log and pull the 4-byte line out for the summary.
    local raw rc
    raw=$(env "$@" \
          mpirun -n 2 --cpu-set "$CPU_MASK" \
          "$OSU_BIN" -d rocm -m "$MSG_RANGE" 2>>"$LOG_FILE")
    rc=$?

    echo "$raw" >> "$LOG_FILE"
    if (( rc != 0 )); then
        echo "  ERROR: mpirun exit=$rc — see $LOG_FILE" | tee -a "$LOG_FILE"
        printf "%-40s  ERROR (rc=%d)\n" "$label" "$rc" >> "$SUMMARY_FILE"
        return
    fi

    # Extract the 4-byte latency (OSU prints `<size>\s+<lat>`); fall back to
    # the first numeric data line if 4B isn't present.
    local lat
    lat=$(echo "$raw" | awk '$1=="4" && $2 ~ /^[0-9.]+$/ {print $2; exit}')
    [[ -z "$lat" ]] && \
        lat=$(echo "$raw" | awk '/^#/{next} $1 ~ /^[0-9]+$/ && $2 ~ /^[0-9.]+$/ {print "first="$1"B "$2; exit}')

    printf "%-40s  4B_lat_us=%s\n" "$label" "$lat" >> "$SUMMARY_FILE"
}

# ---------------------------------------------------------------------------
# Helper for the XCCL benchmark — bypasses MPI collectives entirely and calls
# RCCL directly. The latency this reports is RCCL's own floor on this
# hardware; if it's also ~25us, the UCC/OMPI stack on top can't beat it and
# the gap to Cray is RCCL's, not ours.
# ---------------------------------------------------------------------------
run_xccl() {
    local label="xccl_rccl_direct"
    {
        echo
        echo "################################################################"
        echo "# CONFIG: $label  (osu_xccl_allreduce — RCCL direct, no MPI coll)"
        echo "################################################################"
    } | tee -a "$LOG_FILE"

    if [[ ! -x "$OSU_XCCL_BIN" ]]; then
        echo "  SKIP: $OSU_XCCL_BIN not present or not executable" | tee -a "$LOG_FILE"
        printf "%-40s  SKIPPED (binary missing)\n" "$label" >> "$SUMMARY_FILE"
        return
    fi

    local raw rc
    raw=$(mpirun -n 2 --cpu-set "$CPU_MASK" "$OSU_XCCL_BIN" -m "$MSG_RANGE" 2>>"$LOG_FILE")
    rc=$?
    echo "$raw" >> "$LOG_FILE"
    if (( rc != 0 )); then
        printf "%-40s  ERROR (rc=%d)\n" "$label" "$rc" >> "$SUMMARY_FILE"
        return
    fi
    local lat
    lat=$(echo "$raw" | awk '$1=="4" && $2 ~ /^[0-9.]+$/ {print $2; exit}')
    [[ -z "$lat" ]] && \
        lat=$(echo "$raw" | awk '/^#/{next} $1 ~ /^[0-9]+$/ && $2 ~ /^[0-9.]+$/ {print "first="$1"B "$2; exit}')
    printf "%-40s  4B_lat_us=%s\n" "$label" "$lat" >> "$SUMMARY_FILE"
}

# ---------------------------------------------------------------------------
# Common knobs that every UCC-enabled config needs. Without these UCC stays
# at its default priority of 10 and coll/han (35) wins selection.
# ---------------------------------------------------------------------------
UCC_BASE_ENV=(
    OMPI_MCA_coll_ucc_enable=1
    OMPI_MCA_coll_ucc_priority=100
    OMPI_MCA_accelerator=rocm
)

# Verbose flags used only by config A — they dump enough output to identify
# component/TL selection. We don't want this on every config or the .log file
# becomes uninspectable.
VERBOSE_ENV=(
    OMPI_MCA_coll_base_verbose=10
    OMPI_MCA_pml_base_verbose=10
    UCC_LOG_LEVEL=info
    UCX_LOG_LEVEL=info
)

echo "# Summary of 4-byte allreduce latency by config" > "$SUMMARY_FILE"
echo "# Generated $(date) on $(hostname)" >> "$SUMMARY_FILE"
echo >> "$SUMMARY_FILE"

# A — baseline UCC (mirrors the current production setting), with full verbose
# logging so we can see which TL/CL UCC actually selected.
run_config "A_ucc_default_verbose" \
    "${UCC_BASE_ENV[@]}" "${VERBOSE_ENV[@]}"

# B — drop RCCL from UCC's TL list. If A's floor was set by RCCL's per-call
# launch overhead, B will be noticeably lower at small sizes (and worse at
# large ones).
run_config "B_ucc_no_rccl" \
    "${UCC_BASE_ENV[@]}" \
    UCC_TLS=ucp,self UCC_CL_BASIC_TLS=ucp,self

# C — force RCCL as the only UCC TL. If C is ~the same as A's floor, RCCL is
# being chosen in A too and is the bottleneck. If C is much worse, UCC was
# wisely avoiding RCCL for tiny messages in A.
run_config "C_ucc_rccl_only" \
    "${UCC_BASE_ENV[@]}" \
    UCC_TLS=rccl,self UCC_CL_BASIC_TLS=rccl,self

# D — make UCX prefer rocm_ipc (direct xGMI peer access) over rocm_copy
# (host-bounced). This only affects the UCP TL path inside UCC; it should
# pair best with config B's settings.
run_config "D_ucc_no_rccl_ucx_rocm_ipc" \
    "${UCC_BASE_ENV[@]}" \
    UCC_TLS=ucp,self UCC_CL_BASIC_TLS=ucp,self \
    UCX_TLS=self,sm,rocm_copy,rocm_ipc,cma \
    UCX_MEMTYPE_CACHE=n

# E — disable UCC entirely. This is the pre-UCC baseline we already saw in
# results/osu_allreduce_18599319.csv (~26us floor from coll/han + host
# staging). Re-running it here keeps every config in one comparable .log.
run_config "E_no_ucc_baseline" \
    OMPI_MCA_coll_ucc_enable=0 OMPI_MCA_accelerator=rocm

# F — XCCL direct: RCCL's intrinsic latency on this hardware, with no MPI
# collective layer above it. Lower bound for anything routed through RCCL.
run_xccl

# ---------------------------------------------------------------------------
# Final report.
# ---------------------------------------------------------------------------
{
    echo
    echo "================================================================"
    echo "SUMMARY"
    echo "================================================================"
    cat "$SUMMARY_FILE"
    echo
    echo "Full log:     $LOG_FILE"
    echo "Summary file: $SUMMARY_FILE"
    echo "================================================================"
}
