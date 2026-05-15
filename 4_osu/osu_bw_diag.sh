#!/bin/bash
#SBATCH --job-name=osu_bw_diag
#SBATCH --account=project_462000226
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --exclusive
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=28
#SBATCH --time=00:15:00
#SBATCH --output=results/%x_%j.out
#SBATCH --constraint=eessi
#
# osu_bw_diag.sh — diagnostic capture for the 256B-1KiB slow zone on
# MI250X xGMI with UCX 1.18 / Open MPI 5.0.7 / ROCm 6.4.1 (EESSI 2025.06)
# vs Cray MPICH.
#
# Implements Recommendation #1 from the
#   "MI250X xGMI Small-Message Bandwidth Dip" structural analysis PDF.
#
# Captures everything needed to claim — with certainty — *which* protocol
# UCX/Cray pick at each message size and what the per-transport latency/
# overhead/bandwidth budget looks like on our exact stack.
#
# Single pair (0,1), intra_pkg 4-link.
#
# EESSI arm collects:
#   - ucxinfo_devices.txt  : ucx_info -d -u t — per-TL capabilities
#   - ucxinfo_config.txt   : ucx_info -c (PROTO/RNDV/ROCM/MEMTYPE keys)
#   - proto_<THRESH>.log   : full UCX_PROTO_INFO=y + UCX_LOG_LEVEL=info
#                            output for THRESH in {DEFAULT, 128, 1024}
#
# Native arm collects:
#   - mpich_env.txt        : MPICH_VERSION_DISPLAY + MPICH_ENV_DISPLAY
#                            dump from one representative run
#   - mpich_thresh_<T>.log : per-IPC-threshold osu_bw output for
#                            T in {DEFAULT, 1024, 8192}
#
# Submit:
#   sbatch osu_bw_diag.sh eessi
#   sbatch osu_bw_diag.sh native

STACK="${1:?usage: sbatch $0 <eessi|native>}"
if [[ "$STACK" != "eessi" && "$STACK" != "native" ]]; then
    echo "ERROR: stack must be 'eessi' or 'native', got '$STACK'" >&2
    exit 1
fi

source common.sh
source topology.sh

FILE_BASE="osu_bw_diag_${STACK}"
DIAG_DIR="${RESULT_DIR}/${FILE_BASE}_${SLURM_JOB_ID}"
META_FILE="${RESULT_DIR}/${FILE_BASE}_${SLURM_JOB_ID}.meta"
SUMMARY_FILE="${RESULT_DIR}/${FILE_BASE}_${SLURM_JOB_ID}.log"
mkdir -p "$DIAG_DIR"

node_metadata_dump "$META_FILE"

# Single pair (0,1) — intra_pkg 4-link, the cleanest xGMI path.
G0=0; G1=1
LABEL="intra_pkg_OAM0"

# We don't need statistics here; we want clean diagnostic output.
# 1 warmup (silent) + 1 recorded run per threshold.
NUM_WARMUP=1
NUM_RECORDED=1
RUN_TIMEOUT=300

# Smaller iter count for diagnostic runs — UCX_PROTO_INFO output is captured
# at init time, OSU iter count doesn't change what protocols get printed.
# Keep -i 100 to match the benchmark scripts' table shape.
OSU_FLAGS="-m 1:67108864 -i 100 -d rocm D D"

log_banner() {
    {
        echo
        echo "================================================================"
        echo "# $1"
        echo "================================================================"
    } | tee -a "$SUMMARY_FILE"
}

# ============================================================================
export HSA_ENABLE_SDMA=0

if [[ "$STACK" == "eessi" ]]; then
    log_banner "EESSI STACK (HSA_ENABLE_SDMA=0) — pair (0,1) intra_pkg"
    setup_eessi || { echo "ERROR: setup_eessi failed" >&2; exit 1; }

    # 1. Capture per-transport capabilities — quotable in the thesis.
    log_banner "ucx_info -d -u t  (per-transport caps; expect rocm_copy and rocm_ipc)"
    {
        echo "# === ucx_info -d -u t (capabilities) @ $(date -Iseconds) ==="
        echo "# UCX build: $(ucx_info -v 2>&1 | head -3)"
        echo
        ucx_info -d -u t 2>&1
    } > "$DIAG_DIR/ucxinfo_devices.txt"
    echo "  wrote $DIAG_DIR/ucxinfo_devices.txt" | tee -a "$SUMMARY_FILE"

    # 2. Config defaults relevant to the slow zone analysis.
    log_banner "ucx_info -c  (PROTO_ENABLE / RNDV / ROCM / MEMTYPE defaults)"
    {
        echo "# === ucx_info -c — relevant defaults @ $(date -Iseconds) ==="
        ucx_info -c 2>&1 | grep -E "PROTO_ENABLE|RNDV|ROCM|MEMTYPE|TLS" | sort
    } > "$DIAG_DIR/ucxinfo_config.txt"
    echo "  wrote $DIAG_DIR/ucxinfo_config.txt" | tee -a "$SUMMARY_FILE"

    # 3. Threshold sweep with UCX_PROTO_INFO=y so the per-size protocol
    #    selection table is printed on every run. PDF Rec #1.
    export UCX_PROTO_ENABLE=y    # force protov2 (default since UCX 1.16)
    export UCX_LOG_LEVEL=info
    export UCX_PROTO_INFO=y

    for thresh in DEFAULT 128 1024; do
        log_banner "EESSI run @ UCX_RNDV_THRESH=$thresh"
        if [[ "$thresh" == "DEFAULT" ]]; then
            unset UCX_RNDV_THRESH
        else
            export UCX_RNDV_THRESH=$thresh
        fi

        local_log="$DIAG_DIR/proto_thresh_${thresh}.log"
        {
            echo "# === EESSI / UCX_RNDV_THRESH=$thresh @ $(date -Iseconds) ==="
            echo "# pair=($G0,$G1) tier=$LABEL  HSA_ENABLE_SDMA=$HSA_ENABLE_SDMA"
            echo "# UCX_PROTO_ENABLE=$UCX_PROTO_ENABLE  UCX_PROTO_INFO=$UCX_PROTO_INFO  UCX_LOG_LEVEL=$UCX_LOG_LEVEL"
            echo "# OSU_FLAGS='$OSU_FLAGS'"
            echo
        } > "$local_log"

        # Warmup (silent — discard noise from connection setup churn)
        echo "  [warmup $(date '+%H:%M:%S')]" | tee -a "$SUMMARY_FILE"
        timeout $RUN_TIMEOUT bash -c "
            ROCR_VISIBLE_DEVICES=$G0,$G1 \
            mpirun -n 2 \
              -x HSA_ENABLE_SDMA \
              -x UCX_PROTO_ENABLE -x UCX_PROTO_INFO -x UCX_LOG_LEVEL \
              ${UCX_RNDV_THRESH:+-x UCX_RNDV_THRESH} \
              ${OSU_PT2PT}/osu_bw $OSU_FLAGS" \
            > /dev/null 2>&1

        # Recorded run — captures PROTO_INFO + UCX init log + OSU bandwidth table
        echo "  [recorded $(date '+%H:%M:%S')]" | tee -a "$SUMMARY_FILE"
        echo "# ==== recorded run begin ====" >> "$local_log"
        timeout $RUN_TIMEOUT bash -c "
            ROCR_VISIBLE_DEVICES=$G0,$G1 \
            mpirun -n 2 \
              -x HSA_ENABLE_SDMA \
              -x UCX_PROTO_ENABLE -x UCX_PROTO_INFO -x UCX_LOG_LEVEL \
              ${UCX_RNDV_THRESH:+-x UCX_RNDV_THRESH} \
              ${OSU_PT2PT}/osu_bw $OSU_FLAGS" \
            >> "$local_log" 2>&1
        echo "# ==== recorded run end ====" >> "$local_log"

        echo "  wrote $local_log" | tee -a "$SUMMARY_FILE"
    done

else
    # ========================== NATIVE ARM ==================================
    log_banner "NATIVE STACK (HSA_ENABLE_SDMA=0) — pair (0,1) intra_pkg"
    setup_native || { echo "ERROR: setup_native failed" >&2; exit 1; }

    # 1. One representative run with full MPICH env+version dump.
    log_banner "MPICH_VERSION_DISPLAY=1 MPICH_ENV_DISPLAY=1 (default IPC threshold)"
    env_log="$DIAG_DIR/mpich_env.txt"
    {
        echo "# === MPICH env+version dump @ $(date -Iseconds) ==="
        echo "# pair=($G0,$G1) tier=$LABEL  HSA_ENABLE_SDMA=$HSA_ENABLE_SDMA"
        echo
    } > "$env_log"
    timeout $RUN_TIMEOUT srun --ntasks=2 --ntasks-per-node=2 \
        bash -c "export ROCR_VISIBLE_DEVICES=$G0,$G1 \
            MPICH_VERSION_DISPLAY=1 MPICH_ENV_DISPLAY=1 MPICH_OFI_VERBOSE=1; \
            exec ${OSU_NATIVE_PT2PT}/osu_bw $OSU_FLAGS" \
        >> "$env_log" 2>&1
    echo "  wrote $env_log" | tee -a "$SUMMARY_FILE"

    # 2. IPC-threshold sweep — captures bandwidth curves at each setting
    #    on the same node as the EESSI runs, so the comparison is clean.
    for thresh in DEFAULT 1024 8192; do
        log_banner "NATIVE run @ MPICH_GPU_IPC_THRESHOLD=$thresh"
        if [[ "$thresh" == "DEFAULT" ]]; then
            unset MPICH_GPU_IPC_THRESHOLD
        else
            export MPICH_GPU_IPC_THRESHOLD=$thresh
        fi

        local_log="$DIAG_DIR/mpich_thresh_${thresh}.log"
        {
            echo "# === NATIVE / MPICH_GPU_IPC_THRESHOLD=$thresh @ $(date -Iseconds) ==="
            echo "# pair=($G0,$G1) tier=$LABEL  HSA_ENABLE_SDMA=$HSA_ENABLE_SDMA"
            echo "# OSU_FLAGS='$OSU_FLAGS'"
            echo
        } > "$local_log"

        # Warmup
        echo "  [warmup $(date '+%H:%M:%S')]" | tee -a "$SUMMARY_FILE"
        timeout $RUN_TIMEOUT srun --ntasks=2 --ntasks-per-node=2 \
            bash -c "export ROCR_VISIBLE_DEVICES=$G0,$G1; \
                exec ${OSU_NATIVE_PT2PT}/osu_bw $OSU_FLAGS" \
            > /dev/null 2>&1

        # Recorded
        echo "  [recorded $(date '+%H:%M:%S')]" | tee -a "$SUMMARY_FILE"
        echo "# ==== recorded run begin ====" >> "$local_log"
        timeout $RUN_TIMEOUT srun --ntasks=2 --ntasks-per-node=2 \
            bash -c "export ROCR_VISIBLE_DEVICES=$G0,$G1; \
                exec ${OSU_NATIVE_PT2PT}/osu_bw $OSU_FLAGS" \
            >> "$local_log" 2>&1
        echo "# ==== recorded run end ====" >> "$local_log"
        echo "  wrote $local_log" | tee -a "$SUMMARY_FILE"
    done
fi

log_banner "ALL DIAGNOSTIC RUNS COMPLETE"
echo "  diag dir: $DIAG_DIR" | tee -a "$SUMMARY_FILE"
echo "  summary : $SUMMARY_FILE" | tee -a "$SUMMARY_FILE"
ls -la "$DIAG_DIR" | tee -a "$SUMMARY_FILE"

OLD_OUT="${RESULT_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
NEW_OUT="${RESULT_DIR}/${FILE_BASE}_${SLURM_JOB_ID}.out"
[[ -f "$OLD_OUT" && "$OLD_OUT" != "$NEW_OUT" ]] && mv "$OLD_OUT" "$NEW_OUT" 2>/dev/null || true