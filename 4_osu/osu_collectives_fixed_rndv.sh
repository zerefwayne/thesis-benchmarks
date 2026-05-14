#!/bin/bash
#SBATCH --job-name=osu_collectives_fixed_rndv
#SBATCH --account=project_462000226
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --exclusive
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --time=02:30:00
#SBATCH --output=results/%x_%j.out
#SBATCH --constraint=eessi
#
# osu_collectives_fixed_rndv.sh — A4 collectives with UCX rendezvous tuning.
#
# Companion to osu_collectives.sh. Only difference: EESSI arm exports
# UCX_RNDV_THRESH=1024 to attack the 256B-4K cliff observed in
# osu_collectives_eessi vs native (bcast 66× slower at 256B, alltoall 12.5×,
# allreduce 9× — same UCX eager-rendezvous switch as in pt2pt, multiplied
# by N=8 fan-out).
#
# Same threshold rationale as osu_bw_fixed_rndv.sh — see fix_rndv_reasoning.md.
# This script ISOLATES the rndv fix; the 1B-128B latency-gap fix lives in
# osu_collectives_ucc_shm.sh.
#
# Submit:
#   sbatch osu_collectives_fixed_rndv.sh eessi
#   sbatch osu_collectives_fixed_rndv.sh native

STACK="${1:?usage: sbatch $0 <eessi|native>}"
if [[ "$STACK" != "eessi" && "$STACK" != "native" ]]; then
    echo "ERROR: stack must be 'eessi' or 'native', got '$STACK'" >&2
    exit 1
fi

source common.sh

FILE_BASE="osu_collectives_fixed_rndv_${STACK}"
CSV_FILE="${RESULT_DIR}/${FILE_BASE}_${SLURM_JOB_ID}.csv"
LOG_FILE="${RESULT_DIR}/${FILE_BASE}_${SLURM_JOB_ID}.log"
META_FILE="${RESULT_DIR}/${FILE_BASE}_${SLURM_JOB_ID}.meta"

node_metadata_dump "$META_FILE"

echo "stack,sdma_enabled,benchmark,num_gcds,run,size_bytes,latency_us" > "$CSV_FILE"

NUM_RUNS=6
WARMUP_RUN=1
RUN_TIMEOUT=600

N_VALUES=(2 4 8)

devices_for_n() {
    case "$1" in
        2) echo "0,1" ;;
        4) echo "0,1,2,3" ;;
        8) echo "0,1,2,3,4,5,6,7" ;;
        *) echo "ERROR: unsupported N=$1" >&2; return 1 ;;
    esac
}

WRAPPER_SCRIPT="${RESULT_DIR}/wrap_gpu_${SLURM_JOB_ID}.sh"
cat > "$WRAPPER_SCRIPT" <<'EOF'
#!/bin/bash
export ROCR_VISIBLE_DEVICES=$1
shift
exec "$@"
EOF
chmod +x "$WRAPPER_SCRIPT"

parse_coll() {
    local stack=$1 sdma=$2 bench=$3 ng=$4 run=$5
    awk -v st="$stack" -v sdma="$sdma" -v bench="$bench" \
        -v ng="$ng" -v run="$run" '
        /^#/ { next }
        NF >= 2 && $1 ~ /^[0-9]+$/ {
            printf "%s,%d,%s,%d,%d,%d,%.2f\n", st, sdma, bench, ng, run, $1, $2
        }'
}

run_native() {
    local sdma=$1 benchmark=$2 bin=$3 ng=$4 devices=$5
    { echo; echo "# [native sdma=$sdma] $benchmark  N=$ng  devices=$devices"; } | tee -a "$LOG_FILE"
    for run in $(seq 1 $NUM_RUNS); do
        local t0=$SECONDS
        if (( run == WARMUP_RUN )); then
            echo "  [run $run/$NUM_RUNS] warm-up — $(date '+%H:%M:%S')" | tee -a "$LOG_FILE"
            timeout $RUN_TIMEOUT srun --ntasks=$ng \
                "$WRAPPER_SCRIPT" "$devices" "$bin" -d rocm > /dev/null 2>&1
            echo "  [run $run/$NUM_RUNS] warm-up done in $((SECONDS - t0))s" | tee -a "$LOG_FILE"
            continue
        fi
        echo "  [run $run/$NUM_RUNS] recording — $(date '+%H:%M:%S')" | tee -a "$LOG_FILE"
        local raw
        raw=$(timeout $RUN_TIMEOUT srun --ntasks=$ng \
            "$WRAPPER_SCRIPT" "$devices" "$bin" -d rocm 2>>"$LOG_FILE")
        echo "  [run $run/$NUM_RUNS] done in $((SECONDS - t0))s" | tee -a "$LOG_FILE"
        echo "$raw" >> "$LOG_FILE"
        echo "$raw" | parse_coll native "$sdma" "$benchmark" "$ng" "$run" >> "$CSV_FILE"
    done
}

run_eessi() {
    local sdma=$1 benchmark=$2 bin=$3 ng=$4 devices=$5
    { echo; echo "# [eessi sdma=$sdma rndv=$UCX_RNDV_THRESH] $benchmark  N=$ng  devices=$devices"; } | tee -a "$LOG_FILE"
    for run in $(seq 1 $NUM_RUNS); do
        local t0=$SECONDS
        if (( run == WARMUP_RUN )); then
            echo "  [run $run/$NUM_RUNS] warm-up — $(date '+%H:%M:%S')" | tee -a "$LOG_FILE"
            timeout $RUN_TIMEOUT bash -c \
                "ROCR_VISIBLE_DEVICES=$devices mpirun -n $ng -x UCX_RNDV_THRESH $bin -d rocm" \
                > /dev/null 2>&1
            echo "  [run $run/$NUM_RUNS] warm-up done in $((SECONDS - t0))s" | tee -a "$LOG_FILE"
            continue
        fi
        echo "  [run $run/$NUM_RUNS] recording — $(date '+%H:%M:%S')" | tee -a "$LOG_FILE"
        local raw
        raw=$(timeout $RUN_TIMEOUT bash -c \
            "ROCR_VISIBLE_DEVICES=$devices mpirun -n $ng -x UCX_RNDV_THRESH $bin -d rocm" \
            2>>"$LOG_FILE")
        echo "  [run $run/$NUM_RUNS] done in $((SECONDS - t0))s" | tee -a "$LOG_FILE"
        echo "$raw" >> "$LOG_FILE"
        echo "$raw" | parse_coll eessi "$sdma" "$benchmark" "$ng" "$run" >> "$CSV_FILE"
    done
}

COLLECTIVES=(osu_allreduce osu_alltoall osu_bcast osu_allgather)

# ============================================================================
export HSA_ENABLE_SDMA=0
if [[ "$STACK" == "native" ]]; then
    echo "############ NATIVE STACK (HSA_ENABLE_SDMA=0, default IPC) ############" | tee -a "$LOG_FILE"
    setup_native || { echo "ERROR: setup_native failed" >&2; exit 1; }
    for binary in "${COLLECTIVES[@]}"; do
        if [[ ! -x "${OSU_NATIVE_COLL}/${binary}" ]]; then
            echo "  [skip] $binary not in $OSU_NATIVE_COLL" | tee -a "$LOG_FILE"
            continue
        fi
        for n in "${N_VALUES[@]}"; do
            devices=$(devices_for_n "$n")
            run_native 0 "$binary" "${OSU_NATIVE_COLL}/${binary}" "$n" "$devices"
        done
    done
else
    echo "## EESSI STACK (HSA_ENABLE_SDMA=0, UCC=ucp+self, UCX_RNDV_THRESH=1024) ##" | tee -a "$LOG_FILE"
    setup_eessi || { echo "ERROR: setup_eessi failed" >&2; exit 1; }

    # UCC tuning — same as osu_collectives.sh.
    export OMPI_MCA_opal_common_ucx_devices=any
    export OMPI_MCA_coll_ucc_enable=1
    export OMPI_MCA_coll_ucc_priority=100
    export OMPI_MCA_accelerator=rocm
    export UCC_TLS=ucp,self
    export UCC_CL_BASIC_TLS=ucp,self

    # The fix: push the eager->rendezvous switch from UCX default (~256B)
    # up to 1024B. Same value used in osu_bw_fixed_rndv.sh.
    export UCX_RNDV_THRESH=1024

    for binary in "${COLLECTIVES[@]}"; do
        if [[ ! -x "${OSU_COLL}/${binary}" ]]; then
            echo "  [skip] $binary not in $OSU_COLL" | tee -a "$LOG_FILE"
            continue
        fi
        for n in "${N_VALUES[@]}"; do
            devices=$(devices_for_n "$n")
            run_eessi 0 "$binary" "${OSU_COLL}/${binary}" "$n" "$devices"
        done
    done
fi

rm -f "$WRAPPER_SCRIPT"
echo
echo "ALL BENCHMARKS COMPLETE  CSV=$CSV_FILE" | tee -a "$LOG_FILE"

OLD_OUT="${RESULT_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
NEW_OUT="${RESULT_DIR}/${FILE_BASE}_${SLURM_JOB_ID}.out"
[[ -f "$OLD_OUT" && "$OLD_OUT" != "$NEW_OUT" ]] && mv "$OLD_OUT" "$NEW_OUT" 2>/dev/null || true
