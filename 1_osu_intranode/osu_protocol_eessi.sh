#!/bin/bash
#SBATCH --job-name=osu_protocol_eessi
#SBATCH --account=project_462000226
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --exclusive
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=28
#SBATCH --time=00:15:00
#SBATCH --output=results/%x_%j.out
#SBATCH --constraint=eessi
#
# osu_protocol_eessi.sh — A5: UCX rendezvous threshold sweep (EESSI)
# Single pair (0,1), SDMA=0, 6 thresholds × 4 runs (1 warmup + 3 recorded).
#
# Finer sweep across the eager / rendezvous transition zone — picks
# powers of 2 from 1024 up to 16384, which brackets the 8 KiB point
# where the original osu_bw_eessi recovered to native parity. DEFAULT
# (~256 B) is kept as the baseline-cliff anchor; 16 MiB is omitted —
# the previous sweep already showed it pins bulk at the ~660 MB/s
# eager bounce-buffer ceiling (catastrophic, no need to repeat).
# Goal: find the largest threshold that fixes the 512 B - 4 KiB band
# without regressing bulk bandwidth at >= 8 KiB.
#
# EESSI-only — no module purge needed since this is the first/only section.

source common.sh
source topology.sh

CSV_FILE="${RESULT_DIR}/${SLURM_JOB_NAME:-osu_protocol_eessi}_${SLURM_JOB_ID}.csv"
LOG_FILE="${RESULT_DIR}/${SLURM_JOB_NAME:-osu_protocol_eessi}_${SLURM_JOB_ID}.log"
META_FILE="${RESULT_DIR}/${SLURM_JOB_NAME:-osu_protocol_eessi}_${SLURM_JOB_ID}.meta"

node_metadata_dump "$META_FILE"

echo "stack,sdma_enabled,ucx_rndv_thresh,pair_label,gcd_a,gcd_b,tier,num_links,run,size_bytes,bandwidth_MBps" \
    > "$CSV_FILE"

NUM_RUNS=4   # 1 warmup + 3 recorded
WARMUP_RUN=1
RUN_TIMEOUT=300
OSU_FLAGS="-m 1:67108864 -i 100 -d rocm D D"

G0=0; G1=1; LABEL="intra_pkg_OAM0"
TOPO=$(get_topology $G0 $G1)
TIER="${TOPO%,*}"; NLINKS="${TOPO#*,}"

# EESSI runs alone here (no native step), so we don't need to do native first.
setup_eessi || { echo "ERROR: setup_eessi failed" >&2; exit 1; }
export HSA_ENABLE_SDMA=0

for rndv in DEFAULT 1024 2048 4096 8192 16384; do
    if [[ "$rndv" == "DEFAULT" ]]; then
        unset UCX_RNDV_THRESH
    else
        export UCX_RNDV_THRESH=$rndv
    fi
    {
        echo
        echo "# [eessi sdma=0 UCX_RNDV_THRESH=$rndv] $LABEL  GCD($G0,$G1)"
    } | tee -a "$LOG_FILE"

    for run in $(seq 1 $NUM_RUNS); do
        t0=$SECONDS
        if (( run == WARMUP_RUN )); then
            echo "  [run $run/$NUM_RUNS] warm-up — $(date '+%H:%M:%S')" | tee -a "$LOG_FILE"
            timeout $RUN_TIMEOUT bash -c \
                "ROCR_VISIBLE_DEVICES=$G0,$G1 mpirun -n 2 ${OSU_PT2PT}/osu_bw $OSU_FLAGS" \
                > /dev/null 2>&1
            echo "  [run $run/$NUM_RUNS] warm-up done in $((SECONDS - t0))s" | tee -a "$LOG_FILE"
            continue
        fi
        echo "  [run $run/$NUM_RUNS] recording — $(date '+%H:%M:%S')" | tee -a "$LOG_FILE"
        raw=$(timeout $RUN_TIMEOUT bash -c \
            "ROCR_VISIBLE_DEVICES=$G0,$G1 mpirun -n 2 ${OSU_PT2PT}/osu_bw $OSU_FLAGS" \
            2>>"$LOG_FILE")
        echo "  [run $run/$NUM_RUNS] done in $((SECONDS - t0))s" | tee -a "$LOG_FILE"
        echo "$raw" >> "$LOG_FILE"
        echo "$raw" | awk \
            -v st="eessi" -v sdma=0 -v rndv="$rndv" -v lab="$LABEL" \
            -v g0="$G0" -v g1="$G1" -v tier="$TIER" -v nl="$NLINKS" -v run="$run" '
            /^#/ { next }
            NF == 2 && $1 ~ /^[0-9]+$/ {
                printf "%s,%d,%s,%s,%d,%d,%s,%d,%d,%d,%.2f\n",
                       st, sdma, rndv, lab, g0, g1, tier, nl, run, $1, $2
            }' >> "$CSV_FILE"
    done
done

echo
echo "ALL BENCHMARKS COMPLETE  CSV=$CSV_FILE" | tee -a "$LOG_FILE"
