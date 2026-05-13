#!/bin/bash
#SBATCH --job-name=osu_allreduce
#SBATCH --account=project_462000226
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --exclusive
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --time=00:45:00
#SBATCH --output=results/%x_%j.out
#SBATCH --constraint=eessi

source common.sh

RESULT_DIR="results"
mkdir -p "$RESULT_DIR"
CSV_FILE="$RESULT_DIR/osu_allreduce_${SLURM_JOB_ID}.csv"
LOG_FILE="$RESULT_DIR/osu_allreduce_${SLURM_JOB_ID}.log"

NUM_RUNS=6
WARMUP_RUN=1

# GCD count → which GCDs to use (which devices ROCR_VISIBLE_DEVICES exposes)
# We pick GCDs that minimise topology asymmetry where possible.
gcd_set() {
    case $1 in
        2) echo "0,1" ;;          # intra-package, 4-link
        4) echo "0,1,2,3" ;;      # 2 full packages
        6) echo "0,1,2,3,4,5" ;;  # 3 full packages
        8) echo "0,1,2,3,4,5,6,7" ;;
    esac
}

# Canonical GCD→CPU mapping for cpu-bind (LUMI standard-g layout)
cpu_for_rank() {
    local rank=$1
    case $rank in
        0) echo 49 ;; 1) echo 57 ;;
        2) echo 17 ;; 3) echo 25 ;;
        4) echo  1 ;; 5) echo  9 ;;
        6) echo 33 ;; 7) echo 41 ;;
    esac
}

# Build CPU mask for N ranks, in order
cpu_mask() {
    local n=$1 i=0 mask=""
    while (( i < n )); do
        if (( i == 0 )); then
            mask=$(cpu_for_rank $i)
        else
            mask="${mask},$(cpu_for_rank $i)"
        fi
        ((i++))
    done
    echo "$mask"
}

# CSV header
echo "benchmark,num_gcds,run,size_bytes,latency_us" > "$CSV_FILE"

run_collective() {
    local n=$1 benchmark=$2 binary_path=$3
    local devices; devices=$(gcd_set "$n")
    local mask;    mask=$(cpu_mask "$n")
    local bname;   bname=$(basename "$benchmark")

    {
        echo
        echo "################################################################"
        echo "# $bname  N=$n GCDs   devices=$devices   cpu_mask=$mask"
        echo "################################################################"
    } | tee -a "$LOG_FILE"

    for run in $(seq 1 $NUM_RUNS); do
        if (( run == WARMUP_RUN )); then
            echo "  [run $run/$NUM_RUNS] warm-up starting" | tee -a "$LOG_FILE"
            ROCR_VISIBLE_DEVICES=$devices \
                mpirun -n $n "$binary_path" -d rocm > /dev/null 2>&1
            continue
        fi

        echo "  [run $run/$NUM_RUNS] recording" | tee -a "$LOG_FILE"
        local raw
        raw=$(ROCR_VISIBLE_DEVICES=$devices \
              mpirun -n $n "$binary_path" -d rocm 2>/dev/null)

        echo "$raw" >> "$LOG_FILE"

        # osu collectives output: "Size  Avg_Latency  [optional Min Max Iter]"
        # We only keep size and avg latency (column 1 and 2)
        echo "$raw" | awk \
            -v bench="$bname" -v ng="$n" -v run="$run" '
            /^#/ { next }
            NF >= 2 && $1 ~ /^[0-9]+$/ {
                printf "%s,%d,%d,%d,%.2f\n", bench, ng, run, $1, $2
            }' >> "$CSV_FILE"
    done
}

# ============================================================================
# EXECUTION
# ============================================================================

OSU_COLL="${OSU_COLLECTIVE:-${OSU_PT2PT%/pt2pt}/collective}"

echo "Using OSU collective binaries from: $OSU_COLL" | tee -a "$LOG_FILE"

for n in 2 4 6 8; do
    # run_collective "$n" "osu_alltoall"  "${OSU_COLL}/osu_alltoall"
    run_collective "$n" "osu_allreduce" "${OSU_COLL}/osu_allreduce"
done

echo
echo "================================================================"
echo "ALL BENCHMARKS COMPLETE"
echo "  CSV: $CSV_FILE"
echo "  Log: $LOG_FILE"
echo "================================================================"