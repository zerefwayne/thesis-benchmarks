#!/bin/bash
#SBATCH --job-name=osu_allreduce_native
#SBATCH --account=project_462000226
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --exclusive
#SBATCH --ntasks-per-node=8
#SBATCH --time=00:45:00
#SBATCH --output=results/%x_%j.out

source common.sh

# ESSENTIAL for Cray MPICH Rendezvous protocol (large message GPU-to-GPU)
export MPICH_GPU_SUPPORT_ENABLED=1

RESULT_DIR="results"
mkdir -p "$RESULT_DIR"
CSV_FILE="$RESULT_DIR/osu_allreduce_native_${SLURM_JOB_ID}.csv"
LOG_FILE="$RESULT_DIR/osu_allreduce_native_${SLURM_JOB_ID}.log"

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

# CSV header
echo "benchmark,num_gcds,run,size_bytes,latency_us" > "$CSV_FILE"

run_collective() {
    local n=$1 benchmark=$2 binary_path=$3
    local devices; devices=$(gcd_set "$n")
    local bname;   bname=$(basename "$benchmark")

    {
        echo
        echo "################################################################"
        echo "# $bname  N=$n GCDs   devices=$devices"
        echo "################################################################"
    } | tee -a "$LOG_FILE"

    for run in $(seq 1 $NUM_RUNS); do
        if (( run == WARMUP_RUN )); then
            echo "  [run $run/$NUM_RUNS] warm-up starting" | tee -a "$LOG_FILE"
            srun --ntasks=$n bash -c "export ROCR_VISIBLE_DEVICES=$devices; exec $binary_path -d rocm" > /dev/null 2>&1
            continue
        fi

        echo "  [run $run/$NUM_RUNS] recording" | tee -a "$LOG_FILE"
        local raw
        # The magic trick: inline bash wrapper beats Slurm cgroups to the punch
        raw=$(srun --ntasks=$n bash -c "export ROCR_VISIBLE_DEVICES=$devices; exec $binary_path -d rocm" 2>/dev/null)

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

# Ensure this points to your natively built osu collective directory!
# e.g., /users/joglekar/osu-native/osu-micro-benchmarks-7.5/c/mpi/collective
OSU_COLL="/users/joglekar/osu-native/osu-micro-benchmarks-7.5/c/mpi/collective/blocking"

echo "Using OSU collective binaries from: $OSU_COLL" | tee -a "$LOG_FILE"

for n in 2 4 6 8; do
    run_collective "$n" "osu_allreduce"  "${OSU_COLL}/osu_allreduce"
done

echo
echo "================================================================"
echo "ALL BENCHMARKS COMPLETE"
echo "  CSV: $CSV_FILE"
echo "  Log: $LOG_FILE"
echo "================================================================"