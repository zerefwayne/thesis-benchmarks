#!/bin/bash
#SBATCH --job-name=osu_alltoall_native
#SBATCH --account=project_462000226
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --exclusive
#SBATCH --ntasks-per-node=8
#SBATCH --time=00:45:00
#SBATCH --output=results/%x_%j.out

source common.sh

# ESSENTIAL for Cray MPICH Rendezvous protocol
export MPICH_GPU_SUPPORT_ENABLED=1

RESULT_DIR="results"
mkdir -p "$RESULT_DIR"
CSV_FILE="$RESULT_DIR/osu_alltoall_native_${SLURM_JOB_ID}.csv"
LOG_FILE="$RESULT_DIR/osu_alltoall_native_${SLURM_JOB_ID}.log"

# --- THE FIX: Create a dedicated GPU wrapper script ---
WRAPPER_SCRIPT="$RESULT_DIR/wrap_gpu_${SLURM_JOB_ID}.sh"
cat << 'EOF' > "$WRAPPER_SCRIPT"
#!/bin/bash
export ROCR_VISIBLE_DEVICES=$1
shift
exec "$@"
EOF
chmod +x "$WRAPPER_SCRIPT"
# ----------------------------------------------------

NUM_RUNS=6
WARMUP_RUN=1

gcd_set() {
    case $1 in
        2) echo "0,1" ;;          
        4) echo "0,1,2,3" ;;      
        6) echo "0,1,2,3,4,5" ;;  
        8) echo "0,1,2,3,4,5,6,7" ;;
    esac
}

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
            srun --ntasks=$n "$WRAPPER_SCRIPT" "$devices" "$binary_path" -d rocm > /dev/null 2>&1
            continue
        fi

        echo "  [run $run/$NUM_RUNS] recording" | tee -a "$LOG_FILE"
        
        # Changed to 2>&1 to capture STDERR into the raw variable
        local raw
        raw=$(srun --ntasks=$n "$WRAPPER_SCRIPT" "$devices" "$binary_path" -d rocm 2>&1)

        # This will now dump the exact crash reason into your .log file!
        echo "$raw" >> "$LOG_FILE"

        echo "$raw" | awk \
            -v bench="$bname" -v ng="$n" -v run="$run" '
            /^#/ { next }
            NF >= 2 && $1 ~ /^[0-9]+$/ {
                printf "%s,%d,%d,%d,%.2f\n", bench, ng, run, $1, $2
            }' >> "$CSV_FILE"
    done
}

OSU_COLL="/users/joglekar/osu-native/osu-micro-benchmarks-7.5/c/mpi/collective/blocking"

echo "Using OSU collective binaries from: $OSU_COLL" | tee -a "$LOG_FILE"

for n in 2 4 6 8; do
    run_collective "$n" "osu_allreduce" "${OSU_COLL}/osu_alltoall"
done