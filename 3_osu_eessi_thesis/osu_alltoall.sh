#!/bin/bash
#SBATCH --job-name=osu_alltoall
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

# Route collectives through UCC instead of coll/han (default priority 35) or
# coll/tuned (30). coll/ucc ships with enable=0 and priority=10 in OMPI 5.0.7.
#
# Inside UCC, restrict the transport list to ucp + self — drop rccl. The
# RCCL TL has a ~26us per-call launch overhead at small sizes and segfaults
# at N=2 (see results/osu_allreduce_diag_18631216.summary). For alltoall in
# particular, A_ucc_default (with RCCL in the mix) was ~2x slower than the
# non-UCC baseline at tiny sizes; the ucp,self-only path avoids that.
#
# Drop these once the rompi-2025a OpenMPI and UCC-ROCm modules bake the
# equivalent settings into modextravars.
export OMPI_MCA_opal_common_ucx_devices=any
export OMPI_MCA_coll_ucc_enable=1
export OMPI_MCA_coll_ucc_priority=100
export OMPI_MCA_accelerator=rocm
export UCC_TLS=ucp,self
export UCC_CL_BASIC_TLS=ucp,self

RESULT_DIR="results"
mkdir -p "$RESULT_DIR"
CSV_FILE="$RESULT_DIR/osu_alltoall_${SLURM_JOB_ID}.csv"
LOG_FILE="$RESULT_DIR/osu_alltoall_${SLURM_JOB_ID}.log"

NUM_RUNS=6
WARMUP_RUN=1

# GCD count → which GCDs to use (which devices ROCR_VISIBLE_DEVICES exposes)
# We pick GCDs that minimise topology asymmetry where possible. N=6 omitted
# on purpose: standard-g nodes are 4 packages x 2 GCDs and real workloads
# always scale by full packages (1/2/4/8 GCDs).
gcd_set() {
    case $1 in
        2) echo "0,1" ;;          # intra-package, 4-link
        4) echo "0,1,2,3" ;;      # 2 full packages
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
                mpirun -n $n "$binary_path" -d rocm >> "$LOG_FILE" 2>&1 \
                || echo "  [run $run/$NUM_RUNS] WARN: warm-up mpirun exit=$?" | tee -a "$LOG_FILE"
            continue
        fi

        echo "  [run $run/$NUM_RUNS] recording" | tee -a "$LOG_FILE"
        local raw rc
        raw=$(ROCR_VISIBLE_DEVICES=$devices \
              mpirun -n $n "$binary_path" -d rocm 2>>"$LOG_FILE")
        rc=$?
        if (( rc != 0 )); then
            echo "  [run $run/$NUM_RUNS] ERROR: mpirun exit=$rc (see $LOG_FILE)" | tee -a "$LOG_FILE"
        fi

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

for n in 2 4 8; do
    run_collective "$n" "osu_alltoall"  "${OSU_COLL}/osu_alltoall"
    # run_collective "$n" "osu_allreduce" "${OSU_COLL}/osu_allreduce"
done

echo
echo "================================================================"
echo "ALL BENCHMARKS COMPLETE"
echo "  CSV: $CSV_FILE"
echo "  Log: $LOG_FILE"
echo "================================================================"