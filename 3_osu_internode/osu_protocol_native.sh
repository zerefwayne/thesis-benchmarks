#!/bin/bash
#SBATCH --job-name=osu_protocol_native
#SBATCH --account=project_462000226
#SBATCH --partition=dev-g
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --cpus-per-task=7
#SBATCH --time=00:45:00
#SBATCH --output=results/%x_%j.out
#
# osu_protocol_native.sh — A5 native MPICH OFI / IPC threshold sweep.
#
# Two-dimensional sweep over the single inter-node best-case pair (GCD7-GCD7).
# MPICH on Cray uses libfabric/CXI underneath for inter-node, plus MPICH_OFI*
# knobs to influence NIC selection and IPC fallback policy.
#   MPICH_OFI_NIC_POLICY     : how to map a rank to one of the 4 NICs
#   MPICH_GPU_IPC_THRESHOLD  : eager/IPC threshold for GPU-resident buffers
# This is a coarse sweep; narrow one dimension if walltime balloons.
#
# Native-only; no --constraint=eessi.

source common.sh
source topology.sh

CSV_FILE="${RESULT_DIR}/${SLURM_JOB_NAME:-osu_protocol_native}_${SLURM_JOB_ID}.csv"
LOG_FILE="${RESULT_DIR}/${SLURM_JOB_NAME:-osu_protocol_native}_${SLURM_JOB_ID}.log"
META_FILE="${RESULT_DIR}/${SLURM_JOB_NAME:-osu_protocol_native}_${SLURM_JOB_ID}.meta"

node_metadata_dump "$META_FILE"

NODES=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
[[ ${#NODES[@]} -ge 2 ]] || { echo "ERROR: need 2 nodes" | tee -a "$LOG_FILE"; exit 1; }
NODE_A=${NODES[0]}; NODE_B=${NODES[1]}
echo "NODE_A=$NODE_A  NODE_B=$NODE_B" | tee -a "$LOG_FILE"

echo "stack,sdma_enabled,mpich_ofi_nic_policy,mpich_gpu_ipc_threshold,pair_label,node_a,node_b,gcd_a,gcd_b,hop_class,nic_class,run,size_bytes,bandwidth_MBps" \
    > "$CSV_FILE"

NUM_RUNS=4
WARMUP_RUN=1
RUN_TIMEOUT=300
OSU_FLAGS="-m 1:1048576 -i 100 -d rocm D D"

G0=7; G1=7; LABEL="inter_GCD7_GCD7"; NIC="nic_local"

WRAPPER="${RESULT_DIR}/wrap_proto_${SLURM_JOB_ID}.sh"
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

setup_native || { echo "ERROR: setup_native failed" >&2; exit 1; }
export HSA_ENABLE_SDMA=0
BIN="${OSU_NATIVE_PT2PT}/osu_bw"

NIC_POLICIES=(DEFAULT NUMA GPU)
IPC_THRESHOLDS=(DEFAULT 1024 8192 32768)

for pol in "${NIC_POLICIES[@]}"; do
    if [[ "$pol" == "DEFAULT" ]]; then unset MPICH_OFI_NIC_POLICY
    else                                  export MPICH_OFI_NIC_POLICY=$pol; fi
    for ipc in "${IPC_THRESHOLDS[@]}"; do
        if [[ "$ipc" == "DEFAULT" ]]; then unset MPICH_GPU_IPC_THRESHOLD
        else                                  export MPICH_GPU_IPC_THRESHOLD=$ipc; fi

        {
            echo
            echo "# [native MPICH_OFI_NIC_POLICY=$pol  MPICH_GPU_IPC_THRESHOLD=$ipc] $LABEL"
        } | tee -a "$LOG_FILE"

        for run in $(seq 1 $NUM_RUNS); do
            t0=$SECONDS
            tag=$( (( run == WARMUP_RUN )) && echo "warm-up" || echo "recording" )
            echo "  [run $run/$NUM_RUNS] $tag - $(date '+%H:%M:%S')" | tee -a "$LOG_FILE"
            raw=$(timeout $RUN_TIMEOUT srun --nodes=2 --ntasks=2 --ntasks-per-node=1 \
                "$WRAPPER" "$G0" "$G1" "$BIN" $OSU_FLAGS 2>>"$LOG_FILE")
            echo "  [run $run/$NUM_RUNS] done in $((SECONDS - t0))s" | tee -a "$LOG_FILE"
            echo "$raw" >> "$LOG_FILE"
            if (( run != WARMUP_RUN )); then
                echo "$raw" | awk \
                    -v pol="$pol" -v ipc="$ipc" -v lab="$LABEL" \
                    -v na="$NODE_A" -v nb="$NODE_B" \
                    -v g0="$G0" -v g1="$G1" -v nic="$NIC" -v run="$run" '
                    /^#/ { next }
                    NF == 2 && $1 ~ /^[0-9]+$/ {
                        printf "native,0,%s,%s,%s,%s,%s,%d,%d,inter_node,%s,%d,%d,%.2f\n",
                               pol, ipc, lab, na, nb, g0, g1, nic, run, $1, $2
                    }' >> "$CSV_FILE"
            fi
        done
    done
done

rm -f "$WRAPPER"
echo
echo "ALL BENCHMARKS COMPLETE  CSV=$CSV_FILE" | tee -a "$LOG_FILE"
