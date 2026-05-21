#!/bin/bash
#SBATCH --job-name=osu_protocol_eessi
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
# osu_protocol_eessi.sh — A5 CXI rendezvous threshold sweep (EESSI only).
#
# Inter-node OSU on EESSI goes through the OFI MTL → libfabric CXI provider.
# The CXI provider has its own eager/rdzv threshold (FI_CXI_RDZV_THRESHOLD)
# which mirrors UCX_RNDV_THRESH's role on the UCX path. Sweep the threshold
# over a single inter-node pair (GCD7-GCD7, NIC-adjacent best case) and look
# for an analog of 4_osu's "256 B cliff" — i.e. is there a small-message zone
# where DEFAULT picks eager too late (or rdzv too early)?
#
# Single inter-node pair, 4 runs per threshold (1 warmup + 3 recorded).

source common.sh
source topology.sh

CSV_FILE="${RESULT_DIR}/${SLURM_JOB_NAME:-osu_protocol_eessi}_${SLURM_JOB_ID}.csv"
LOG_FILE="${RESULT_DIR}/${SLURM_JOB_NAME:-osu_protocol_eessi}_${SLURM_JOB_ID}.log"
META_FILE="${RESULT_DIR}/${SLURM_JOB_NAME:-osu_protocol_eessi}_${SLURM_JOB_ID}.meta"

node_metadata_dump "$META_FILE"

NODES=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
[[ ${#NODES[@]} -ge 2 ]] || { echo "ERROR: need 2 nodes" | tee -a "$LOG_FILE"; exit 1; }
NODE_A=${NODES[0]}; NODE_B=${NODES[1]}
echo "NODE_A=$NODE_A  NODE_B=$NODE_B" | tee -a "$LOG_FILE"

echo "stack,sdma_enabled,fi_cxi_rdzv_threshold,pair_label,node_a,node_b,gcd_a,gcd_b,hop_class,nic_class,run,size_bytes,bandwidth_MBps" \
    > "$CSV_FILE"

NUM_RUNS=4   # 1 warmup + 3 recorded
WARMUP_RUN=1
RUN_TIMEOUT=300
OSU_FLAGS="-m 1:1048576 -i 100 -d rocm D D"

# Single pair — NIC-adjacent best case.
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

setup_eessi || { echo "ERROR: setup_eessi failed" >&2; exit 1; }
export HSA_ENABLE_SDMA=0

# Powers of 2 bracketing the typical CXI eager/rdzv transition zone. DEFAULT
# is the libfabric/CXI default (unset). 4_osu showed 16 MiB caps eager at the
# bounce-buffer ceiling — not retesting that catastrophic value here.
for rdzv in DEFAULT 1024 4096 8192 16384 65536 262144; do
    if [[ "$rdzv" == "DEFAULT" ]]; then
        unset FI_CXI_RDZV_THRESHOLD
    else
        export FI_CXI_RDZV_THRESHOLD=$rdzv
    fi
    {
        echo
        echo "# [eessi sdma=0 FI_CXI_RDZV_THRESHOLD=$rdzv] $LABEL"
    } | tee -a "$LOG_FILE"

    for run in $(seq 1 $NUM_RUNS); do
        t0=$SECONDS
        tag=$( (( run == WARMUP_RUN )) && echo "warm-up" || echo "recording" )
        echo "  [run $run/$NUM_RUNS] $tag - $(date '+%H:%M:%S')" | tee -a "$LOG_FILE"
        raw=$(timeout $RUN_TIMEOUT mpirun -n 2 --host "$NODE_A:1,$NODE_B:1" \
            --map-by ppr:1:node \
            "$WRAPPER" "$G0" "$G1" "${OSU_PT2PT}/osu_bw" $OSU_FLAGS 2>>"$LOG_FILE")
        echo "  [run $run/$NUM_RUNS] done in $((SECONDS - t0))s" | tee -a "$LOG_FILE"
        echo "$raw" >> "$LOG_FILE"
        if (( run != WARMUP_RUN )); then
            echo "$raw" | awk \
                -v rdzv="$rdzv" -v lab="$LABEL" -v na="$NODE_A" -v nb="$NODE_B" \
                -v g0="$G0" -v g1="$G1" -v nic="$NIC" -v run="$run" '
                /^#/ { next }
                NF == 2 && $1 ~ /^[0-9]+$/ {
                    printf "eessi,0,%s,%s,%s,%s,%d,%d,inter_node,%s,%d,%d,%.2f\n",
                           rdzv, lab, na, nb, g0, g1, nic, run, $1, $2
                }' >> "$CSV_FILE"
        fi
    done
done

rm -f "$WRAPPER"
echo
echo "ALL BENCHMARKS COMPLETE  CSV=$CSV_FILE" | tee -a "$LOG_FILE"
