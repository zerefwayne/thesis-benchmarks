#!/bin/bash
#SBATCH --job-name=osu_protocol_internode_eessi
#SBATCH --account=project_462000226
#SBATCH --partition=dev-g
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --cpus-per-task=7
#SBATCH --time=01:00:00
#SBATCH --output=results/%x_%j.out
#SBATCH --constraint=eessi
#
# osu_protocol_internode_eessi.sh — Phase 3: sweep EESSI UCX/OMPI/libfabric
# knobs on a single inter-node pair (NODE_A:GCD7, NODE_B:GCD7), SDMA=0,
# 1 warm-up + 3 recorded runs per config.
#
# Goal: find a knob combination that brings EESSI's inter-node bandwidth
# closer to native Cray MPICH (~24 GB/s @ 16 MiB per diagnostics 18633652).
#
# Phase 1 (18633651) revealed:
#   - UCX has NO cxi transport (only tcp can reach the other node)
#   - rocm_copy_ep.c SIGPOOL exhaustion at 8 MiB+, degrades BW from 4.5 to 3.6 GB/s
#   - ompi_info shows BTL OFI is compiled in; CXI not in btl_ofi_provider_exclude
# These observations drive the 6 configs below.

source common.sh
source topology.sh

CSV_FILE="${RESULT_DIR}/${SLURM_JOB_NAME:-osu_protocol_internode_eessi}_${SLURM_JOB_ID}.csv"
LOG_FILE="${RESULT_DIR}/${SLURM_JOB_NAME:-osu_protocol_internode_eessi}_${SLURM_JOB_ID}.log"
META_FILE="${RESULT_DIR}/${SLURM_JOB_NAME:-osu_protocol_internode_eessi}_${SLURM_JOB_ID}.meta"

node_metadata_dump "$META_FILE"

NODES=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
if [[ ${#NODES[@]} -lt 2 ]]; then
    echo "ERROR: need 2 nodes, got ${#NODES[@]} (${NODES[*]})" | tee -a "$LOG_FILE"
    exit 1
fi
NODE_A=${NODES[0]}; NODE_B=${NODES[1]}
echo "NODE_A=$NODE_A  NODE_B=$NODE_B" | tee -a "$LOG_FILE"

echo "stack,sdma_enabled,config_label,ucx_tls,ompi_mca_pml,ompi_mca_btl,ompi_mca_mtl,ucx_net_devices,fi_provider,extra_env,pair_label,gcd_a,gcd_b,run,size_bytes,bandwidth_MBps" \
    > "$CSV_FILE"

NUM_RUNS=4   # 1 warm-up + 3 recorded
WARMUP_RUN=1
RUN_TIMEOUT=300
OSU_FLAGS="-m 8:268435456 -i 100 -d rocm D D"

# Fixed pair for the sweep: NIC-adjacent GCD7-GCD7 (best case)
GA=7; GB=7
LABEL="inter_node_GCD7_GCD7"

WRAPPER="${RESULT_DIR}/wrap_proto_${SLURM_JOB_ID}.sh"
cat > "$WRAPPER" <<'EOF'
#!/bin/bash
GCD=$1; shift
export ROCR_VISIBLE_DEVICES=$GCD
exec "$@"
EOF
chmod +x "$WRAPPER"

setup_eessi || { echo "ERROR: setup_eessi failed" | tee -a "$LOG_FILE"; exit 1; }
export HSA_ENABLE_SDMA=0

clear_knobs() {
    unset UCX_TLS OMPI_MCA_pml OMPI_MCA_btl OMPI_MCA_mtl UCX_NET_DEVICES FI_PROVIDER \
          UCX_ROCM_COPY_SIGPOOL_MAX_ELEMS UCX_TCP_BRIDGE_ENABLE
}

# Apply a config, run osu_bw N times, write CSV rows with all knob values.
# extra_env is a single "NAME=VALUE" string or "unset"; lets us carry one-off
# knobs (sigpool size, tcp bridge enable) without growing the arg list further.
sweep_config() {
    local label=$1 ucx_tls=$2 pml=$3 btl=$4 mtl=$5 net_dev=$6 fi_prov=$7 extra=$8

    clear_knobs
    [[ "$ucx_tls"  != "unset" ]] && export UCX_TLS="$ucx_tls"
    [[ "$pml"      != "unset" ]] && export OMPI_MCA_pml="$pml"
    [[ "$btl"      != "unset" ]] && export OMPI_MCA_btl="$btl"
    [[ "$mtl"      != "unset" ]] && export OMPI_MCA_mtl="$mtl"
    [[ "$net_dev"  != "unset" ]] && export UCX_NET_DEVICES="$net_dev"
    [[ "$fi_prov"  != "unset" ]] && export FI_PROVIDER="$fi_prov"
    if [[ "$extra" != "unset" ]]; then
        # "NAME=VALUE" — export it. Trust the static config table (no user input).
        export "${extra?}"
    fi

    {
        echo
        echo "################################################################"
        echo "# config=$label"
        echo "#   UCX_TLS=$ucx_tls  OMPI_MCA_pml=$pml  OMPI_MCA_btl=$btl  OMPI_MCA_mtl=$mtl"
        echo "#   UCX_NET_DEVICES=$net_dev  FI_PROVIDER=$fi_prov  EXTRA=$extra"
        echo "################################################################"
    } | tee -a "$LOG_FILE"

    for run in $(seq 1 $NUM_RUNS); do
        local t0=$SECONDS raw
        local tag=$( (( run == WARMUP_RUN )) && echo "warm-up" || echo "recording" )
        echo "  [run $run/$NUM_RUNS] $tag - $(date '+%H:%M:%S')" | tee -a "$LOG_FILE"

        raw=$(timeout $RUN_TIMEOUT mpirun -n 2 --host "$NODE_A:1,$NODE_B:1" \
            --map-by ppr:1:node "$WRAPPER" "$GA" "${OSU_PT2PT}/osu_bw" $OSU_FLAGS \
            2>>"$LOG_FILE")
        # rank 0 wrapper picks GCD=$GA; rank 1 also gets $GA which equals $GB=7 here.

        echo "  [run $run/$NUM_RUNS] done in $((SECONDS - t0))s" | tee -a "$LOG_FILE"
        echo "$raw" >> "$LOG_FILE"
        if (( run != WARMUP_RUN )); then
            echo "$raw" | awk \
                -v st="eessi" -v sdma=0 -v lab="$label" \
                -v tls="$ucx_tls" -v pml="$pml" -v btl="$btl" -v mtl="$mtl" \
                -v nd="$net_dev" -v fp="$fi_prov" -v ex="$extra" \
                -v plab="$LABEL" -v ga="$GA" -v gb="$GB" -v run="$run" '
                /^#/ { next }
                NF == 2 && $1 ~ /^[0-9]+$/ {
                    printf "%s,%d,%s,%s,%s,%s,%s,%s,%s,%s,%s,%d,%d,%d,%d,%.2f\n",
                           st, sdma, lab, tls, pml, btl, mtl, nd, fp, ex,
                           plab, ga, gb, run, $1, $2
                }' >> "$CSV_FILE"
        fi
    done
}

# Config sweep — 6 knob combinations on the same inter-node pair.
# Args: label, UCX_TLS, OMPI_MCA_pml, OMPI_MCA_btl, OMPI_MCA_mtl, UCX_NET_DEVICES, FI_PROVIDER, extra_env
sweep_config "baseline"      "unset"                  "unset" "unset"          "unset" "unset" "unset" "unset"
sweep_config "sigpool_fix"   "unset"                  "unset" "unset"          "unset" "unset" "unset" "UCX_ROCM_COPY_SIGPOOL_MAX_ELEMS=65536"
sweep_config "ucx_no_rocm"   "^rocm_ipc,^rocm_copy"   "ucx"   "unset"          "unset" "unset" "unset" "unset"
sweep_config "ucx_net_hsn0"  "unset"                  "ucx"   "unset"          "unset" "hsn0"  "unset" "UCX_TCP_BRIDGE_ENABLE=n"
sweep_config "ompi_ofi_btl"  "unset"                  "ob1"   "ofi,self,vader" "unset" "unset" "cxi"   "unset"
sweep_config "ompi_ofi_mtl"  "unset"                  "cm"    "unset"          "ofi"   "unset" "cxi"   "unset"

rm -f "$WRAPPER"
echo
echo "ALL BENCHMARKS COMPLETE  CSV=$CSV_FILE" | tee -a "$LOG_FILE"
