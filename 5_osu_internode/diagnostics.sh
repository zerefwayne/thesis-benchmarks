#!/bin/bash
#SBATCH --job-name=diagnostics_internode
#SBATCH --account=project_462000226
#SBATCH --partition=small-g
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --time=00:15:00
#SBATCH --output=results/%x_%j.out
#SBATCH --constraint=eessi
#
# diagnostics.sh — Phase 1: fabric/transport probe on 2 nodes, ONE stack per job.
#
# Submit twice — once per stack — to get stack-tagged output:
#   sbatch diagnostics.sh native
#   sbatch diagnostics.sh eessi
#
# Each run renames its job (squeue) and writes
# results/diagnostics_internode_<stack>_<jobid>.{out,csv,log,meta}.
#
# Why split? LUMI/25.03 + EESSI/2025.06 cannot coexist in one allocation —
# EESSI's CVMFS init re-evaluates partition/G.lua against its own SitePackage
# and the LUMI-side get_user_prefix_EasyBuild global is unavailable. Confirmed
# at results/diagnostics_internode_18632905.log:404. One stack per job.

STACK="${1:-}"
case "$STACK" in
    native|eessi) ;;
    *) echo "ERROR: usage: sbatch $0 <native|eessi>  (got: '$STACK')" >&2; exit 2 ;;
esac

source common.sh

BASE_NAME="${SLURM_JOB_NAME:-diagnostics_internode}"
TAGGED_NAME="${BASE_NAME}_${STACK}"
scontrol update job=$SLURM_JOB_ID JobName="$TAGGED_NAME" 2>/dev/null || true

LOG_FILE="${RESULT_DIR}/${TAGGED_NAME}_${SLURM_JOB_ID}.log"
META_FILE="${RESULT_DIR}/${TAGGED_NAME}_${SLURM_JOB_ID}.meta"
ORIG_OUT="${RESULT_DIR}/${BASE_NAME}_${SLURM_JOB_ID}.out"
TAGGED_OUT="${RESULT_DIR}/${TAGGED_NAME}_${SLURM_JOB_ID}.out"
trap "cp -f '$ORIG_OUT' '$TAGGED_OUT' 2>/dev/null || true" EXIT

node_metadata_dump "$META_FILE"

NODES=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
if [[ ${#NODES[@]} -lt 2 ]]; then
    echo "ERROR: need 2 nodes, got ${#NODES[@]} (${NODES[*]})" | tee -a "$LOG_FILE"
    exit 1
fi
NODE_A=${NODES[0]}; NODE_B=${NODES[1]}

section() {
    {
        echo
        echo "################################################################"
        echo "# $*"
        echo "################################################################"
    } | tee -a "$LOG_FILE"
}

run_log() {
    # Run a command with a timeout, tee output to log, never fail the script.
    # NOTE: `module` is a shell function and cannot be passed through `timeout`;
    # call it directly (without run_log) and pipe to the log instead.
    local label=$1; shift
    echo "## $label" | tee -a "$LOG_FILE"
    timeout 60 "$@" 2>&1 | tee -a "$LOG_FILE" || true
    echo "## end $label (exit=$?)" >> "$LOG_FILE"
    echo >> "$LOG_FILE"
}

section "NODES (stack=$STACK)"
echo "NODE_A=$NODE_A  NODE_B=$NODE_B  STACK=$STACK" | tee -a "$LOG_FILE"
run_log "hostnames on both nodes" \
    srun --nodes=2 --ntasks=2 --ntasks-per-node=1 \
        bash -c 'echo "$(hostname): xname=$(cat /etc/cray/xname 2>/dev/null || echo NA)"'

if [[ "$STACK" == "native" ]]; then
    # ========================================================================
    # NATIVE-ONLY DIAGNOSTICS
    # ========================================================================
    section "NATIVE SECTION — Cray PE / libfabric"
    setup_native || { echo "ERROR: setup_native failed" | tee -a "$LOG_FILE"; exit 1; }

    echo "## loaded modules" | tee -a "$LOG_FILE"
    module list 2>&1 | tee -a "$LOG_FILE"
    echo "## end loaded modules" >> "$LOG_FILE"; echo >> "$LOG_FILE"

    run_log "fi_info (all providers)" srun --nodes=2 --ntasks=2 --ntasks-per-node=1 \
        bash -c 'echo "=== $(hostname) ==="; fi_info 2>&1 | head -80'

    run_log "fi_info -p cxi (Slingshot details)" srun --nodes=2 --ntasks=2 --ntasks-per-node=1 \
        bash -c 'echo "=== $(hostname) ==="; fi_info -p cxi 2>&1 | head -30'

    run_log "ip a (network interfaces)" srun --nodes=2 --ntasks=2 --ntasks-per-node=1 \
        bash -c 'echo "=== $(hostname) ==="; ip -br a 2>&1'

    run_log "Cray/SLURM env vars" \
        bash -c "env | grep -E '^(MPICH_|FI_|SLINGSHOT_|CXI_|SLURM_|CRAY_)' | sort"

    run_log "native osu_latency 2-node H H smoke test" \
        timeout 90 srun --nodes=2 --ntasks=2 --ntasks-per-node=1 \
            "${OSU_NATIVE_PT2PT}/osu_latency" -m 8:16384 H H

    run_log "native osu_latency 2-node D D (GPU-aware) smoke test" \
        timeout 120 srun --nodes=2 --ntasks=2 --ntasks-per-node=1 \
            bash -c 'export ROCR_VISIBLE_DEVICES=0; exec '"${OSU_NATIVE_PT2PT}/osu_latency"' -m 8:16384 -d rocm D D'

    run_log "native osu_bw 2-node D D smoke test (1KiB-256MiB)" \
        timeout 120 srun --nodes=2 --ntasks=2 --ntasks-per-node=1 \
            bash -c 'export ROCR_VISIBLE_DEVICES=0; exec '"${OSU_NATIVE_PT2PT}/osu_bw"' -m 1024:268435456 -d rocm D D'

else
    # ========================================================================
    # EESSI-ONLY DIAGNOSTICS
    # ========================================================================
    section "EESSI SECTION — rompi-2025a / UCX / OpenMPI"
    setup_eessi || { echo "ERROR: setup_eessi failed" | tee -a "$LOG_FILE"; exit 1; }

    echo "## loaded modules" | tee -a "$LOG_FILE"
    module list 2>&1 | tee -a "$LOG_FILE"
    echo "## end loaded modules" >> "$LOG_FILE"; echo >> "$LOG_FILE"

    run_log "ucx_info -b (build features, look for CXI / rocm)" ucx_info -b

    run_log "ucx_info -d (transports + devices visible to UCX)" ucx_info -d

    run_log "ucx_info -d (KEY DIAGNOSTIC: filtered to Transport: + Device: lines)" \
        bash -c "ucx_info -d 2>&1 | grep -E '^(#\s+)?(Transport|Device):' | sort -u"

    run_log "ucx_info -d on both nodes via mpirun" \
        timeout 60 mpirun -n 2 --map-by ppr:1:node \
            bash -c 'echo "=== $(hostname) ==="; ucx_info -d 2>&1 | grep -E "^(#\s+)?(Transport|Device):" | sort -u'

    run_log "ompi_info — UCX/OFI/CXI components" \
        bash -c "ompi_info --param all all --level 9 2>&1 | grep -iE '(pml|btl|mtl|coll)_(ucx|ofi|cxi)' | head -30"

    run_log "EESSI env vars (UCX_, OMPI_, UCC_, HSA_)" \
        bash -c "env | grep -E '^(UCX_|OMPI_|UCC_|HSA_)' | sort"

    run_log "EESSI osu_latency 2-node H H smoke test" \
        timeout 90 mpirun -n 2 --map-by ppr:1:node \
            "${OSU_PT2PT}/osu_latency" -m 8:16384 H H

    run_log "EESSI osu_latency 2-node D D smoke test (ROCR=0 on both ranks)" \
        timeout 120 mpirun -n 2 --map-by ppr:1:node \
            bash -c 'export ROCR_VISIBLE_DEVICES=0; exec '"${OSU_PT2PT}/osu_latency"' -m 8:16384 -d rocm D D'

    run_log "EESSI osu_bw 2-node D D smoke test (1KiB-16MiB)" \
        timeout 180 mpirun -n 2 --map-by ppr:1:node \
            bash -c 'export ROCR_VISIBLE_DEVICES=0; exec '"${OSU_PT2PT}/osu_bw"' -m 1024:16777216 -d rocm D D'

fi

section "DIAGNOSTICS COMPLETE (stack=$STACK)"
echo "Log:  $LOG_FILE"        | tee -a "$LOG_FILE"
echo "Meta: $META_FILE"       | tee -a "$LOG_FILE"
echo "Out:  $TAGGED_OUT (copy of $ORIG_OUT, made by EXIT trap)" | tee -a "$LOG_FILE"
