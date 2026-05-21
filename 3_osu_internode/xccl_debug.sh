#!/bin/bash
#SBATCH --job-name=xccl_debug
#SBATCH --account=project_462000226
#SBATCH --partition=dev-g
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --cpus-per-task=7
#SBATCH --time=00:15:00
#SBATCH --output=results/%x_%j.out
#SBATCH --constraint=eessi
#
# xccl_debug.sh <gdr_level> — fast RCCL bring-up probe. Runs ONLY osu_xccl_allreduce
# at small sizes with a SHORT timeout, so a hang shows up in seconds, not 600s.
# $1 = NCCL_NET_GDR_LEVEL (e.g. 0 to disable GPUDirect RDMA / force host staging, 3=PHB).

set -o pipefail
GDR="${1:-3}"
NODMABUF="${2:-0}"   # $2=1 -> OFI_NCCL_DISABLE_DMABUF=1 (legacy GDR registration)
NOFLUSH="${3:-0}"    # $3=1 -> OFI_NCCL_GDR_FLUSH_DISABLE=1 (skip post-GDR-recv flush)
source common.sh
setup_eessi || { echo "ERROR setup_eessi" >&2; exit 1; }
[[ "$NODMABUF" == "1" ]] && export OFI_NCCL_DISABLE_DMABUF=1
[[ "$NOFLUSH" == "1" ]] && export OFI_NCCL_GDR_FLUSH_DISABLE=1
# RCCL core disables dmabuf export by default ("Dmabuf feature disabled without
# NCCL_DMABUF_ENABLE=1"), forcing the legacy peer-direct GDR path that hangs on
# cxi. libfabric advertises "DMA-BUF registrations: true", so enable the dmabuf
# path end-to-end. $4=0 to turn it back off.
NCCL_DMABUF="${4:-1}"
[[ "$NCCL_DMABUF" == "1" ]] && export NCCL_DMABUF_ENABLE=1
# $5=1 -> FI_CXI_DISABLE_DMABUF_ROCR=1: cxi provider registers GPU memory via the
# direct cxi MAP/ATS path instead of dmabuf (the default dmabuf path hangs/EINVALs).
NOCXIDMABUF="${5:-0}"
[[ "$NOCXIDMABUF" == "1" ]] && export FI_CXI_DISABLE_DMABUF_ROCR=1
# $6=1 -> force LUMI's HOST (Cray) libfabric + libcxi (the native stack that does
# GPUDirect over cxi) instead of our from-source ones, via LD_PRELOAD. Decisive
# test of "is GPUDirect broken in OUR libs, or unavailable on the node?".
HOSTLF="${6:-0}"
# Don't export globally (it breaks the script's own coreutils, which would then
# load host libcxi -> libnl-3.so.200). Bake it into the per-rank wrapper instead.
HOST_PRELOAD=""
[[ "$HOSTLF" == "1" ]] && HOST_PRELOAD="/opt/cray/libfabric/1.22.0/lib64/libfabric.so.1:/usr/lib64/libcxi.so.1"
echo "HOST_PRELOAD='${HOST_PRELOAD}'"

AWS=/users/joglekar/eessi/aws-ofi-nccl-scratch/lib
export LD_LIBRARY_PATH="${AWS}:${LD_LIBRARY_PATH}"
export NCCL_NET_PLUGIN="${AWS}/librccl-net.so"
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export FI_PROVIDER=cxi
export NCCL_NET_GDR_LEVEL="$GDR"
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH
export HSA_ENABLE_SDMA=0

NODES=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
LOG=results/xccl_debug_gdr${GDR}_${SLURM_JOB_ID}.log
echo "GDR_LEVEL=$GDR  NODES=${NODES[*]}" | tee "$LOG"

W=results/wrap_dbg_${SLURM_JOB_ID}.sh
{
    echo '#!/bin/bash'
    echo 'export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7'
    [[ -n "$HOST_PRELOAD" ]] && echo "export LD_PRELOAD=\"${HOST_PRELOAD}\${LD_PRELOAD:+:\$LD_PRELOAD}\""
    echo 'exec "$@"'
} > "$W"
chmod +x "$W"

echo "=== osu_xccl_allreduce -m 1:8192, 90s timeout ===" | tee -a "$LOG"
timeout 90 mpirun -n 16 --map-by ppr:8:node "$W" \
    "${OSU_XCCL_DIR}/osu_xccl_allreduce" -m 1:8192 2>&1 | tee -a "$LOG"
rc=$?
echo "exit_code=$rc (124=timeout/hang)" | tee -a "$LOG"
rm -f "$W"
