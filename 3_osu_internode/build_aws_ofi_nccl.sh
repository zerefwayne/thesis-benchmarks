#!/bin/bash
#SBATCH --job-name=build_aws_ofi_nccl
#SBATCH --account=project_462000226
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=00:25:00
#SBATCH --output=results/%x_%j.out
#SBATCH --constraint=eessi
#
# build_aws_ofi_nccl.sh — SCRATCH build of the aws-ofi-nccl plugin (libnccl-net.so)
# that lets RCCL use the Slingshot cxi libfabric. Validation build (not yet a
# hermetic easyconfig). Source: /users/joglekar/code/aws-ofi-nccl @ v1.19.2.
#
# Configure (autotools): --with-libfabric (hermetic cxi) --with-hwloc --with-rocm
# --disable-tests. RCCL is NOT needed at build time (plugin bundles the NCCL net
# API; RCCL dlopens libnccl-net.so at runtime via NCCL_NET_PLUGIN).

set -o pipefail
source common.sh
setup_eessi || { echo "ERROR: setup_eessi failed" >&2; exit 1; }

SRC=/users/joglekar/code/aws-ofi-nccl
PREFIX=/users/joglekar/eessi/aws-ofi-nccl-scratch
# Absolute — referenced after we cd into $SRC, so a relative path would break tee.
LOG="${SLURM_SUBMIT_DIR}/results/build_aws_ofi_nccl_${SLURM_JOB_ID}.log"

echo "=== EESSI dependency paths ===" | tee "$LOG"
echo "EBROOTLIBFABRIC=$EBROOTLIBFABRIC"   | tee -a "$LOG"
echo "EBROOTHWLOC=$EBROOTHWLOC"           | tee -a "$LOG"
echo "EBROOTROCMMINLLVM=$EBROOTROCMMINLLVM" | tee -a "$LOG"
echo "EBROOTRCCL=$EBROOTRCCL"             | tee -a "$LOG"

# Locate the HIP runtime (hip_runtime.h + libamdhip64) — try ROCm-LLVM first,
# then any EESSI software tree.
echo "=== locating HIP runtime ===" | tee -a "$LOG"
ROCM=""
for cand in "$EBROOTROCMMINLLVM" "$EBROOTROCM" "$EBROOTHIP"; do
    [[ -n "$cand" && -f "$cand/include/hip/hip_runtime.h" ]] && { ROCM="$cand"; break; }
done
if [[ -z "$ROCM" ]]; then
    hdr=$(find /cvmfs/software.eessi.io/versions/2025.06/software -maxdepth 8 -name hip_runtime.h -path '*include/hip/*' 2>/dev/null | head -1)
    [[ -n "$hdr" ]] && ROCM="${hdr%/include/hip/hip_runtime.h}"
fi
echo "ROCM(HIP)=$ROCM" | tee -a "$LOG"
echo "  hip_runtime.h: $(ls $ROCM/include/hip/hip_runtime.h 2>&1)" | tee -a "$LOG"
echo "  libamdhip64:   $(ls $ROCM/lib/libamdhip64.so* 2>&1 | head -1)" | tee -a "$LOG"
echo "  libfabric.so:  $(ls $EBROOTLIBFABRIC/lib/libfabric.so 2>&1)" | tee -a "$LOG"
[[ -z "$ROCM" || -z "$EBROOTLIBFABRIC" || -z "$EBROOTHWLOC" ]] && { echo "FATAL: missing a dependency path" | tee -a "$LOG"; exit 1; }

cd "$SRC" || exit 1
echo "=== autogen ===" | tee -a "$LOG"
./autogen.sh 2>&1 | tee -a "$LOG" || { echo "autogen FAILED" | tee -a "$LOG"; exit 1; }

echo "=== configure ===" | tee -a "$LOG"
make distclean >/dev/null 2>&1 || true
./configure --prefix="$PREFIX" \
    --with-libfabric="$EBROOTLIBFABRIC" \
    --with-hwloc="$EBROOTHWLOC" \
    --with-rocm="$ROCM" \
    --disable-tests 2>&1 | tee -a "$LOG" || { echo "configure FAILED" | tee -a "$LOG"; exit 1; }

echo "=== make ===" | tee -a "$LOG"
make -j16 2>&1 | tee -a "$LOG" || { echo "make FAILED" | tee -a "$LOG"; exit 1; }

echo "=== make install ===" | tee -a "$LOG"
make install 2>&1 | tee -a "$LOG" || { echo "install FAILED" | tee -a "$LOG"; exit 1; }

echo "=== installed plugin ===" | tee -a "$LOG"
find "$PREFIX" -name "*nccl-net*.so*" -o -name "*rccl-net*.so*" 2>/dev/null | tee -a "$LOG"
echo "BUILD COMPLETE  PREFIX=$PREFIX" | tee -a "$LOG"
