# Common environment setup for OSU benchmarks on LUMI standard-g
# Source this from every benchmark script.

# --- EESSI + module setup ---
module purge
source /cvmfs/software.eessi.io/versions/2025.06/init/bash
module load EESSI-extend/2025.06-easybuild
module load OSU-Micro-Benchmarks/7.5-rompi-2025a

# --- Override: force UCX-ROCm to win over plain UCX (Option C) ---
# Until UCX-ROCm's modulefile gets a prepend_path fix, do it here.
if [[ -n "${EBROOTUCXMINROCM:-}" ]]; then
    export PATH="${EBROOTUCXMINROCM}/bin:${PATH}"
    export LD_LIBRARY_PATH="${EBROOTUCXMINROCM}/lib:${LD_LIBRARY_PATH}"
else
    echo "WARNING: EBROOTUCXMINROCM not set; UCX-ROCm may not be active" >&2
fi

# --- ROCm / HSA settings ---
# export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export HSA_AMD_P2P=1
# Disable UCX memtype cache for clean GPU buffer detection
# export UCX_MEMTYPE_CACHE=n

# --- UCX transports: full set including ROCm ---
# rocm_copy = host<->device + intra-process device copy
# rocm_ipc  = direct GPU<->GPU peer access via xGMI
# export UCX_TLS=xpmem,rocm_copy,rocm_ipc,cma,sm,self

# --- OpenMPI: force UCX PML so we don't fall back to ob1 ---
# export OMPI_MCA_pml=ucx
# export OMPI_MCA_btl='^uct,openib'

# --- OSU benchmark paths ---
# OSU 7.5 installs benchmarks under libexec/osu-micro-benchmarks/{mpi,xccl}/...
OSU_MPI_DIR="${EBROOTOSUMINMICROMINBENCHMARKS}/libexec/osu-micro-benchmarks/mpi"
OSU_PT2PT="${OSU_MPI_DIR}/pt2pt"
OSU_COLL="${OSU_MPI_DIR}/collective"
OSU_ONESIDED="${OSU_MPI_DIR}/one-sided"
OSU_STARTUP="${OSU_MPI_DIR}/startup"

# Sanity check
if [[ ! -d "$OSU_MPI_DIR" ]]; then
    echo "ERROR: OSU MPI benchmark dir not found at $OSU_MPI_DIR" >&2
    echo "EBROOTOSUMINMICROMINBENCHMARKS=$EBROOTOSUMINMICROMINBENCHMARKS" >&2
    exit 1
fi

# --- Print active versions ---
echo "================================================================"
echo "Loaded environment:"
echo "  ucx_info:    $(which ucx_info)"
echo "  mpirun:      $(which mpirun)"
echo "  OSU prefix:  $EBROOTOSUMINMICROMINBENCHMARKS"
echo "  Hostname:    $(hostname)"
echo "  Date:        $(date)"
echo "================================================================"

# Confirm UCX-ROCm is actually active (not the plain one)
if ucx_info -b 2>&1 | grep -qi "rocm"; then
    echo "OK: ucx_info reports ROCm support compiled in"
else
    echo "WARNING: ucx_info does not report ROCm support — check PATH" >&2
fi
echo
