# common.sh — shared setup for 6_gromacs EESSI-vs-native GROMACS runs.
#
# Mirrors the 4_osu/common.sh pattern: FUNCTIONS only, no module loads
# at source time. Each benchmark script picks ONE of setup_native or
# setup_eessi based on STACK=$1.
#
# This file deliberately does NOT enable `set -u`: the EESSI CVMFS init
# (/cvmfs/software.eessi.io/.../eessi_defaults) references unbound
# EESSI_VERSION_OVERRIDE and dies under nounset.
#
# Convention reminder (CLAUDE.md): ONE stack per SLURM job. Submit twice:
#   sbatch benchmark_<system>.sh eessi
#   sbatch benchmark_<system>.sh native

RESULT_DIR="results"
mkdir -p "$RESULT_DIR"

# ----------------------------------------------------------------------
# AMD ROCm Blog "GROMACS on LUMI" recipe — env vars common to BOTH stacks.
# Source: https://rocm.blogs.amd.com/.../gromacs-lumi-guide/README.html
#
# OMP_NUM_THREADS=7 is CRITICAL — without it GROMACS sees SMT2's 14 logical
# cores per --cpus-per-task=7 and oversubscribes 2x. We saw exactly this:
# "Using 14 OpenMP threads per MPI process" in the native md.log, which
# explained the >10x performance gap vs EESSI.
# ----------------------------------------------------------------------
export OMP_NUM_THREADS=7
export GMX_ENABLE_DIRECT_GPU_COMM=1        # PP<->PME and halo via GPU peer copy
export GMX_FORCE_UPDATE_DEFAULT_GPU=1      # let mdrun default to -update gpu when supported
export ROC_ACTIVE_WAIT_TIMEOUT=0           # disable HSA's spin-wait — reclaims host CPU
export AMD_DIRECT_DISPATCH=1               # ROCm direct kernel dispatch path

# L3-cache-complex-aware CPU mask for the 8 LUMI-G ranks: each rank gets
# 7 cores (cores 1-7) of a specific L3CC, ordered to match the xGMI-near
# GCD numbering. Verbatim from the AMD ROCm Blog LUMI guide.
CPU_BIND="mask_cpu:fe000000000000,fe00000000000000,fe0000,fe000000,fe,fe00,fe00000000,fe0000000000"

# Snapshot node identity + stack identity + gmx version into a sidecar so
# every CSV can be traced to a known node and toolchain state.
node_metadata_dump() {
    local outfile="$1"
    {
        echo "# === node metadata @ $(date -Iseconds) ==="
        echo "hostname=$(hostname)"
        echo "slurm_job_id=${SLURM_JOB_ID:-NA}"
        echo "slurm_nodelist=${SLURM_JOB_NODELIST:-NA}"
        [[ -r /etc/cray/xname ]] && echo "cray_xname=$(cat /etc/cray/xname)"
        echo "kernel=$(uname -r)"
        echo "stack=${STACK_LABEL:-?}"
        echo "toolchain=${STACK_TOOLCHAIN:-?}"
        echo "gmx_mpi_bin=$(command -v gmx_mpi 2>/dev/null || echo NA)"
        echo "# rocm-smi topology:"
        rocm-smi --showtoponuma 2>&1 | sed 's/^/  /'
        echo "# gmx_mpi --version:"
        gmx_mpi --version 2>&1 | grep -E '(GROMACS version|GPU support|FFT library|SIMD|Compiler|HIP|SYCL|ROCm)' | sed 's/^/  /'
    } > "$outfile" 2>/dev/null || true
}

# Native Cray PE + AMD ROCm + AdaptiveCpp (cpeAMD-25.03) GROMACS 2025.1.
# Mirrors the 4_osu/common.sh::setup_native shape.
setup_native() {
    module load LUMI/25.03 partition/G
    module load GROMACS/2025.1-cpeAMD-25.03-VkFFT-rocm
    export MPICH_GPU_SUPPORT_ENABLED=1
    # Cray MPICH exposes GPU-aware MPI when MPICH_GPU_SUPPORT_ENABLED=1,
    # but GROMACS' auto-detection only recognises OpenMPI+UCX. Without
    # this var GROMACS stages every halo exchange through host memory
    # (D->H->MPI->H->D) — catastrophic at small system sizes. See the
    # mdrun warning "GPU-aware MPI was not detected, will not use direct
    # GPU communication". Setting both is the LUMI-recommended recipe.
    export GMX_FORCE_GPU_AWARE_MPI=1
    # Two more from the AMD ROCm Blog LUMI recipe — Cray-MPICH-specific.
    export MPICH_SMP_SINGLE_COPY_MODE=CMA   # intra-node copies via CMA (faster, no XPMEM dep)
    export MPICH_MALLOC_FALLBACK=1          # tolerate non-symmetric heap allocations
    export STACK_LABEL="native"
    export STACK_TOOLCHAIN="cpeAMD-25.03-VkFFT-rocm"

    if ! command -v gmx_mpi >/dev/null 2>&1; then
        echo "ERROR: gmx_mpi missing after setup_native" >&2
        return 1
    fi
    echo "[setup_native] srun=$(which srun)  gmx_mpi=$(which gmx_mpi)"
}

# EESSI 2025.06 + GROMACS 2025.1 rfoss-2025a-SYCL (AdaptiveCpp).
# Same hard Lmod-wipe pattern as 4_osu/common.sh::setup_eessi — `module
# purge` alone does not reliably unload LUMI's auto-loaded modules; the
# SitePackage callbacks then re-evaluate against EESSI's replacement and
# fail. Unsetting LOADEDMODULES + _LMFILES_ tells Lmod nothing is loaded.
setup_eessi() {
    module --force purge 2>/dev/null || true
    unset LOADEDMODULES _LMFILES_
    unset LMOD_PACKAGE_PATH LMOD_RC LMOD_SYSTEM_DEFAULT_MODULES

    source /cvmfs/software.eessi.io/versions/2025.06/init/bash
    module load EESSI-extend/2025.06-easybuild
    module load GROMACS/2025.1-rfoss-2025a-SYCL

    export STACK_LABEL="eessi"
    export STACK_TOOLCHAIN="rfoss-2025a-SYCL"

    if ! command -v gmx_mpi >/dev/null 2>&1; then
        echo "ERROR: gmx_mpi missing after setup_eessi" >&2
        return 1
    fi
    echo "[setup_eessi] mpirun=$(which mpirun)  gmx_mpi=$(which gmx_mpi)"
}
