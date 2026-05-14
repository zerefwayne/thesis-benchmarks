# common.sh — shared setup for 4_osu dual-stack benchmarks
#
# Functions only — no module loads at source time. Scripts call setup_native
# FIRST (when the SLURM shell is fresh and the LUMI Lmod state is intact),
# then setup_eessi SECOND (which takes over Lmod completely via the CVMFS
# init). Reversing this order breaks LUMI/25.03 loading because EESSI
# overwrites Lmod's SitePackage and the LUMI module then can't find
# detect_LUMI_partition.
#
# HSA_ENABLE_SDMA is NOT exported here — it's a measured dimension and each
# script sets it per-cell.
#
# UCC tuning (UCC_TLS, OMPI_MCA_coll_ucc_*, etc.) is NOT here — it belongs
# in per-script env, matching 3_osu_eessi_thesis/osu_allreduce.sh:17-36.

RESULT_DIR="results"
mkdir -p "$RESULT_DIR"

# Snapshot node identity + GPU topology to a sidecar so every CSV row can
# be traced to a known node state.
node_metadata_dump() {
    local outfile="$1"
    {
        echo "# === node metadata @ $(date -Iseconds) ==="
        echo "hostname=$(hostname)"
        echo "slurm_job_id=${SLURM_JOB_ID:-NA}"
        echo "slurm_nodelist=${SLURM_JOB_NODELIST:-NA}"
        [[ -r /etc/cray/xname ]] && echo "cray_xname=$(cat /etc/cray/xname)"
        echo "kernel=$(uname -r)"
        echo "# rocm-smi topology:"
        rocm-smi --showtoponuma 2>&1 | sed 's/^/  /'
    } > "$outfile" 2>/dev/null || true
}

# Native Cray MPICH. Run this FIRST in a dual-stack job — relies on the
# fresh-login LUMI Lmod state. Mirrors 3_osu_native_thesis/common.sh.
setup_native() {
    module load LUMI/25.03
    module load PrgEnv-amd
    module load rocm
    module load craype-accel-amd-gfx90a
    export MPICH_GPU_SUPPORT_ENABLED=1

    local NATIVE_BASE="/pfs/lustrep2/users/joglekar/code/osu-native/osu-micro-benchmarks-7.5/c/mpi"
    export OSU_NATIVE_PT2PT="${NATIVE_BASE}/pt2pt/standard"
    export OSU_NATIVE_COLL="${NATIVE_BASE}/collective/blocking"
    export OSU_NATIVE_ONESIDED="${NATIVE_BASE}/one-sided"

    if [[ ! -x "$OSU_NATIVE_PT2PT/osu_bw" ]]; then
        echo "ERROR: native OSU binary missing at $OSU_NATIVE_PT2PT/osu_bw" >&2
        return 1
    fi
    echo "[setup_native] srun=$(which srun)  OSU=$OSU_NATIVE_PT2PT"
}

# EESSI rompi-2025a. Run this SECOND — the CVMFS init replaces Lmod state.
# Mirrors 3_osu_eessi_thesis/common.sh.
setup_eessi() {
    # Wipe Lmod state HARD before EESSI takes over. `module purge` does not
    # reliably unload LUMI's auto-loaded `partition/G` — its SitePackage
    # function (get_user_prefix_EasyBuild) then gets re-evaluated against
    # EESSI's replacement SitePackage and dies, leaving
    # EBROOTOSUMINMICROMINBENCHMARKS unset and OSU paths broken. Unsetting
    # LOADEDMODULES + _LMFILES_ tells Lmod "nothing is loaded".
    module --force purge 2>/dev/null || true
    unset LOADEDMODULES _LMFILES_
    unset LMOD_PACKAGE_PATH LMOD_RC LMOD_SYSTEM_DEFAULT_MODULES

    source /cvmfs/software.eessi.io/versions/2025.06/init/bash
    module load EESSI-extend/2025.06-easybuild
    module load OSU-Micro-Benchmarks/7.5-rompi-2025a

    export OSU_MPI_DIR="${EBROOTOSUMINMICROMINBENCHMARKS}/libexec/osu-micro-benchmarks/mpi"
    export OSU_PT2PT="${OSU_MPI_DIR}/pt2pt"
    export OSU_COLL="${OSU_MPI_DIR}/collective"
    export OSU_ONESIDED="${OSU_MPI_DIR}/one-sided"
    export OSU_XCCL_DIR="${EBROOTOSUMINMICROMINBENCHMARKS}/libexec/osu-micro-benchmarks/xccl/collective"

    if [[ ! -d "$OSU_MPI_DIR" ]]; then
        echo "ERROR: EESSI OSU dir missing at $OSU_MPI_DIR" >&2
        return 1
    fi
    echo "[setup_eessi] mpirun=$(which mpirun)  EBROOT=$EBROOTOSUMINMICROMINBENCHMARKS"
}
