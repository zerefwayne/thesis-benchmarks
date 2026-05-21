# common.sh — shared setup for 5_osu_internode dual-stack 2-node benchmarks
#
# Functions only — no module loads at source time. Scripts call setup_native
# FIRST (when the SLURM shell is fresh and the LUMI Lmod state is intact),
# then setup_eessi SECOND (which takes over Lmod completely via the CVMFS
# init). Reversing this order breaks LUMI/25.03 loading because EESSI
# overwrites Lmod's SitePackage and the LUMI module then can't find
# detect_LUMI_partition.
#
# Independent copy of 4_osu/common.sh — kept verbatim except where noted so
# the two benchmark phases can evolve separately.
#
# HSA_ENABLE_SDMA is NOT exported here — it's a measured dimension and each
# script sets it per-cell.

RESULT_DIR="results"
mkdir -p "$RESULT_DIR"

# Snapshot node identity + GPU topology to a sidecar so every CSV row can
# be traced to a known node state. For inter-node jobs the sidecar covers
# only the script-launcher node; per-node detail goes into the .log via
# diagnostics.sh or the explicit srun --nodes=N hostname dumps in scripts.
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
# fresh-login LUMI Lmod state.
setup_native() {
    module load LUMI/25.03
    module load PrgEnv-amd
    module load rocm
    module load craype-accel-amd-gfx90a
    export MPICH_GPU_SUPPORT_ENABLED=1

    local NATIVE_BASE="/pfs/lustrep2/users/joglekar/code/osu-native/osu-micro-benchmarks-7.5/c/mpi"
    export OSU_NATIVE_PT2PT="${NATIVE_BASE}/pt2pt/standard"
    export OSU_NATIVE_COLL="${NATIVE_BASE}/collective/blocking"
    export OSU_NATIVE_ONESIDED="${NATIVE_BASE}/one-sided/standard"

    if [[ ! -x "$OSU_NATIVE_PT2PT/osu_bw" ]]; then
        echo "ERROR: native OSU binary missing at $OSU_NATIVE_PT2PT/osu_bw" >&2
        return 1
    fi
    echo "[setup_native] srun=$(which srun)  OSU=$OSU_NATIVE_PT2PT"
}

# EESSI rompi-2025a. Run this SECOND — the CVMFS init replaces Lmod state.
setup_eessi() {
    # Critical: purge LUMI's partition/G (auto-loaded by LUMI/25.03 in
    # setup_native) before EESSI takes over Lmod. Without this, the next
    # `module load` re-evaluates partition/G.lua against EESSI's SitePackage
    # which lacks get_user_prefix_EasyBuild -> EBROOTOSUMINMICROMINBENCHMARKS
    # ends up empty and OSU paths break.
    module purge --force 2>/dev/null || true

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

    # --- Slingshot CXI selection -----------------------------------------
    # The rebuilt libfabric carries the `cxi` provider, but OpenMPI 5.0.7
    # defaults to the UCX PML, and UCX has no CXI transport -> inter-node
    # falls back to TCP (~2 GB/s vs native ~24 GB/s). Force the OFI path so
    # OpenMPI actually uses the cxi provider.
    #
    # NOTE: this also routes intra-node traffic through the NIC (cxi loopback)
    # instead of UCX/xGMI, so the intra-node reference cell will read lower
    # than a UCX run — inter-node is the measurement that matters here.
    # Comment this block out to fall back to the stock UCX/TCP behaviour.
    export OMPI_MCA_pml=cm
    export OMPI_MCA_mtl=ofi
    export OMPI_MCA_opal_common_ofi_provider_include=cxi
    export FI_PROVIDER=cxi
    export FI_CXI_RX_MATCH_MODE=hybrid          # avoid HW match-list exhaustion
    export FI_CXI_DEFAULT_CQ_SIZE=131072        # avoid Cassini EQ overflow
    export CXI_FORK_SAFE=1
    # ranks on the allocation head node otherwise fail to get a CXI service
    export PRTE_MCA_ras_base_launch_orted_on_hn=1

    echo "[setup_eessi] mpirun=$(which mpirun)  EBROOT=$EBROOTOSUMINMICROMINBENCHMARKS"
    echo "[setup_eessi] FI_PROVIDER=$FI_PROVIDER  OMPI_MCA_pml=$OMPI_MCA_pml/$OMPI_MCA_mtl"
}
