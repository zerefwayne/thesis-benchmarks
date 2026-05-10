#!/bin/bash
#SBATCH --job-name=lumi_topo
#SBATCH --account=project_462000XXX
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --time=00:05:00
#SBATCH --output=results/topology_%j.out
#SBATCH --constraint=eessi

# ============================================================================
# LUMI standard-g topology discovery
# Prints everything you need to interpret the benchmark results that follow:
#   - GCD<->GCD xGMI link counts (rocm-smi topology matrix)
#   - CPU-to-GCD NUMA affinity
#   - HSA agent layout
#   - NIC layout (Slingshot)
# ============================================================================

source utils/common.sh

mkdir -p results

echo "######################################################################"
echo "# 1. Node hardware summary"
echo "######################################################################"
echo
echo "--- CPU info (sockets, NUMA) ---"
lscpu | grep -E "Model name|Socket|Core|NUMA|CPU\(s\)"
echo
echo "--- NUMA topology ---"
numactl --hardware
echo
echo "--- Memory ---"
free -h
echo

echo "######################################################################"
echo "# 2. GPU enumeration & xGMI topology (the key matrix for this work)"
echo "######################################################################"
echo
echo "--- rocm-smi: device list ---"
rocm-smi --showid --showproductname --showbus
echo
echo "--- rocm-smi: xGMI link weight/hops matrix ---"
echo "    (lower weight = more direct; check 'numLinks' for xGMI link count)"
rocm-smi --showtopo
echo
echo "--- rocm-smi: topology weight matrix only ---"
rocm-smi --showtopoweight
echo
echo "--- rocm-smi: topology hops matrix ---"
rocm-smi --showtopohops
echo
echo "--- rocm-smi: topology link type matrix ---"
rocm-smi --showtopotype
echo
echo "--- rocm-smi: topology numa node per GCD ---"
rocm-smi --showtoponuma
echo

echo "######################################################################"
echo "# 3. HSA agent enumeration (what UCX/MPI actually sees)"
echo "######################################################################"
echo
if command -v rocminfo >/dev/null 2>&1; then
    rocminfo | grep -E "^Agent|Name:|Marketing|Device Type|Compute Unit|Pool Info|Cache Info" | head -100
else
    echo "rocminfo not in PATH — skipping"
fi
echo

echo "######################################################################"
echo "# 4. NIC layout (Slingshot HSN)"
echo "######################################################################"
echo
echo "--- ibv_devices (if available) ---"
if command -v ibv_devices >/dev/null 2>&1; then
    ibv_devices
fi
echo
echo "--- /sys/class/net interfaces ---"
ls -la /sys/class/net/ | grep -v lo
echo
echo "--- libfabric providers (if fi_info available) ---"
if command -v fi_info >/dev/null 2>&1; then
    fi_info -l 2>/dev/null || true
fi
echo

echo "######################################################################"
echo "# 5. UCX device list (proof rocm transports are registered)"
echo "######################################################################"
echo
ucx_info -d 2>&1 | grep -E "Memory domain|Component|Transport|Device" | head -60
echo
echo "--- Build-time UCX config (definitive ROCm support check) ---"
ucx_info -b 2>&1 | grep -iE "rocm|hip|hsa|xpmem|cuda" | head -30
echo

echo "######################################################################"
echo "# 6. CPU<->GCD recommended bindings (LUMI canonical)"
echo "######################################################################"
echo
cat <<'EOF'
LUMI standard-g canonical CCD<->GCD pairing (low-noise core in each CCD):
   GCD 0 -> CCD 6 (cores 49-55)   close to GPU 0 die 0
   GCD 1 -> CCD 7 (cores 57-63)   close to GPU 0 die 1
   GCD 2 -> CCD 2 (cores 17-23)   close to GPU 1 die 0
   GCD 3 -> CCD 3 (cores 25-31)   close to GPU 1 die 1
   GCD 4 -> CCD 0 (cores 1-7)     close to GPU 2 die 0
   GCD 5 -> CCD 1 (cores 9-15)    close to GPU 2 die 1
   GCD 6 -> CCD 4 (cores 33-39)   close to GPU 3 die 0
   GCD 7 -> CCD 5 (cores 41-47)   close to GPU 3 die 1
(core 0,8,16,24,32,40,48,56 of each CCD are reserved for the OS.)

Expected xGMI structure on MI250X (4 GPUs / 8 GCDs):
- Each GCD pair (0,1) (2,3) (4,5) (6,7) shares 1 internal xGMI link (same package)
- Most inter-package GCD pairs have either 1 or 2 direct xGMI links
- A handful of pairs are reachable only via 2 hops or via PCIe
Use the 'rocm-smi --showtopo' matrix above as ground truth.
EOF
echo

echo "Done. Topology saved to results/topology_${SLURM_JOB_ID}.out"
