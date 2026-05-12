# Make results directory if doesn't exist
mkdir -p results

# 1. Clean the environment and load the Native AMD/Cray Stack
module load LUMI/25.03  # Make sure this matches the version you compiled with
module load PrgEnv-amd
module load rocm
module load craype-accel-amd-gfx90a

# 2. Network Tuning (The Cray Way)
# MPICH natively uses the FI_CXI provider for Slingshot. We don't need UCX flags here.
export MPICH_GPU_SUPPORT_ENABLED=1
