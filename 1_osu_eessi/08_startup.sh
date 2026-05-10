#!/bin/bash
#SBATCH --job-name=osu_startup
#SBATCH --account=project_462000XXX
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --time=00:15:00
#SBATCH --output=results/startup_%j.out
#SBATCH --constraint=eessi

# ============================================================================
# Startup / init overhead.
# Useful to compare GPU-aware vs non-GPU-aware MPI init costs.
# ============================================================================

source utils/common.sh

mkdir -p results/startup

SELECT_CPU="49-55,57-63,17-23,25-31,1-7,9-15,33-39,41-47"

cat > /tmp/wrap_${SLURM_JOB_ID}.sh <<'EOF'
#!/bin/bash
GCDS=(0 1 2 3 4 5 6 7)
export ROCR_VISIBLE_DEVICES=${GCDS[$OMPI_COMM_WORLD_LOCAL_RANK]}
exec "$@"
EOF
chmod +x /tmp/wrap_${SLURM_JOB_ID}.sh

echo "######################################################################"
echo "# osu_init  (MPI_Init time)"
echo "######################################################################"
mpirun -np 8 \
    --bind-to cpulist:ordered \
    --cpu-set $SELECT_CPU \
    /tmp/wrap_${SLURM_JOB_ID}.sh \
    ${OSU_STARTUP}/osu_init
echo

echo "######################################################################"
echo "# osu_hello  (init+finalize time)"
echo "######################################################################"
mpirun -np 8 \
    --bind-to cpulist:ordered \
    --cpu-set $SELECT_CPU \
    /tmp/wrap_${SLURM_JOB_ID}.sh \
    ${OSU_STARTUP}/osu_hello
echo

# Scaling: 2, 4, 8 ranks
for np in 2 4 8; do
    echo "######################################################################"
    echo "# osu_init at np=$np"
    echo "######################################################################"
    mpirun -np $np \
        /tmp/wrap_${SLURM_JOB_ID}.sh \
        ${OSU_STARTUP}/osu_init
    echo
done

rm -f /tmp/wrap_${SLURM_JOB_ID}.sh
echo "Done."
