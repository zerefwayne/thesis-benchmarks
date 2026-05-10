#!/bin/bash
#SBATCH --job-name=gromacs_hegfr_dimer
#SBATCH --account=project_462000226
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --exclusive
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --time=00:45:00
#SBATCH --output=results/%x_%j.out
#SBATCH --constraint=eessi

source common.sh
module load GROMACS/2025.1-rfoss-2025a-SYCL

cd ./GROMACS_Benchmark_Suite/HECBioSim/hEGFRDimer

mpirun -np 8 --oversubscribe gmx_mpi mdrun \
    -s benchmark.tpr \
    -nb gpu -pme gpu -bonded gpu \
    -nsteps 100000 -resetstep 20000 -noconfout -npme 1