#!/bin/sh
#SBATCH --account=stats
#SBATCH --partition=swan
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --job-name="slob_calibration"
#SBATCH --mail-user=gntmic002@myuct.ac.za
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40

module load compilers/julia-1.5.2
export JULIA_NUM_THREADS=39
julia slob_calibration.jl > out.txt
