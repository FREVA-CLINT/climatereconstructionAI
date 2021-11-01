#!/usr/bin/env bash

#SBATCH -J JohannesMeuer
#SBATCH -p amd
#SBATCH -A bb1152
#SBATCH -n 1
#SBATCH --cpus-per-task=128
#SBATCH --time=24:00:00
#SBATCH --mem=256G
#SBATCH --nodelist=vader2
module source slurm/setup-modules.txt
export HDF5_USE_FILE_LOCKING='FALSE'

singularity run --bind /work/bb1152/k204233/ --nv /work/bb1152/k204233/climatereconstructionAI/torch_trial_img.sif python /work/bb1152/k204233/climatereconstructionAI/climatereconstructionAI/train.py
