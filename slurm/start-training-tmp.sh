#!/usr/bin/env bash

#SBATCH -J JohannesMeuer
#SBATCH -p amd
#SBATCH -A bb1152
#SBATCH -n 1
#SBATCH --cpus-per-task=256
#SBATCH --time=12:00:00
#SBATCH --mem=256G
#SBATCH --nodelist=vader3
module source setup-modules.txt
export HDF5_USE_FILE_LOCKING='FALSE'

bash copy-data.sh
##singularity run --bind /work/bb1152/k204233/climatereconstructionAI/ --nv /work/bb1152/k204233/climatereconstructionAI/torch_trial_img.sif bash test.sh
##python /work/bb1152/k204233/climatereconstructionAI/climatereconstructionAI/train.py
