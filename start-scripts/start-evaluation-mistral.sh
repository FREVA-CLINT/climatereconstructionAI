#!/usr/bin/env bash

#SBATCH -J JohannesMeuer
#SBATCH -p gpu
#SBATCH -A bb1152
#SBATCH -n 1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --mem=256G
#SBATCH --nodelist=mg201

module load cuda/10.0.130
module load singularity/3.6.1-gcc-9.1.0

singularity run --bind /work/bb1152/k204233/ --nv /work/bb1152/k204233/climatereconstructionAI/torch_img_mistral.sif \
 python /work/bb1152/k204233/climatereconstructionAI/climatereconstructionAI/evaluate.py \
 --device cuda --image-size 512 --pooling-layers 3 --encoding-layers 4 --data-type pr \
 --data-root-dir /work/bb1152/k204233/climatereconstructionAI/data/radolan-complete-scaled/ \
 --mask-dir /work/bb1152/k204233/climatereconstructionAI/climatereconstructionAI/masks/single_radar_fail.h5 \
 --snapshot-dir /work/bb1152/k204233/climatereconstructionAI/climatereconstructionAI/snapshots/precipitation/radolan-prev-next/ckpt/200000.pth \
 --evaluation-dir /work/bb1152/k204233/climatereconstructionAI/climatereconstructionAI/evaluation/precipitation/radolan-prev-next/ \
 --prev-next 0 \
 --partitions 2009 \
 --infill test \
 --create-images \
 --create-video \
 --create-report \
