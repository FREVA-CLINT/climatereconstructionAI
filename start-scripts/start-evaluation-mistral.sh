#!/usr/bin/env bash

#SBATCH -J JohannesMeuer
#SBATCH -p gpu
#SBATCH -A bb1152
#SBATCH -n 1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --mem=256G
#SBATCH --nodelist=mg206

module load cuda/10.0.130
module load singularity/3.6.1-gcc-9.1.0
module load cdo

singularity run --bind /work/bb1152/k204233/ --nv /work/bb1152/k204233/climatereconstructionAI/torch_img_mistral.sif \
 python /work/bb1152/k204233/climatereconstructionAI/climatereconstructionAI/train_and_evaluate/evaluate.py \
 --device cuda --image-sizes 512 --pooling-layers 3 --encoding-layers 4 --data-types pr \
 --img-names radolan.h5 --mask-names single_radar_fail.h5 \
 --data-root-dir /work/bb1152/k204233/climatereconstructionAI/data/radolan-rea2/ \
 --mask-dir masks/ \
 --snapshot-dir snapshots/precipitation/radolan-simple-hole1/ckpt/200000.pth \
 --evaluation-dirs evaluation/precipitation/radolan-simple-hole1/ \
 --lstm-steps 0 \
 --prev-next-steps 0 \
 --partitions 1177 \
 --eval-names attention2 \
 --out-channels 1 \
 --infill test \
# --create-report \
# --create-images 2017-07-12-14:00,2017-07-12-14:00 \
# --create-video \