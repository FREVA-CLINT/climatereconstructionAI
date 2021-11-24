#!/usr/bin/env bash

#SBATCH -J JohannesMeuer
#SBATCH -p gpu
#SBATCH -A bb1152
#SBATCH -n 1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --nodelist=mg206

module load cuda/10.0.130
module load singularity/3.6.1-gcc-9.1.0

singularity run --bind /work/bb1152/k204233/ --nv /work/bb1152/k204233/climatereconstructionAI/torch_img_mistral.sif \
 python /work/bb1152/k204233/climatereconstructionAI/climatereconstructionAI/train_and_evaluate/trainLSTM.py \
 --device cuda --batch-size 2 --image-size 72 --pooling-layers 0 --encoding-layers 3 --data-type tas \
 --data-root-dir /work/bb1152/k204233/climatereconstructionAI/data/20cr/ \
 --mask-dir /work/bb1152/k204233/climatereconstructionAI/climatereconstructionAI/masks/hadcrut4-missmask.h5 \
 --snapshot-dir /work/bb1152/k204233/climatereconstructionAI/climatereconstructionAI/snapshots/temperature/20cr-lstm/ \
 --log-dir /work/bb1152/k204233/climatereconstructionAI/climatereconstructionAI/logs/temperature/20cr-lstm// \
 --lstm-steps 0 \
 --max-iter 1000000 \
 --save-model-interval 10000