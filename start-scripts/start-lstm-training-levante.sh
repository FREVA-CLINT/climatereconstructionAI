#!/usr/bin/env bash

#SBATCH -J JohannesMeuer
#SBATCH -p amd
#SBATCH -A bb1152
#SBATCH -n 1
#SBATCH --cpus-per-task=128
#SBATCH --time=100:00:00
#SBATCH --mem=128GB
#SBATCH --nodelist=vader3

module source start-scripts/setup-modules.txt

singularity run --bind /work/bb1152/k204233/ --nv /work/bb1152/k204233/climatereconstructionAI/torch_img_levante.sif \
 python /work/bb1152/k204233/climatereconstructionAI/climatereconstructionAI/train_and_evaluate/trainLSTM.py \
 --device cuda --batch-size 4 --image-size 512 --pooling-layers 3 --encoding-layers 4 --data-type pr \
 --data-root-dir /work/bb1152/k204233/climatereconstructionAI/data/radolan-complete-scaled/ \
 --mask-dir /work/bb1152/k204233/climatereconstructionAI/climatereconstructionAI/masks/single_radar_fail.h5 \
 --snapshot-dir /work/bb1152/k204233/climatereconstructionAI/climatereconstructionAI/snapshots/precipitation/radolan-lstm/ \
 --log-dir /work/bb1152/k204233/climatereconstructionAI/climatereconstructionAI/logs/precipitation/radolan-lstm/ \
 --lstm-steps 3 \
 --max-iter 100000 \
 --save-model-interval 10000
singularity run --bind /work/bb1152/k204233/ --nv /work/bb1152/k204233/climatereconstructionAI/torch_img_levante.sif \
 python /work/bb1152/k204233/climatereconstructionAI/climatereconstructionAI/trainLSTM.py \
 --device cuda --batch-size 4 --image-size 512 --pooling-layers 3 --encoding-layers 4 --data-type pr \
 --data-root-dir /work/bb1152/k204233/climatereconstructionAI/data/radolan-complete-scaled/ \
 --mask-dir /work/bb1152/k204233/climatereconstructionAI/climatereconstructionAI/masks/single_radar_fail.h5 \
 --snapshot-dir /work/bb1152/k204233/climatereconstructionAI/climatereconstructionAI/snapshots/precipitation/radolan-lstm/ \
 --log-dir /work/bb1152/k204233/climatereconstructionAI/climatereconstructionAI/logs/precipitation/radolan-lstm/ \
 --lstm-steps 3 \
 --max-iter 200000 \
 --resume /work/bb1152/k204233/climatereconstructionAI/climatereconstructionAI/snapshots/precipitation/radolan-lstm/ckpt/100000.pth \
 --finetune \
 --save-model-interval 10000
