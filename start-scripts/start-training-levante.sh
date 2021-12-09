#!/usr/bin/env bash

#SBATCH -J JohannesMeuer
#SBATCH -p amd
#SBATCH -A bb1152
#SBATCH -n 1
#SBATCH --cpus-per-task=128
#SBATCH --time=100:00:00
#SBATCH --mem=128GB
#SBATCH --nodelist=vader2

module source start-scripts/setup-modules.txt

singularity run --bind /work/bb1152/k204233/ --nv /work/bb1152/k204233/climatereconstructionAI/torch_img_levante.sif \
 python /work/bb1152/k204233/climatereconstructionAI/climatereconstructionAI/train_and_evaluate/train.py \
 --device cuda --batch-size 4 --image-size 512 --pooling-layers 3 --encoding-layers 4 --data-types pr \
 --data-root-dir /work/bb1152/k204233/climatereconstructionAI/data/radolan-rea2/ \
 --mask-dir /work/bb1152/k204233/climatereconstructionAI/climatereconstructionAI/masks/ \
 --img-names radolan.h5 --mask-names single_radar_fail.h5 \
 --snapshot-dir /work/bb1152/k204233/climatereconstructionAI/climatereconstructionAI/snapshots/precipitation/radolan-lstm-2007-2013/ \
 --log-dir /work/bb1152/k204233/climatereconstructionAI/climatereconstructionAI/logs/precipitation/radolan-lstm-2007-2013/ \
 --lstm-steps 3 \
 --resume /work/bb1152/k204233/climatereconstructionAI/climatereconstructionAI/snapshots/precipitation/radolan-lstm-2007-2013/ckpt/50000.pth \
 --max-iter 100000 \
 --save-model-interval 10000
singularity run --bind /work/bb1152/k204233/ --nv /work/bb1152/k204233/climatereconstructionAI/torch_img_levante.sif \
 python /work/bb1152/k204233/climatereconstructionAI/climatereconstructionAI/train_and_evaluate/train.py \
 --device cuda --batch-size 4 --image-size 512 --pooling-layers 3 --encoding-layers 4 --data-types pr \
 --data-root-dir /work/bb1152/k204233/climatereconstructionAI/data/radolan-rea2/ \
 --mask-dir /work/bb1152/k204233/climatereconstructionAI/climatereconstructionAI/masks/ \
 --img-names radolan.h5 --mask-names single_radar_fail.h5 \
 --snapshot-dir /work/bb1152/k204233/climatereconstructionAI/climatereconstructionAI/snapshots/precipitation/radolan-lstm-2007-2013/ \
 --log-dir /work/bb1152/k204233/climatereconstructionAI/climatereconstructionAI/logs/precipitation/radolan-lstm-2007-2013/ \
 --lstm-steps 3 \
 --max-iter 200000 \
 --resume /work/bb1152/k204233/climatereconstructionAI/climatereconstructionAI/snapshots/precipitation/radolan-lstm-2007-2013/ckpt/100000.pth \
 --finetune \
 --save-model-interval 10000
