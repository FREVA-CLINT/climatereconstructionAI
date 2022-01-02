#!/usr/bin/env bash

#SBATCH -J JohannesMeuer
#SBATCH -p amd
#SBATCH -A bb1152
#SBATCH -n 1
#SBATCH --cpus-per-task=128
#SBATCH --time=12:00:00
#SBATCH --mem=256G
#SBATCH --nodelist=vader2

module source start-scripts/setup-modules.txt

singularity run --bind /work/bb1152/k204233/ --nv /work/bb1152/k204233/climatereconstructionAI/torch_img_levante.sif \
 python /work/bb1152/k204233/climatereconstructionAI/climatereconstructionAI/train_and_evaluate/evaluate.py \
 --device cuda --image-sizes 512 --pooling-layers 3 --encoding-layers 4 --data-types pr \
 --img-names radolan.h5 --mask-names single_radar_fail.h5 \
 --data-root-dir /work/bb1152/k204233/climatereconstructionAI/data/radolan-rea2/ \
 --mask-dir masks/ \
 --snapshot-dir snapshots/precipitation/radolan-prev-next-lstm/ckpt/200000.pth \
 --evaluation-dirs evaluation/precipitation/radolan-prev-next-lstm/ \
 --lstm-steps 2 \
 --prev-next-steps 0 \
 --partitions 1177 \
 --eval-names LSTM-pn \
 --out-channels 1 \
 --create-report \
 --infill test \
# --create-images 2017-07-12-14:00,2017-07-12-14:00 \
# --create-video \