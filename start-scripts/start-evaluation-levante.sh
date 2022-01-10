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
 --device cuda --image-sizes 128 --pooling-layers 3 --encoding-layers 4 --data-types pr \
 --img-names radolan.h5 --mask-names single_radar_fail_128x128.h5 \
 --data-root-dir /work/bb1152/k204233/climatereconstructionAI/data/radolan-rea2-128x128/ \
 --mask-dir masks/ \
 --snapshot-dir snapshots/precipitation/128x128/2007-2011/simple-hole/ckpt/500000.pth \
 --evaluation-dirs evaluation/precipitation/128x128/2007-2011/simple-hole/ \
 --prev-next-steps 0 \
 --lstm-steps 0 \
 --partitions 1177 \
 --eval-names Simple,LSTM,LSTM-hole \
 --out-channels 1 \
 --create-images 2140,2160 \
 --infill test \
# --create-report \
# --create-video \