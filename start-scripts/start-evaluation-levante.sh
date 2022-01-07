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
 --device cuda --image-sizes 512,256,256,256 --pooling-layers 3,2,2,2 --encoding-layers 4,4,4,4 --data-types pr,tas,uas,vas \
 --img-names radolan.h5,rea2-tas.h5,rea2-uas.h5,rea2-vas.h5 --mask-names single_radar_fail.h5,mask_ones_tas.h5,mask_ones_uas.h5,mask_ones_vas.h5 \
 --data-root-dir /work/bb1152/k204233/climatereconstructionAI/data/radolan-rea2/ \
 --mask-dir masks/ \
 --snapshot-dir snapshots/precipitation/radolan-rea-attention1-hole/ckpt/200000.pth \
 --evaluation-dirs evaluation/precipitation/radolan-rea-attention1-hole1/ \
 --prev-next-steps 0 \
 --partitions 1177 \
 --eval-names Rea \
 --out-channels 1 \
 --infill test \
# --create-report \
# --create-images 2143,2149 \
# --create-video
