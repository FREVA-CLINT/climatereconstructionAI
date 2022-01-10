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
 --device cuda --image-sizes 128,64,64,64 --pooling-layers 3,2,2,2 --encoding-layers 4,4,4,4 --data-types pr,tas,uas,vas \
 --img-names radolan.h5,rea2-tas.h5,rea2-uas.h5,rea2-vas.h5 --mask-names single_radar_fail_128x128.h5,mask_ones_tas_64x64.h5,mask_ones_uas_64x64.h5,mask_ones_vas_64x64.h5 \
 --data-root-dir /work/bb1152/k204233/climatereconstructionAI/data/radolan-rea2-128x128/ \
 --mask-dir masks/ \
 --snapshot-dir snapshots/precipitation/128x128/2007-2011/rea-attention-hole/ckpt/500000.pth \
 --evaluation-dirs evaluation/precipitation/128x128/2007-2011/rea-attention-hole/ \
 --prev-next-steps 0 \
 --lstm-steps 0 \
 --partitions 1177 \
 --eval-names Simple,LSTM,LSTM-hole \
 --out-channels 1 \
 --create-images 2140,2160 \
 --infill test \
# --create-report \
# --create-video \
