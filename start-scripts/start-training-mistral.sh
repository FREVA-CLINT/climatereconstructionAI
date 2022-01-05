#!/usr/bin/env bash

#SBATCH -J JohannesMeuer
#SBATCH -p gpu
#SBATCH -A bb1152
#SBATCH -n 1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH --nodelist=mg206

module load cuda/10.0.130
module load singularity/3.6.1-gcc-9.1.0

singularity run --bind /work/bb1152/k204233/ --nv /work/bb1152/k204233/climatereconstructionAI/torch_img_mistral.sif \
 python /work/bb1152/k204233/climatereconstructionAI/climatereconstructionAI/train_and_evaluate/train.py \
 --device cuda --batch-size 4 --image-sizes 512,256,256,256 --pooling-layers 3,2,2,2 --encoding-layers 4,4,4,4 --data-types pr,tas,uas,vas \
 --data-root-dir /work/bb1152/k204233/climatereconstructionAI/data/radolan-rea2/ \
 --mask-dir /work/bb1152/k204233/climatereconstructionAI/climatereconstructionAI/masks/ \
 --img-names radolan.h5,rea2-tas.h5,rea2-uas.h5,rea2-vas.h5 --mask-names single_radar_fail.h5,mask_ones_tas.h5,mask_ones_uas.h5,mask_ones_vas.h5 \
 --snapshot-dir /work/bb1152/k204233/climatereconstructionAI/climatereconstructionAI/snapshots/precipitation/radolan-rea-attention/ \
 --log-dir /work/bb1152/k204233/climatereconstructionAI/climatereconstructionAI/logs/precipitation/radolan-rea-attention/ \
 --out-channels 1 \
 --lstm-steps 0 \
 --prev-next-steps 0 \
 --max-iter 100000 \
 --eval-timesteps 2143,2144,2145,2146,2147 \
 --save-model-interval 5000 \
 --log-interval 100 \
 --loss-criterion 1
singularity run --bind /work/bb1152/k204233/ --nv /work/bb1152/k204233/climatereconstructionAI/torch_img_levante.sif \
 python /work/bb1152/k204233/climatereconstructionAI/climatereconstructionAI/train_and_evaluate/train.py \
 --device cuda --batch-size 4 --image-sizes 512,256,256,256 --pooling-layers 3,2,2,2 --encoding-layers 4,4,4,4 --data-types pr,tas,uas,vas \
 --data-root-dir /work/bb1152/k204233/climatereconstructionAI/data/radolan-rea2/ \
 --mask-dir /work/bb1152/k204233/climatereconstructionAI/climatereconstructionAI/masks/ \
 --img-names radolan.h5,rea2-tas.h5,rea2-uas.h5,rea2-vas.h5 --mask-names single_radar_fail.h5,mask_ones_tas.h5,mask_ones_uas.h5,mask_ones_vas.h5 \
 --snapshot-dir /work/bb1152/k204233/climatereconstructionAI/climatereconstructionAI/snapshots/precipitation/radolan-rea-attention/ \
 --log-dir /work/bb1152/k204233/climatereconstructionAI/climatereconstructionAI/logs/precipitation/radolan-rea-attention/ \
 --lstm-steps 0 \
 --prev-next-steps 0 \
 --out-channels 1 \
 --max-iter 200000 \
 --resume-iter 100000 \
 --finetune \
 --eval-timesteps 2143,2144,2145,2146,2147 \
 --save-model-interval 5000 \
 --log-interval 100 \
 --loss-criterion 1
