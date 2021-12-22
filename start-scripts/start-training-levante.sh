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
 python /work/bb1152/k204233/climatereconstructionAI/climatereconstructionAI/train_and_evaluate/train.py \
 --device cuda --batch-size 4 --image-sizes 512,512 --pooling-layers 3,2 --encoding-layers 4,4 --data-types pr,tas \
 --data-root-dir /work/bb1152/k204233/climatereconstructionAI/data/radolan-rea2/ \
 --mask-dir /work/bb1152/k204233/climatereconstructionAI/climatereconstructionAI/masks/ \
 --img-names radolan.h5,rea2-tas-celsius.h5 --mask-names single_radar_fail.h5,mask_ones_tas.h5 \
 --snapshot-dir /work/bb1152/k204233/climatereconstructionAI/climatereconstructionAI/snapshots/precipitation/radolan-fusion/ \
 --log-dir /work/bb1152/k204233/climatereconstructionAI/climatereconstructionAI/logs/precipitation/radolan-fusion/ \
 --out-channels 1 \
 --lstm-steps 0 \
 --prev-next-steps 0 \
 --max-iter 100000 \
 --resume-iter 50000 \
 --log-interval 100 \
 --eval-timesteps 2143,2144,2145,2146,2147 \
 --save-model-interval 10000
singularity run --bind /work/bb1152/k204233/ --nv /work/bb1152/k204233/climatereconstructionAI/torch_img_levante.sif \
 python /work/bb1152/k204233/climatereconstructionAI/climatereconstructionAI/train_and_evaluate/train.py \
 --device cuda --batch-size 4 --image-sizes 512,512 --pooling-layers 3,2 --encoding-layers 4,4 --data-types pr,tas \
 --data-root-dir /work/bb1152/k204233/climatereconstructionAI/data/radolan-rea2/ \
 --mask-dir /work/bb1152/k204233/climatereconstructionAI/climatereconstructionAI/masks/ \
 --img-names radolan.h5,rea2-tas-celsius.h5 --mask-names single_radar_fail.h5,mask_ones_tas.h5 \
 --snapshot-dir /work/bb1152/k204233/climatereconstructionAI/climatereconstructionAI/snapshots/precipitation/radolan-fusion/ \
 --log-dir /work/bb1152/k204233/climatereconstructionAI/climatereconstructionAI/logs/precipitation/radolan-fusion/ \
 --lstm-steps 0 \
 --prev-next-steps 0 \
 --out-channels 1 \
 --max-iter 200000 \
 --resume-iter 100000 \
 --finetune \
 --log-interval 100 \
 --eval-timesteps 2143,2144,2145,2146,2147 \
 --save-model-interval 10000
