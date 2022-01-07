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
 --device cuda --batch-size 4 --image-sizes 128 --pooling-layers 3 --encoding-layers 4 --data-types pr \
 --data-root-dir /work/bb1152/k204233/climatereconstructionAI/data/radolan-rea2-128x128/ \
 --mask-dir /work/bb1152/k204233/climatereconstructionAI/climatereconstructionAI/masks/ \
 --img-names radolan.h5 --mask-names single_radar_fail_128x128.h5 \
 --snapshot-dir /work/bb1152/k204233/climatereconstructionAI/climatereconstructionAI/snapshots/precipitation/128x128/2007-2013/lstm-prev-next-hole/ \
 --log-dir /work/bb1152/k204233/climatereconstructionAI/climatereconstructionAI/logs/precipitation/128x128/2007-2013/lstm-prev-next-hole/ \
 --out-channels 1 \
 --lstm-steps 2 \
 --prev-next-steps 0 \
 --max-iter 250000 \
 --eval-timesteps 2143,2144,2145,2146,2147 \
 --save-model-interval 5000 \
 --log-interval 100 \
 --loss-criterion 1
singularity run --bind /work/bb1152/k204233/ --nv /work/bb1152/k204233/climatereconstructionAI/torch_img_mistral.sif \
 python /work/bb1152/k204233/climatereconstructionAI/climatereconstructionAI/train_and_evaluate/train.py \
 --device cuda --batch-size 4 --image-sizes 128 --pooling-layers 3 --encoding-layers 4 --data-types pr \
 --data-root-dir /work/bb1152/k204233/climatereconstructionAI/data/radolan-rea2-128x128/ \
 --mask-dir /work/bb1152/k204233/climatereconstructionAI/climatereconstructionAI/masks/ \
 --img-names radolan.h5 --mask-names single_radar_fail_128x128.h5 \
 --snapshot-dir /work/bb1152/k204233/climatereconstructionAI/climatereconstructionAI/snapshots/precipitation/128x128/2007-2013/lstm-prev-next-hole/ \
 --log-dir /work/bb1152/k204233/climatereconstructionAI/climatereconstructionAI/logs/precipitation/128x128/2007-2013/lstm-prev-next-hole/ \
 --lstm-steps 2 \
 --prev-next-steps 0 \
 --out-channels 1 \
 --max-iter 500000 \
 --resume-iter 200000 \
 --finetune \
 --eval-timesteps 2143,2144,2145,2146,2147 \
 --save-model-interval 5000 \
 --log-interval 100 \
 --loss-criterion 1