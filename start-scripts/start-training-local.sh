#!/usr/bin/env bash

python /home/joe/PycharmProjects/climatereconstructionAI/climatereconstructionAI/trainMultiCUNet.py \
 --device cpu --batch-size 2 --image-size 72 --pooling-layers 0 --encoding-layers 1 --data-type tas \
 --data-root-dir ../data/20cr/ \
 --mask-dir masks/hadcrut4-missmask.h5 \
 --snapshot-dir snapshots/temperature/20cr/ \
 --log-dir logs/temperature/20cr/ \
 --prev-next 4 \
 --save-model-interval 1000 \
# --resume snapshots/precipitation/20cr/ckpt/100000.pth \
# --finetune