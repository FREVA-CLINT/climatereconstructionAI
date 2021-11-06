#!/usr/bin/env bash

python /home/joe/PycharmProjects/climatereconstructionAI/climatereconstructionAI/evaluate.py \
 --device cpu --image-size 512 --pooling-layers 0 --encoding-layers 3 --data-type pr \
 --data-root-dir ../data/20cr/ \
 --mask-dir masks/single_radar_fail.h5 \
 --snapshot-dir snapshots/temperature/20cr/ckpt/1000.pth \
 --evaluation-dir evaluation/precipitation/radolan-prev-next-3/ \
 --prev-next 3 \
 --partitions 1 \
 --create-report \
 --create-video \
 --create-images 2017-07-12-14:00,2017-07-12-14:00 \
# --infill infill \
