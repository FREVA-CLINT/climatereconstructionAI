#!/usr/bin/env bash

python /home/joe/PycharmProjects/climatereconstructionAI/climatereconstructionAI/evaluate.py \
 --device cpu --image-size 512 --pooling-layers 3 --encoding-layers 4 --data-type pr \
 --data-root-dir ../data/20cr/ \
 --mask-dir masks/single_radar_fail.h5 \
 --snapshot-dir snapshots/temperature/20cr/ckpt/1000.pth \
 --evaluation-dir evaluation/radolan-prev-next/ \
 --prev-next 1 \
 --partitions 272 \
 --create-images 2017-07-12-14:00,2017-07-12-14:00 \
# --infill infill \
# --create-video \
# --create-report \
