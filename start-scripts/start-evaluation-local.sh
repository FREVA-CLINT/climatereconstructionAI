#!/usr/bin/env bash

python /home/joe/PycharmProjects/climatereconstructionAI/climatereconstructionAI/evaluate.py \
 --device cpu --image-size 72 --pooling-layers 0 --encoding-layers 3 --data-type tas \
 --data-root-dir ../data/20cr/ \
 --mask-dir masks/hadcrut4-missmask.h5 \
 --snapshot-dir snapshots/temperature/20cr/ckpt/1000.pth \
 --evaluation-dir evaluation/temperature/20cr/ \
 --prev-next 3 \
 --partitions 1 \
 --create-images 2000-07-12-14:00,2017-07-12-14:00 \
 --infill infill \
 --create-video \
 --create-report \
