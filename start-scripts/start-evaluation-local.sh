#!/usr/bin/env bash

python /home/joe/PycharmProjects/climatereconstructionAI/climatereconstructionAI/evaluate.py \
 --device cpu --image-size 72 --pooling-layers 0 --encoding-layers 3 --data-type tas \
 --data-root-dir ../data/20cr/ \
 --mask-dir masks/hadcrut4-missmask.h5 \
 --snapshot-dir snapshots/temperature/20cr/ckpt/1000.pth \
 --evaluation-dir evaluation/temperature/20cr/ \
 --prev-next 0 \
 --partitions 272 \
 --infill infill \
# --create-images \
# --create-video \
# --create-report \
