#!/usr/bin/env bash

python /home/joe/PycharmProjects/climatereconstructionAI/climatereconstructionAI/train_and_evaluate/evaluate.py \
 --device cpu --image-size 72 --pooling-layers 0 --encoding-layers 1 --data-type tas \
 --data-root-dir ../data/20cr/ \
 --mask-dir masks/hadcrut4-missmask.h5 \
 --snapshot-dir snapshots/temperature/20cr/ckpt/1400.pth \
 --evaluation-dir evaluation/temperature/20cr-lstm/ \
 --lstm-steps 3 \
 --partitions 1 \
 --infill infill \
# --create-report \
# --create-video \
# --create-images 2017-07-12-14:00,2017-07-12-14:00 \
