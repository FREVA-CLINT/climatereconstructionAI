#!/usr/bin/env bash

python train_and_evaluate/evaluate.py \
 --device cpu --image-size 72 --pooling-layers 0 --encoding-layers 3 --data-types tas \
 --img-names single.h5 --mask-names single_temp_mask.h5 \
 --data-root-dir ../data/20cr/ \
 --mask-dir masks/ \
 --snapshot-dir snapshots/temperature/20cr-lstm-test1/ckpt/20.pth \
 --evaluation-dirs evaluation/precipitation/test/ \
 --lstm-steps 0 \
 --partitions 1 \
 --eval-names Output \
 --create-sum-maps
# --mask-zero 0.05
# --create-video \
# --create-images 2017-07-12-14:00,2017-07-12-14:00 \
