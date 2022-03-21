#!/usr/bin/env bash

python train_and_evaluate/train.py \
 --device cpu --batch-size 2 --image-sizes 72 --pooling-layers 0 --encoding-layers 3 --data-types tas \
 --img-names 20cr-train.h5 --mask-names single_temp_mask.h5 \
 --data-root-dir ../data/20cr/ \
 --mask-dir masks/ \
 --snapshot-dir snapshots/temperature/20cr-lstm-test1/ \
 --log-dir logs/temperature/20cr-lstm-2/ \
 --prev-next-steps 0 \
 --out-channels 1 \
 --max-iter 1000000 \
 --save-model-interval 10 \
 --log-interval 1000 \
 --save-snapshot-image \
