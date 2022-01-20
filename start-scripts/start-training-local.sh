#!/usr/bin/env bash

python train_and_evaluate/train.py \
 --device cpu --batch-size 2 --image-sizes 128,64,64,64 --pooling-layers 3,2,2,2 --encoding-layers 4,4,4,4 --data-types pr,tas,uas,vas \
 --img-names radolan.h5,rea2-tas.h5,rea2-uas.h5,rea2-vas.h5 --mask-names single_radar_fail_128x128.h5,mask_ones_tas_64x64.h5,mask_ones_uas_64x64.h5,mask_ones_vas_64x64.h5 \
 --data-root-dir ../data/radolan-rea2-128x128/ \
 --mask-dir masks/ \
 --snapshot-dir snapshots/temperature/20cr-lstm-test1/ \
 --log-dir logs/temperature/20cr-lstm-2/ \
 --lstm-steps 2 \
 --prev-next-steps 0 \
 --out-channels 1 \
 --max-iter 1000000 \
 --save-model-interval 10000 \
 --log-interval 1000 \
 --save-snapshot-image \
 --attention