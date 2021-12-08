#!/usr/bin/env bash

python /home/joe/PycharmProjects/climatereconstructionAI/climatereconstructionAI/train_and_evaluate/evaluate.py \
 --device cpu --image-size 72 --pooling-layers 0 --encoding-layers 1 --data-types pr \
 --img-names tas_20cr_RECONSTRUCT_187001-200512.h5 --mask-names single_radar_fail.h5 \
 --data-root-dir ../data/20cr/ \
 --mask-dir masks/ \
 --snapshot-dir snapshots/temperature/cmip/ckpt/500000.pth \
 --evaluation-dirs evaluation/precipitation/test/ \
 --create-report \
 --lstm-steps 0 \
 --partitions 1 \
 --eval-names Output1,Output2,Output3,Output4 \
# --mask-zero 0.05
# --infill test \
# --create-video \
# --create-images 2017-07-12-14:00,2017-07-12-14:00 \
