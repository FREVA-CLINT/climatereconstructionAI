#!/usr/bin/env bash

python /home/joe/PycharmProjects/climatereconstructionAI/climatereconstructionAI/train_and_evaluate/trainLSTM.py \
 --device cpu --batch-size 2 --image-size 72 --pooling-layers 0 --encoding-layers 3 --data-type tas \
 --data-root-dir /home/joe/PycharmProjects/climatereconstructionAI/data/20cr/ \
 --mask-dir /home/joe/PycharmProjects/climatereconstructionAI/climatereconstructionAI/masks/hadcrut4-missmask.h5 \
 --snapshot-dir /home/joe/PycharmProjects/climatereconstructionAI/climatereconstructionAI/snapshots/temperature/20cr-lstm/ \
 --log-dir /home/joe/PycharmProjects/climatereconstructionAI/climatereconstructionAI/logs/temperature/20cr-lstm// \
 --lstm-steps 0 \
 --max-iter 1000000 \
 --save-model-interval 10000