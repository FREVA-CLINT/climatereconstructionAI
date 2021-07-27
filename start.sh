#!/usr/bin/env bash

####RAW
#python train.py --root /work/bb1152/k204233/climatereconstructionAI/data/radolan-complete-skaled --batch_size 4 --n_threads 64 --max_iter 100000 --mask_root /work/bb1152/k204233/climatereconstructionAI/climatereconstructionAI/masks/box_mask.h5 --save_dir ./snapshots/radolan-complete-skaled --log_dir ./logs/radolan-complete-skaled
####FINE
#python train.py --root /work/bb1152/k204233/climatereconstructionAI/data/radolan-complete-skaled --mask_root /work/bb1152/k204233/climatereconstructionAI/climatereconstructionAI/masks/box_mask.h5 --finetune --resume /work/bb1152/k204233/climatereconstructionAI/climatereconstructionAI/snapshots/radolan-complete-skaled/ckpt/100000.pth --batch_size 4 --n_threads 64 --max_iter 200000 --save_dir ./snapshots/radolan-complete-skaled --log_dir ./logs/radolan-complete-skaled
####TEST
python test.py --root /work/bb1152/k204233/climatereconstructionAI/data/radolan-complete-skaled --mask_root /work/bb1152/k204233/climatereconstructionAI/climatereconstructionAI/masks/box_mask.h5 --snapshot /work/bb1152/k204233/climatereconstructionAI/climatereconstructionAI/snapshots/radolan-complete-skaled/ckpt/200000.pth --partitions 2
