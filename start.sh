#!/usr/bin/env bash

####RAW
python train.py --root /work/bb1152/k204233/climatereconstructionAI/data --batch_size 18 --n_threads 32 --max_iter 500000 --mask_root /work/bb1152/k204233/climatereconstructionAI/climatereconstructionAI/masks/radclim_missmask.h5 --save_dir ./snapshots/radclim --log_dir ./logs/radclim
####FINE
#python train.py --root /work/bb1152/k204233/climatereconstructionAI/data --mask_root /work/bb1152/k204233/climatereconstructionAI/climatereconstructionAI/masks/hadcrut4-missmask.h5 --finetune --resume /work/bb1152/k204233/climatereconstructionAI/climatereconstructionAI/snapshots/20cr/ckpt/500000.pth --batch_size 18 --n_threads 4 --max_it$
####TEST
#python test.py --root /work/bb1152/k204233/climatereconstructionAI/data/ --mask_root /work/bb1152/k204233/climatereconstructionAI/climatereconstructionAI/masks/hadcrut_missmask_187001-200512.h5 --snapshot /work/bb1152/k204233/climatereconstructionAI/climatereconstructionAI/snapshots/20cr/ckpt/1000000.pth --partitions 2
