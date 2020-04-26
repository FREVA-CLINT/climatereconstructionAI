#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --qos=medium
#SBATCH --nodes=1
#SBATCH --ntasks=18
#SBATCH --time=01:55:00
#SBATCH --mem=10000
###SBATCH -w, --nodelist=g001
module load Python/3.6.6-intel-2018b
source /home/kadow/pytorch/venv/bin/activate
module load CUDA/10.0.130  
export HDF5_USE_FILE_LOCKING='FALSE'

####RAW
#python train.py --root /scratch/kadow/climate/hdf5 --batch_size 18 --n_threads 36 --max_iter 500000 --mask_root /home/kadow/pytorch/pytorch-hdf5-numpy/masks/hadcrut4-missmask.h5 --save_dir ./snapshots/20cr --log_dir ./logs/20cr #--resume /home/kadow/pytorch/pytorch-hdf5-numpy/snapshots/20cr/ckpt/200000.pth
####FINE
#python train.py --root /scratch/kadow/climate/hdf5 --mask_root /home/kadow/pytorch/pytorch-hdf5-numpy/masks/hadcrut4-missmask.h5 --finetune --resume /home/kadow/pytorch/pytorch-hdf5-numpy/snapshots/20cr/ckpt/500000.pth --batch_size 18 --n_threads 36 --max_iter 1000000 --save_dir ./snapshots/20cr --log_dir ./logs/20cr
####TEST
python test.py --root /scratch/kadow/climate/hdf5/to_test --mask_root /scratch/kadow/climate/hdf5/to_test/mask --snapshot /home/kadow/pytorch/pytorch-hdf5-numpy/snapshots/20cr/ckpt/1000000.pth
