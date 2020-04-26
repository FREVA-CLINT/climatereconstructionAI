#!/bin/bash                                                                                                                                

#SBATCH --partition=gpu                                                                                                                    
#SBATCH --qos=medium                                                                                                                       
#SBATCH --nodes=1                                                                                                                          
#SBATCH --ntasks=18                                                                                                                        
#SBATCH --time=02:55:00                                                                                                                    
#SBATCH --mem=10000                                                                                                                        
###SBATCH -w, --nodelist=g001                                                                                                              
module load Python/3.6.6-intel-2018b
source /home/kadow/pytorch/venv/bin/activate
module load CUDA/10.0.130
export HDF5_USE_FILE_LOCKING='FALSE'

####TEST

for ((i=1; i<=100; i++)); do
    echo $i
    rm /scratch/kadow/climate/hdf5/to_test/val_large
    ln -s /scratch/kadow/climate/hdf5/TEST-SUITE/hadcrut/FULL-100/HADCRUT4/r${i}i1p1 /scratch/kadow/climate/hdf5/to_test/val_large;
    python test.py --root /scratch/kadow/climate/hdf5/to_test --mask_root /scratch/kadow/climate/hdf5/to_test/mask --snapshot /home/kadow/pytorch/pytorch-hdf5-numpy/snapshots/cmip/ckpt/1000000.pth
    sleep 1
    mkdir -p /home/kadow/pytorch/pytorch-hdf5-numpy/RESULTS/cmipAI/h5-cmipAI-hadcrut-FULL-100/r${i}i1p1
    mv /home/kadow/pytorch/pytorch-hdf5-numpy/h5/* /home/kadow/pytorch/pytorch-hdf5-numpy/RESULTS/cmipAI/h5-cmipAI-hadcrut-FULL-100/r${i}i1p1
done
