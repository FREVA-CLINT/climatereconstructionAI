# TRAIN OPTIONS
mask_train_dir = "/work/bb1152/k204233/climatereconstructionAI/climatereconstructionAI/masks/box_mask.h5"
data_root_train_dir = "/work/bb1152/k204233/climatereconstructionAI/data/radolan-complete-skaled"
save_dir = "./snapshots/radolan-complete-skaled"
log_dir = "./logs/radolan-complete-skaled"
resume_dir = "/work/bb1152/k204233/climatereconstructionAI/climatereconstructionAI/snapshots/radolan-complete-skaled/ckpt/100000.pth"
data_type = 'pr'
device = 'cuda'
resume = False
fine_tune = False
batch_size = 4
n_threads = 64
max_iter = 100000
log_interval = 10
vis_interval = 5000
save_model_interval = 50000
lr_finetune = 5e-5
lr = 2e-4

# TEST OPTIONS
mask_test_dir = "/work/bb1152/k204233/climatereconstructionAI/climatereconstructionAI/masks/box_mask.h5"
data_root_test_dir = "/work/bb1152/k204233/climatereconstructionAI/data/radolan-complete-skaled"
snapshot_dir = "/work/bb1152/k204233/climatereconstructionAI/climatereconstructionAI/snapshots/radolan-complete-skaled/ckpt/200000.pth"
partitions = 2

# TEST/TRAIN OPTIONS
image_size = 256
