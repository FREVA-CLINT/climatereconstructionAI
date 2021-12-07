import argparse

import torch

LAMBDA_DICT_IMG_INPAINTING = {
    'valid': 1.0, 'hole': 6.0, 'tv': 0.1, 'prc': 0.05, 'style': 120.0
}
LAMBDA_DICT_PR_INPAINTING = {
    'SSL-OUT': 1.0,  # 'SSL-OUT-COMP': 1.0, 'valid': 1.0, 'hole': 6.0, 'tv': 0.1, 'prc': 0.05, 'style': 120.0
}

data_types = None
mask_names = None
img_names = None
evaluation_dirs = None
partitions = None
infill = None
create_images = None
create_video = None
create_report = None
log_dir = None
snapshot_dir = None
data_root_dir = None
mask_dir = None
resume = None
device = None
batch_size = None
n_threads = None
finetune = None
lr = None
lr_finetune = None
max_iter = None
log_interval = None
save_model_interval = None
lstm_steps = None
encoding_layers = None
pooling_layers = None
image_size = None
eval_names = None
mask_zero = None


def set_train_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data-types', type=str, default='tas')
    arg_parser.add_argument('--log-dir', type=str, default='logs/')
    arg_parser.add_argument('--snapshot-dir', type=str, default='snapshots/')
    arg_parser.add_argument('--data-root-dir', type=str, default='../data/')
    arg_parser.add_argument('--mask-dir', type=str, default='masks/')
    arg_parser.add_argument('--img-names', type=str, default='train.h5')
    arg_parser.add_argument('--mask-names', type=str, default='mask.h5')
    arg_parser.add_argument('--resume', type=str)
    arg_parser.add_argument('--device', type=str, default='cuda')
    arg_parser.add_argument('--batch-size', type=int, default=18)
    arg_parser.add_argument('--n-threads', type=int, default=64)
    arg_parser.add_argument('--finetune', action='store_true')
    arg_parser.add_argument('--lr', type=float, default=2e-4)
    arg_parser.add_argument('--lr-finetune', type=float, default=5e-5)
    arg_parser.add_argument('--max-iter', type=int, default=1000000)
    arg_parser.add_argument('--log-interval', type=int, default=100)
    arg_parser.add_argument('--save-model-interval', type=int, default=50000)
    arg_parser.add_argument('--lstm-steps', type=int, default=0)
    arg_parser.add_argument('--encoding-layers', type=int, default=3)
    arg_parser.add_argument('--pooling-layers', type=int, default=0)
    arg_parser.add_argument('--image-size', type=int, default=72)
    args = arg_parser.parse_args()

    global data_types
    global img_names
    global mask_names
    global log_dir
    global snapshot_dir
    global data_root_dir
    global mask_dir
    global resume
    global device
    global batch_size
    global n_threads
    global finetune
    global lr
    global lr_finetune
    global max_iter
    global log_interval
    global save_model_interval
    global lstm_steps
    global encoding_layers
    global pooling_layers
    global image_size

    data_types = args.data_types.split(',')
    img_names = args.img_names.split(',')
    mask_names = args.mask_names.split(',')
    log_dir = args.log_dir
    snapshot_dir = args.snapshot_dir
    data_root_dir = args.data_root_dir
    mask_dir = args.mask_dir
    resume = args.resume
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)
    batch_size = args.batch_size
    n_threads = args.n_threads
    finetune = args.finetune
    lr = args.lr
    lr_finetune = args.lr_finetune
    max_iter = args.max_iter
    log_interval = args.log_interval
    save_model_interval = args.save_model_interval
    lstm_steps = args.lstm_steps
    encoding_layers = args.encoding_layers
    pooling_layers = args.pooling_layers
    image_size = args.image_size


def set_evaluation_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data-types', type=str, default='tas')
    arg_parser.add_argument('--img-names', type=str, default='train.h5')
    arg_parser.add_argument('--mask-names', type=str, default='mask.h5')
    arg_parser.add_argument('--evaluation-dirs', type=str, default='evaluation/')
    arg_parser.add_argument('--snapshot-dir', type=str, default='snapshots/')
    arg_parser.add_argument('--data-root-dir', type=str, default='../data/')
    arg_parser.add_argument('--mask-dir', type=str, default='masks/')
    arg_parser.add_argument('--device', type=str, default='cuda')
    arg_parser.add_argument('--partitions', type=int, default=1)
    arg_parser.add_argument('--prev-next', type=int, default=0)
    arg_parser.add_argument('--lstm-steps', type=int, default=0)
    arg_parser.add_argument('--encoding-layers', type=int, default=3)
    arg_parser.add_argument('--pooling-layers', type=int, default=0)
    arg_parser.add_argument('--image-size', type=int, default=72)
    arg_parser.add_argument('--infill', type=str, default=None)
    arg_parser.add_argument('--create-images', type=str, default=None)
    arg_parser.add_argument('--create-video', action='store_true')
    arg_parser.add_argument('--create-report', action='store_true')
    arg_parser.add_argument('--eval-names', type=str, default='Output')
    arg_parser.add_argument('--mask-zero', type=float, default=None)
    args = arg_parser.parse_args()

    global data_types
    global img_names
    global mask_names
    global evaluation_dirs
    global snapshot_dir
    global data_root_dir
    global mask_dir
    global device
    global partitions
    global prev_next
    global lstm_steps
    global encoding_layers
    global pooling_layers
    global image_size
    global infill
    global create_images
    global create_video
    global create_report
    global eval_names
    global mask_zero

    data_types = args.data_types.split(',')
    img_names = args.img_names.split(',')
    mask_names = args.mask_names.split(',')
    evaluation_dirs = args.evaluation_dirs.split(',')
    snapshot_dir = args.snapshot_dir
    data_root_dir = args.data_root_dir
    mask_dir = args.mask_dir
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)
    partitions = args.partitions
    prev_next = args.prev_next
    lstm_steps = args.lstm_steps
    encoding_layers = args.encoding_layers
    pooling_layers = args.pooling_layers
    image_size = args.image_size
    infill = args.infill
    create_images = args.create_images
    create_video = args.create_video
    create_report = args.create_report
    eval_names = args.eval_names.split(',')
    mask_zero = args.mask_zero

