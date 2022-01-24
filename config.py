import argparse

import torch

LAMBDA_DICT_IMG_INPAINTING = {
    'hole': 6.0, 'tv': 0.1, 'valid': 1.0, 'prc': 0.05, 'style': 120.0
}
LAMBDA_DICT_HOLE = {
    'hole': 1.0
}

PDF_BINS = [0, 0.01, 0.02, 0.1, 1, 2, 10, 100]

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
snapshot_dirs = None
data_root_dir = None
mask_dir = None
resume_iter = None
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
prev_next_steps = None
encoding_layers = None
pooling_layers = None
image_sizes = None
eval_names = None
eval_threshold = None
eval_range = None
ts_range = None
eval_timesteps = None
out_channels = None
gt_channels = None
channel_reduction_rate = None
save_snapshot_image = None
loss_criterion = None
attention = None
smoothing_factor = None
weights = None
skip_layers = None


def set_train_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data-types', type=str, default='tas')
    arg_parser.add_argument('--log-dir', type=str, default='logs/')
    arg_parser.add_argument('--snapshot-dir', type=str, default='snapshots/')
    arg_parser.add_argument('--data-root-dir', type=str, default='../data/')
    arg_parser.add_argument('--mask-dir', type=str, default='masks/')
    arg_parser.add_argument('--img-names', type=str, default='train.h5')
    arg_parser.add_argument('--mask-names', type=str, default='mask.h5')
    arg_parser.add_argument('--resume-iter', type=int)
    arg_parser.add_argument('--device', type=str, default='cuda')
    arg_parser.add_argument('--batch-size', type=int, default=18)
    arg_parser.add_argument('--n-threads', type=int, default=64)
    arg_parser.add_argument('--finetune', action='store_true')
    arg_parser.add_argument('--lr', type=float, default=2e-4)
    arg_parser.add_argument('--lr-finetune', type=float, default=5e-5)
    arg_parser.add_argument('--max-iter', type=int, default=1000000)
    arg_parser.add_argument('--log-interval', type=int, default=None)
    arg_parser.add_argument('--save-snapshot-image', action='store_true')
    arg_parser.add_argument('--save-model-interval', type=int, default=50000)
    arg_parser.add_argument('--lstm-steps', type=int, default=0)
    arg_parser.add_argument('--prev-next-steps', type=int, default=0)
    arg_parser.add_argument('--encoding-layers', type=str, default='3')
    arg_parser.add_argument('--pooling-layers', type=str, default='0')
    arg_parser.add_argument('--image-sizes', type=str, default='72')
    arg_parser.add_argument('--out-channels', type=int, default=1)
    arg_parser.add_argument('--loss-criterion', type=int, default=0)
    arg_parser.add_argument('--eval-timesteps', type=str, default="0,1,2,3,4")
    arg_parser.add_argument('--weights', type=str, default=None)
    arg_parser.add_argument('--channel-reduction-rate', type=int, default=1)
    arg_parser.add_argument('--attention', action='store_true')
    arg_parser.add_argument('--disable-skip-layers', action='store_true')
    args = arg_parser.parse_args()

    global data_types
    global img_names
    global mask_names
    global log_dir
    global snapshot_dir
    global data_root_dir
    global mask_dir
    global resume_iter
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
    global prev_next_steps
    global encoding_layers
    global pooling_layers
    global image_sizes
    global eval_timesteps
    global out_channels
    global gt_channels
    global channel_reduction_rate
    global save_snapshot_image
    global loss_criterion
    global attention
    global skip_layers
    global weights

    data_types = args.data_types.split(',')
    img_names = args.img_names.split(',')
    mask_names = args.mask_names.split(',')
    eval_timesteps = args.eval_timesteps.split(',')
    log_dir = args.log_dir
    snapshot_dir = args.snapshot_dir
    data_root_dir = args.data_root_dir
    mask_dir = args.mask_dir
    resume_iter = args.resume_iter
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
    prev_next_steps = args.prev_next_steps
    encoding_layers = list(map(int, args.encoding_layers.split(',')))
    pooling_layers = list(map(int, args.pooling_layers.split(',')))
    image_sizes = list(map(int, args.image_sizes.split(',')))
    channel_reduction_rate = args.channel_reduction_rate
    out_channels = args.out_channels
    save_snapshot_image = args.save_snapshot_image
    gt_channels = []
    loss_criterion = args.loss_criterion
    attention = args.attention
    weights = args.weights
    if args.disable_skip_layers:
        skip_layers = 0
    else:
        skip_layers = 1
    for i in range(out_channels):
        gt_channels.append((i + 1) * prev_next_steps + i * (prev_next_steps + 1))


def set_evaluation_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data-types', type=str, default='tas')
    arg_parser.add_argument('--img-names', type=str, default='train.h5')
    arg_parser.add_argument('--mask-names', type=str, default='mask.h5')
    arg_parser.add_argument('--evaluation-dirs', type=str, default='evaluation/')
    arg_parser.add_argument('--snapshot-dirs', type=str, default='snapshots/')
    arg_parser.add_argument('--data-root-dir', type=str, default='../data/')
    arg_parser.add_argument('--mask-dir', type=str, default='masks/')
    arg_parser.add_argument('--device', type=str, default='cuda')
    arg_parser.add_argument('--partitions', type=int, default=1)
    arg_parser.add_argument('--prev-next', type=int, default=0)
    arg_parser.add_argument('--lstm-steps', type=int, default=0)
    arg_parser.add_argument('--prev-next-steps', type=int, default=0)
    arg_parser.add_argument('--encoding-layers', type=str, default='3')
    arg_parser.add_argument('--pooling-layers', type=str, default='0')
    arg_parser.add_argument('--image-sizes', type=str, default='72')
    arg_parser.add_argument('--infill', type=str, default=None)
    arg_parser.add_argument('--create-images', type=str, default=None)
    arg_parser.add_argument('--create-video', action='store_true')
    arg_parser.add_argument('--create-report', action='store_true')
    arg_parser.add_argument('--eval-names', type=str, default='Output')
    arg_parser.add_argument('--eval-range', type=str, default=None)
    arg_parser.add_argument('--ts-range', type=str, default=None)
    arg_parser.add_argument('--out-channels', type=int, default=1)
    arg_parser.add_argument('--eval-threshold', type=float, default=None)
    arg_parser.add_argument('--smoothing-factor', type=int, default=1)
    arg_parser.add_argument('--channel-reduction-rate', type=int, default=1)
    arg_parser.add_argument('--attention', action='store_true')
    arg_parser.add_argument('--disable-skip-layers', action='store_true')
    args = arg_parser.parse_args()

    global data_types
    global img_names
    global mask_names
    global evaluation_dirs
    global snapshot_dirs
    global data_root_dir
    global mask_dir
    global device
    global partitions
    global lstm_steps
    global prev_next_steps
    global encoding_layers
    global pooling_layers
    global image_sizes
    global infill
    global create_images
    global create_video
    global create_report
    global eval_names
    global eval_threshold
    global eval_range
    global ts_range
    global out_channels
    global channel_reduction_rate
    global attention
    global smoothing_factor
    global skip_layers

    data_types = args.data_types.split(',')
    img_names = args.img_names.split(',')
    mask_names = args.mask_names.split(',')
    evaluation_dirs = args.evaluation_dirs.split(',')
    snapshot_dirs = args.snapshot_dirs.split(',')
    data_root_dir = args.data_root_dir
    mask_dir = args.mask_dir
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)
    partitions = args.partitions
    lstm_steps = args.lstm_steps
    prev_next_steps = args.prev_next_steps
    encoding_layers = list(map(int, args.encoding_layers.split(',')))
    pooling_layers = list(map(int, args.pooling_layers.split(',')))
    image_sizes = list(map(int, args.image_sizes.split(',')))
    infill = args.infill
    if args.create_images:
        create_images = args.create_images.split(',')
    create_video = args.create_video
    create_report = args.create_report
    eval_names = args.eval_names.split(',')
    if args.eval_range:
        eval_range = args.eval_range.split(',')
    if args.ts_range:
        ts_range = args.ts_range.split(',')
    eval_threshold = args.eval_threshold
    out_channels = args.out_channels
    channel_reduction_rate = args.channel_reduction_rate
    attention = args.attention
    smoothing_factor = args.smoothing_factor
    gt_channels = []
    if args.disable_skip_layers:
        skip_layers = 0
    else:
        skip_layers = 1
    for i in range(out_channels):
        gt_channels.append((i + 1) * prev_next_steps + i * (prev_next_steps + 1))

