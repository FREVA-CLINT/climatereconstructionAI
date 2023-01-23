import argparse
import json
import os
import os.path
import pkgutil

LAMBDA_DICT_IMG_INPAINTING = {
    'hole': 6.0, 'tv': 0.1, 'valid': 1.0, 'prc': 0.05, 'style': 120.0
}
LAMBDA_DICT_IMG_INPAINTING2 = {
    'hole': 0.0, 'tv': 0.1, 'valid': 7.0, 'prc': 0.05, 'style': 120.0
}
LAMBDA_DICT_HOLE = {
    'hole': 1.0
}
LAMBDA_DICT_VALID = {
    'valid': 1.0
}


def get_format(dataset_name):
    json_data = pkgutil.get_data(__name__, "static/dataset_format.json")
    dataset_format = json.loads(json_data)

    return dataset_format[str(dataset_name)]


class LoadFromFile(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        parser.parse_args(open(values).read().split(), namespace)


def str_list(arg):
    return arg.split(',')


def int_list(arg):
    return list(map(int, arg.split(',')))


def float_list(arg):
    return list(map(float, arg.split(',')))


def lim_list(arg):
    lim = list(map(float, arg.split(',')))
    assert len(lim) == 2
    return lim


def global_args(parser, arg_file=None, prog_func=None):
    import torch

    if arg_file is None:
        import sys
        argv = sys.argv[1:]
    else:
        argv = ["--load-from-file", arg_file]

    global progress_fwd
    progress_fwd = prog_func

    args = parser.parse_args(argv)

    args_dict = vars(args)
    for arg in args_dict:
        globals()[arg] = args_dict[arg]

    torch.backends.cudnn.benchmark = True
    globals()[device] = torch.device(device)

    globals()["dataset_format"] = get_format(args.dataset_name)

    global skip_layers

    if disable_skip_layers:
        skip_layers = 0
    else:
        skip_layers = 1

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    global recurrent_steps
    global n_recurrent_steps
    global time_steps
    time_steps = [0, 0]
    if lstm_steps:
        recurrent_steps = lstm_steps[0]
        time_steps = lstm_steps
    elif gru_steps:
        recurrent_steps = gru_steps[0]
        time_steps = gru_steps
    else:
        recurrent_steps = 0

    n_recurrent_steps = sum(time_steps) + 1

    global n_channel_steps
    global gt_channels

    n_channel_steps = 1
    gt_channels = [0 for i in range(out_channels)]
    if channel_steps:
        time_steps = channel_steps
        n_channel_steps = sum(channel_steps) + 1
        for i in range(out_channels):
            gt_channels[i] = (i + 1) * channel_steps[0] + i * (channel_steps[1] + 1)

    assert len(time_steps) == 2


def set_common_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data-root-dir', type=str, default='../data/',
                            help="Root directory containing the climate datasets")
    arg_parser.add_argument('--mask-dir', type=str, default='masks/', help="Directory containing the mask datasets")
    arg_parser.add_argument('--log-dir', type=str, default='logs/', help="Directory where the log files will be stored")
    arg_parser.add_argument('--data-names', type=str_list, default='train.nc',
                            help="Comma separated list of netCDF files (climate dataset) for training/infilling")
    arg_parser.add_argument('--mask-names', type=str_list, default=None,
                            help="Comma separated list of netCDF files (mask dataset). "
                                 "If None, it extracts the masks from the climate dataset")
    arg_parser.add_argument('--data-types', type=str_list, default='tas',
                            help="Comma separated list of variable types, "
                                 "in the same order as data-names and mask-names")
    arg_parser.add_argument('--n-target-data', type=int, default=0,
                            help="Number of data-names (from last) to be used as target data")
    arg_parser.add_argument('--device', type=str, default='cuda', help="Device used by PyTorch (cuda or cpu)")
    arg_parser.add_argument('--shuffle-masks', action='store_true', help="Select mask indices randomly")
    arg_parser.add_argument('--channel-steps', type=int_list, default=None,
                            help="Comma separated number of considered sequences for channeled memory:"
                                 "past_steps,future_steps")
    arg_parser.add_argument('--lstm-steps', type=int_list, default=None,
                            help="Comma separated number of considered sequences for lstm: past_steps,future_steps")
    arg_parser.add_argument('--gru-steps', type=int_list, default=None,
                            help="Comma separated number of considered sequences for gru: past_steps,future_steps")
    arg_parser.add_argument('--encoding-layers', type=int_list, default='3',
                            help="Number of encoding layers in the CNN")
    arg_parser.add_argument('--pooling-layers', type=int_list, default='0', help="Number of pooling layers in the CNN")
    arg_parser.add_argument('--conv-factor', type=int, default=None, help="Number of channels in the deepest layer")
    arg_parser.add_argument('--weights', type=str, default=None, help="Initialization weight")
    arg_parser.add_argument('--steady-masks', type=str_list, default=None,
                            help="Comma separated list of netCDF files containing a single mask to be applied "
                                 "to all timesteps. The number of steady-masks must be the same as out-channels")
    arg_parser.add_argument('--loop-random-seed', type=int, default=None,
                            help="Random seed for iteration loop")
    arg_parser.add_argument('--cuda-random-seed', type=int, default=None,
                            help="Random seed for CUDA")
    arg_parser.add_argument('--deterministic', action='store_true', help="Disable cudnn backends for reproducibility")
    arg_parser.add_argument('--attention', action='store_true', help="Enable the attention module")
    arg_parser.add_argument('--channel-reduction-rate', type=int, default=1,
                            help="Channel reduction rate for the attention module")
    arg_parser.add_argument('--disable-skip-layers', action='store_true', help="Disable the skip layers")
    arg_parser.add_argument('--disable-first-bn', action='store_true',
                            help="Disable the batch normalization on the first layer")
    arg_parser.add_argument('--masked-bn', action='store_true',
                            help="Use masked batch normalization instead of standard BN")
    arg_parser.add_argument('--lazy-load', action='store_true', help="Use lazy loading for large datasets")
    arg_parser.add_argument('--global-padding', action='store_true', help="Use a custom padding for global dataset")
    arg_parser.add_argument('--normalize-data', action='store_true',
                            help="Normalize the input climate data to 0 mean and 1 std")
    arg_parser.add_argument('--n-filters', type=int, default=None, help="Number of filters for the first/last layer")
    arg_parser.add_argument('--out-channels', type=int, default=1, help="Number of channels for the output data")
    arg_parser.add_argument('--dataset-name', type=str, default=None, help="Name of the dataset for format checking")
    arg_parser.add_argument('--min-bounds', type=float_list, default="inf",
                            help="Comma separated list of values defining the permitted lower-bound of output values")
    arg_parser.add_argument('--max-bounds', type=float_list, default="inf",
                            help="Comma separated list of values defining the permitted upper-bound of output values")
    arg_parser.add_argument('--profile', action='store_true', help="Profile code using tensorboard profiler")
    return arg_parser


def set_train_args(arg_file=None):
    arg_parser = set_common_args()
    arg_parser.add_argument('--val-names', type=str_list, default=None,
                            help="Comma separated list of netCDF files (climate dataset) for validation")
    arg_parser.add_argument('--snapshot-dir', type=str, default='snapshots/',
                            help="Parent directory of the training checkpoints and the snapshot images")
    arg_parser.add_argument('--resume-iter', type=int, help="Iteration step from which the training will be resumed")
    arg_parser.add_argument('--batch-size', type=int, default=18, help="Batch size")
    arg_parser.add_argument('--n-threads', type=int, default=64, help="Number of workers used in the data loader")
    arg_parser.add_argument('--multi-gpus', action='store_true', help="Use multiple GPUs, if any")
    arg_parser.add_argument('--finetune', action='store_true',
                            help="Enable the fine tuning mode (use fine tuning parameterization "
                                 "and disable batch normalization")
    arg_parser.add_argument('--lr', type=float, default=2e-4, help="Learning rate")
    arg_parser.add_argument('--lr-finetune', type=float, default=5e-5, help="Learning rate for fine tuning")
    arg_parser.add_argument('--max-iter', type=int, default=1000000, help="Maximum number of iterations")
    arg_parser.add_argument('--log-interval', type=int, default=None,
                            help="Iteration step interval at which a tensorboard summary log should be written")
    arg_parser.add_argument('--lr-scheduler-patience', type=int, default=None, help="Patience for the lr scheduler")
    arg_parser.add_argument('--save-snapshot-image', action='store_true',
                            help="Save evaluation images for the iteration steps defined in --log-interval")
    arg_parser.add_argument('--save-model-interval', type=int, default=50000,
                            help="Iteration step interval at which the model should be saved")
    arg_parser.add_argument('--n-final-models', type=int, default=1,
                            help="Number of final models to be saved")
    arg_parser.add_argument('--final-models-interval', type=int, default=1000,
                            help="Iteration step interval at which the final models should be saved")
    arg_parser.add_argument('--loss-criterion', type=int, default=0,
                            help="Index defining the loss function "
                                 "(0=original from Liu et al., 1=MAE of the hole region)")
    arg_parser.add_argument('--eval-timesteps', type=int_list, default="0,1,2,3,4",
                            help="Iteration steps for which an evaluation is performed")
    arg_parser.add_argument('-f', '--load-from-file', type=str, action=LoadFromFile,
                            help="Load all the arguments from a text file")
    arg_parser.add_argument('--vlim', type=lim_list, default=None,
                            help="Comma separated list of vmin,vmax values for the color scale of the snapshot images")
    global_args(arg_parser, arg_file)

    if globals()["val_names"] is None:
        globals()["val_names"] = globals()["data_names"].copy()


def set_evaluate_args(arg_file=None, prog_func=None):
    arg_parser = set_common_args()
    arg_parser.add_argument('--model-dir', type=str, default='snapshots/ckpt/', help="Directory of the trained models")
    arg_parser.add_argument('--model-names', type=str_list, default='1000000.pth', help="Model names")
    arg_parser.add_argument('--evaluation-dirs', type=str_list, default='evaluation/',
                            help="Directory where the output files will be stored")
    arg_parser.add_argument('--eval-names', type=str_list, default='output',
                            help="Prefix used for the output filenames")
    arg_parser.add_argument('--create-graph', action='store_true', help="Create a Tensorboard graph of the NN")
    arg_parser.add_argument('--plot-results', type=int_list, default=[],
                            help="Create plot images of the results for the comma separated list of time indices")
    arg_parser.add_argument('--partitions', type=int, default=1,
                            help="Split the climate dataset into several partitions along the time coordinate")
    arg_parser.add_argument('--maxmem', type=int, default=None,
                            help="Maximum available memory in MB (overwrite partitions parameter)")
    arg_parser.add_argument('--split-outputs', action='store_true',
                            help="Do not merge the outputs when using multiple models and/or partitions")
    arg_parser.add_argument('-f', '--load-from-file', type=str, action=LoadFromFile,
                            help="Load all the arguments from a text file")
    global_args(arg_parser, arg_file, prog_func)
