import argparse
import json
import os
import os.path
import pkgutil


def get_format(dataset_name):
    json_data = pkgutil.get_data(__name__, "static/dataset_format.json")
    dataset_format = json.loads(json_data)

    return dataset_format[str(dataset_name)]


def get_passed_arguments(args, parser):
    sentinel = object()
    ns = argparse.Namespace(**{key: sentinel for key in vars(args)})
    parser.parse_known_args(namespace=ns)
    return {key: val for key, val in vars(ns).items() if val is not sentinel}


class LoadFromFile(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        parser.parse_known_args(open(values).read().split(), namespace)


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


def key_value_list(arg):
    args = arg.split(',')
    keys = [arg for arg in args if not str.isnumeric(arg[0])]
    values = [float(arg) for arg in args if str.isnumeric(arg[0])]
    return dict(zip(keys, values))


def set_lambdas():
    global lambda_dict

    lambda_dict = {}

    if loss_criterion in ("0", "inpainting"):
        lambda_dict['valid'] = 1.
        lambda_dict['hole'] = 6.
        lambda_dict['tv'] = .1
        lambda_dict['prc'] = .05
        lambda_dict['style'] = 120.

    elif loss_criterion in ("1", "l1-hole"):
        lambda_dict['hole'] = 1.

    elif loss_criterion in ("2", "downscaling"):
        lambda_dict['valid'] = 7.
        lambda_dict['hole'] = 0.
        lambda_dict['tv'] = .1
        lambda_dict['prc'] = .05
        lambda_dict['style'] = 120.

    elif loss_criterion in ("3", "l1-valid"):
        lambda_dict['valid'] = 1.

    elif loss_criterion in ("4", "extreme"):
        lambda_dict['-extreme'] = 1.
        lambda_dict['+extreme'] = 1.

    if vae_zdim != 0:
        lambda_dict['kldiv'] = 1.

    if lambda_loss is not None:
        lambda_dict.update(lambda_loss)

def set_steps(evaluate=False):

    assert sum(bool(x) for x in [lstm_steps, gru_steps, channel_steps]) < 2, \
        "lstm, gru and channel options are mutually exclusive"

    global recurrent_steps, n_recurrent_steps
    global time_steps
    time_steps = [0, 0]
    if lstm_steps:
        time_steps = lstm_steps
        recurrent_steps = lstm_steps[0]
    elif gru_steps:
        time_steps = gru_steps
        recurrent_steps = gru_steps[0]
    else:
        recurrent_steps = 0

    n_recurrent_steps = sum(time_steps) + 1

    global n_channel_steps, gt_channels
    if channel_steps:
        time_steps = channel_steps
        n_channel_steps = sum(channel_steps) + 1
        gt_channels = [i * n_channel_steps + channel_steps[0] for i in range(n_output_data)]
    else:
        n_channel_steps = 1
        gt_channels = [0 for i in range(n_output_data)]

    global n_time_steps, in_steps, out_steps, n_pred_steps, pred_timestep, out_channels

    n_time_steps = sum(time_steps) + 1
    pred_timestep = list(range(-pred_steps[0], pred_steps[1] + 1))
    n_pred_steps = len(pred_timestep)

    if evaluate:
        in_steps = range(0, n_time_steps)
        out_steps = [time_steps[0]]
        out_channels = n_output_data * n_pred_steps
    else:
        in_step = max(pred_steps[0] - time_steps[0], 0)
        in_steps = range(in_step, in_step + n_time_steps)
        n_time_steps = len(in_steps)
        interval = [max(time_steps[i], pred_steps[i]) for i in range(2)]
        out_steps = range(interval[0] - pred_steps[0], interval[0] + pred_steps[1] + 1)
        time_steps = interval

        out_channels = n_output_data * len(out_steps)

    assert len(time_steps) == 2

def global_args(parser, arg_file=None, prog_func=None):
    import torch

    if arg_file is None:
        import sys
        argv = sys.argv[1:]
    else:
        argv = ['--load-from-file', arg_file]

    global progress_fwd
    progress_fwd = prog_func
    args, unknown = parser.parse_known_args(argv)

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

    global n_output_data
    if n_target_data > 0:
        n_output_data = n_target_data

    global min_bounds, max_bounds
    if len(min_bounds) == 1:
        min_bounds = [min_bounds[0] for i in range(n_output_data)]
    if len(max_bounds) == 1:
        max_bounds = [max_bounds[0] for i in range(n_output_data)]

    assert len(min_bounds) == n_output_data
    assert len(max_bounds) == n_output_data

    if all('.json' in data_name for data_name in data_names) and (lstm_steps or channel_steps):
        print('Warning: Each input file defined in your ".json" files will be considered individually.'
              ' This means the defined timesteps will not go beyond each files\' boundary.')

    return args


def set_common_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data-root-dir', type=str, default='../data/',
                            help="Root directory containing the climate datasets")
    arg_parser.add_argument('--mask-dir', type=str, default='masks/', help="Directory containing the mask datasets")
    arg_parser.add_argument('--log-dir', type=str, default='logs/', help="Directory where the log files will be stored")
    arg_parser.add_argument('--data-names', type=str_list, default='train.nc',
                            help="Comma separated list of netCDF files (climate dataset) or JSON files"
                                 " containing a list of paths to netCDF files for training/infilling")
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
    arg_parser.add_argument('--vae-zdim', type=int, default=0, help="Use VAE with latent space dimension")
    arg_parser.add_argument('--channel-steps', type=int_list, default=None,
                            help="Comma separated number of considered sequences for channeled memory:"
                                 "past_steps,future_steps")
    arg_parser.add_argument('--lstm-steps', type=int_list, default=None,
                            help="Comma separated number of considered sequences for lstm: past_steps,future_steps")
    arg_parser.add_argument('--gru-steps', type=int_list, default=None,
                            help="Comma separated number of considered sequences for gru: past_steps,future_steps")
    arg_parser.add_argument('--pred-steps', type=int_list, default=[0,0],
                            help="Comma separated number of considered sequences for pred: past_steps,future_steps")
    arg_parser.add_argument('--encoding-layers', type=int_list, default='3',
                            help="Number of encoding layers in the CNN")
    arg_parser.add_argument('--pooling-layers', type=int_list, default='0', help="Number of pooling layers in the CNN")
    arg_parser.add_argument('--conv-factor', type=int, default=None, help="Number of channels in the deepest layer")
    arg_parser.add_argument('--weights', type=str, default=None, help="Initialization weight")
    arg_parser.add_argument('--steady-masks', type=str_list, default=None,
                            help="Comma separated list of netCDF files containing a single mask to be applied "
                                 "to all timesteps. The number of steady-masks must be the same as n-output-data")
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
    arg_parser.add_argument('--n-output-data', type=int, default=1, help="Number of output data")
    arg_parser.add_argument('--dataset-name', type=str, default=None, help="Name of the dataset for format checking")
    arg_parser.add_argument('--min-bounds', type=float_list, default="-inf",
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
    arg_parser.add_argument('--save-model-interval', type=int, default=50000,
                            help="Iteration step interval at which the model should be saved")
    arg_parser.add_argument('--n-final-models', type=int, default=1,
                            help="Number of final models to be saved")
    arg_parser.add_argument('--final-models-interval', type=int, default=1000,
                            help="Iteration step interval at which the final models should be saved")
    arg_parser.add_argument('--loss-criterion', type=str, default="l1-hole",
                            help="Index/string defining the loss function (inpainting/l1-hole/l1-valid/etc.)")
    arg_parser.add_argument('--eval-timesteps', type=int_list, default=None,
                            help="Sample indices for which a snapshot is created at each iter defined by log-interval")
    arg_parser.add_argument('-f', '--load-from-file', type=str, action=LoadFromFile,
                            help="Load all the arguments from a text file")
    arg_parser.add_argument('--vlim', type=lim_list, default=None,
                            help="Comma separated list of vmin,vmax values for the color scale of the snapshot images")
    arg_parser.add_argument('--lambda-loss', type=key_value_list, default=None,
                            help="Comma separated list of lambda factors (key) followed by their corresponding values."
                                 "Overrides the loss_criterion pre-setting")
    arg_parser.add_argument('--val-metrics', type=str_list, default=None,
                            help="Comma separated list of metrics that are evaluated on the validation dataset "
                                 "at each log-interval iteration")
    arg_parser.add_argument('--tensor-plots', type=str_list, default="",
                            help="Comma separated list of 2D plots to be added to tensorboard "
                                 "(error, distribution, correlation)")
    arg_parser.add_argument('--early-stopping-delta', type=float, default=1e-5,
                            help="Mean relative delta of the val loss used for the termination criterion")
    arg_parser.add_argument('--early-stopping-patience', type=int, default=10,
                            help="Number of log-interval iterations used for the termination criterion")
    arg_parser.add_argument('--n-iters-val', type=int, default=1,
                            help="Number of batch iterations used to average the validation loss")

    args = global_args(arg_parser, arg_file)
    set_steps()

    global passed_args
    passed_args = get_passed_arguments(args, arg_parser)

    global early_stopping
    if ('early_stopping_delta' in passed_args.keys()) or ('early_stopping_patience' in passed_args.keys()):
        early_stopping = True
    else:
        early_stopping = False

    if globals()["val_names"] is None:
        globals()["val_names"] = globals()["data_names"].copy()

    set_lambdas()


def set_evaluate_args(arg_file=None, prog_func=None):
    arg_parser = set_common_args()
    arg_parser.add_argument('--model-dir', type=str, default='snapshots/ckpt/', help="Directory of the trained models")
    arg_parser.add_argument('--model-names', type=str_list, default='final.pth', help="Model names")
    arg_parser.add_argument('--evaluation-dirs', type=str_list, default='evaluation/',
                            help="Directory where the output files will be stored")
    arg_parser.add_argument('--eval-names', type=str_list, default='output',
                            help="Prefix used for the output filenames")
    arg_parser.add_argument('--use-train-stats', action='store_true',
                            help="Use mean and std from training data for normalization")
    arg_parser.add_argument('--n-evaluations', type=int, default=1, help="Number of evaluations")
    arg_parser.add_argument('--create-graph', action='store_true', help="Create a Tensorboard graph of the NN")
    arg_parser.add_argument('--plot-results', type=int_list, default=[],
                            help="Create plot images of the results for the comma separated list of time indices")
    arg_parser.add_argument('--partitions', type=int, default=1,
                            help="Split the climate dataset into several partitions along the time coordinate")
    arg_parser.add_argument('--maxmem', type=int, default=None,
                            help="Maximum available memory in MB (overwrite partitions parameter)")
    arg_parser.add_argument('--time-freq', type=str, default=None,
                            help="Time frequency for pred-steps option (only for D,H,M,S,etc.)")
    arg_parser.add_argument('--split-outputs', type=str, default="all", const=None, nargs='?',
                            help="Split the outputs according to members and/or partitions")
    arg_parser.add_argument('-f', '--load-from-file', type=str, action=LoadFromFile,
                            help="Load all the arguments from a text file")
    global_args(arg_parser, arg_file, prog_func)
    set_steps(evaluate=True)
    assert len(eval_names) == n_output_data
    globals()["model_names"] *= globals()["n_evaluations"]
