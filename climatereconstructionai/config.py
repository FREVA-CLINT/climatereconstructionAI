import argparse

LAMBDA_DICT_IMG_INPAINTING = {
    'hole': 6.0, 'tv': 0.1, 'valid': 1.0, 'prc': 0.05, 'style': 120.0
}
LAMBDA_DICT_HOLE = {
    'hole': 1.0
}

PDF_BINS = [0, 0.01, 0.02, 0.1, 1, 2, 10, 100]


class LoadFromFile (argparse.Action):
    def __call__ (self, parser, namespace, values, option_string = None):
        parser.parse_args(open(values).read().split(), namespace)

def str_list(arg):
   return arg.split(',')

def int_list(arg):
   return list(map(int, arg.split(',')))

def global_args(parser,arg_file):

    import torch

    if arg_file is None:
        import sys
        argv = sys.argv[1:]
    else:
        argv = ["--load-from-file",arg_file]

    args = parser.parse_args(argv)

    args_dict = vars(args)
    for arg in args_dict:
        globals()[arg] = args_dict[arg]

    torch.backends.cudnn.benchmark = True
    globals()[device] = torch.device(device)

    global skip_layers
    global gt_channels

    if disable_skip_layers:
        skip_layers = 0
    else:
        skip_layers = 1

    gt_channels = []
    for i in range(out_channels):
        gt_channels.append((i + 1) * prev_next_steps + i * (prev_next_steps + 1))

def set_common_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data-root-dir', type=str, default='../data/', help="Root directory containing the climate datasets")
    arg_parser.add_argument('--mask-dir', type=str, default='masks/', help="Directory containing the mask datasets")
    arg_parser.add_argument('--log-dir', type=str, default='logs/', help="Directory where the log files will be stored")
    arg_parser.add_argument('--img-names', type=str_list, default='train.nc', help="Comma separated list of netCDF files (climate dataset)")
    arg_parser.add_argument('--mask-names', type=str_list, default=None, help="Comma separated list of netCDF files (mask dataset). If None, it extracts the masks from the climate dataset")
    arg_parser.add_argument('--data-types', type=str_list, default='tas', help="Comma separated list of variable types, in the same order as img-names and mask-names")
    arg_parser.add_argument('--device', type=str, default='cuda', help="Device used by PyTorch (cuda or cpu)")
    arg_parser.add_argument('--prev-next', type=int, default=0, help="")
    arg_parser.add_argument('--lstm-steps', type=int, default=0, help="Number of considered sequences for lstm (0 = lstm module is disabled)")
    arg_parser.add_argument('--prev-next-steps', type=int, default=0, help="")
    arg_parser.add_argument('--encoding-layers', type=int_list, default='3', help="Number of encoding layers in the CNN")
    arg_parser.add_argument('--pooling-layers', type=int_list, default='0', help="Number of pooling layers in the CNN")
    arg_parser.add_argument('--image-sizes', type=int_list, default='72', help="Spatial size of the datasets (latxlon must be of shape NxN)")
    arg_parser.add_argument('--weights', type=str, default=None, help="Initialization weight")
    arg_parser.add_argument('--attention', action='store_true', help="Enable the attention module")
    arg_parser.add_argument('--channel-reduction-rate', type=int, default=1, help="Channel reduction rate for the attention module")
    arg_parser.add_argument('--disable-skip-layers', action='store_true', help="Disable the skip layers")
    arg_parser.add_argument('--disable-first-last-bn', action='store_true', help="Disable the batch normalization on the first and last layer")
    arg_parser.add_argument('--out-channels', type=int, default=1, help="Number of channels for the output image")
    return arg_parser

def set_train_args(arg_file=None):
    arg_parser = set_common_args()
    arg_parser.add_argument('--snapshot-dir', type=str, default='snapshots/', help="Parent directory of the training checkpoints and the snapshot images")
    arg_parser.add_argument('--resume-iter', type=int, help="Iteration step from which the training will be resumed")
    arg_parser.add_argument('--batch-size', type=int, default=18, help="Batch size")
    arg_parser.add_argument('--n-threads', type=int, default=64, help="Number of threads")
    arg_parser.add_argument('--finetune', action='store_true', help="Enable the fine tuning mode (use fine tuning parameterization and disable batch normalization")
    arg_parser.add_argument('--lr', type=float, default=2e-4, help="Learning rate")
    arg_parser.add_argument('--lr-finetune', type=float, default=5e-5, help="Learning rate for fine tuning")
    arg_parser.add_argument('--max-iter', type=int, default=1000000, help="Maximum number of iterations")
    arg_parser.add_argument('--log-interval', type=int, default=None, help="Iteration step interval at which a tensorboard summary log should be written")
    arg_parser.add_argument('--save-snapshot-image', action='store_true', help="Save evaluation images for the iteration steps defined in --log-interval")
    arg_parser.add_argument('--save-model-interval', type=int, default=50000, help="Iteration step interval at which the model should be saved")
    arg_parser.add_argument('--loss-criterion', type=int, default=0, help="Index defining the loss function (0=original from Liu et al., 1=MAE of the hole region)")
    arg_parser.add_argument('--eval-timesteps', type=str, default="0,1,2,3,4", help="Iteration steps for which an evaluation is performed")
    arg_parser.add_argument('--load-from-file', type=str, action=LoadFromFile, help="Load all the arguments from a text file")
    global_args(arg_parser,arg_file)

def set_evaluate_args(arg_file=None):
    arg_parser = set_common_args()
    arg_parser.add_argument('--model-dir', type=str, default='snapshots/ckpt/', help="Directory of the trained models")
    arg_parser.add_argument('--model-names', type=str_list, default='1000000.pth', help="Model names")
    arg_parser.add_argument('--dataset-name', type=str, default=None, help="Name of the dataset for format checking")
    arg_parser.add_argument('--evaluation-dirs', type=str_list, default='evaluation/', help="Directory where the output files will be stored")
    arg_parser.add_argument('--eval-names', type=str_list, default='output', help="Prefix used for the output filenames")
    arg_parser.add_argument('--infill', type=str, default="infill", choices=["infill","test"], help="Infill the climate dataset ('test' if mask order is irrelevant, 'infill' if mask order is relevant)")
    arg_parser.add_argument('--create-graph', action='store_true', help="Create a Tensorboard graph of the NN")
    arg_parser.add_argument('--original-network', action='store_true', help="Use the original network architecture (from Kadow et al.)")
    arg_parser.add_argument('--partitions', type=int, default=1, help="Split the climate dataset into several partitions along the time coordinate")
    arg_parser.add_argument('--maxmem', type=int, default=None, help="Maximum available memory in MB (overwrite partitions parameter)")
    arg_parser.add_argument('--load-from-file', type=str, action=LoadFromFile, help="Load all the arguments from a text file")
    global_args(arg_parser,arg_file)
