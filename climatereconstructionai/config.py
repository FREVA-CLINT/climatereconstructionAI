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
    arg_parser.add_argument('--data-types', type=str_list, default='tas')
    arg_parser.add_argument('--img-names', type=str_list, default='train.h5')
    arg_parser.add_argument('--mask-names', type=str_list, default=None)
    arg_parser.add_argument('--evaluation-dirs', type=str_list, default='evaluation/')
    arg_parser.add_argument('--snapshot-dirs', type=str_list, default='snapshots/')
    arg_parser.add_argument('--data-root-dir', type=str, default='../data/')
    arg_parser.add_argument('--mask-dir', type=str, default='masks/')
    arg_parser.add_argument('--device', type=str, default='cuda')
    arg_parser.add_argument('--prev-next', type=int, default=0)
    arg_parser.add_argument('--lstm-steps', type=int, default=0)
    arg_parser.add_argument('--prev-next-steps', type=int, default=0)
    arg_parser.add_argument('--encoding-layers', type=int_list, default='3')
    arg_parser.add_argument('--pooling-layers', type=int_list, default='0')
    arg_parser.add_argument('--image-sizes', type=int_list, default='72')
    return arg_parser

def set_train_args(arg_file=None):
    arg_parser = set_common_args()
    arg_parser.add_argument('--log-dir', type=str, default='logs/')
    arg_parser.add_argument('--resume-iter', type=int)
    arg_parser.add_argument('--batch-size', type=int, default=18)
    arg_parser.add_argument('--n-threads', type=int, default=64)
    arg_parser.add_argument('--finetune', action='store_true')
    arg_parser.add_argument('--lr', type=float, default=2e-4)
    arg_parser.add_argument('--lr-finetune', type=float, default=5e-5)
    arg_parser.add_argument('--max-iter', type=int, default=1000000)
    arg_parser.add_argument('--log-interval', type=int, default=None)
    arg_parser.add_argument('--save-snapshot-image', action='store_true')
    arg_parser.add_argument('--save-model-interval', type=int, default=50000)
    arg_parser.add_argument('--out-channels', type=int, default=1)
    arg_parser.add_argument('--loss-criterion', type=int, default=0)
    arg_parser.add_argument('--eval-timesteps', type=str, default="0,1,2,3,4")
    arg_parser.add_argument('--weights', type=str, default=None)
    arg_parser.add_argument('--channel-reduction-rate', type=int, default=1)
    arg_parser.add_argument('--attention', action='store_true')
    arg_parser.add_argument('--disable-skip-layers', action='store_true')
    global_args(arg_parser,arg_file)

def set_evaluate_args(arg_file=None):
    arg_parser = set_common_args()
    arg_parser.add_argument('--partitions', type=int, default=1)
    arg_parser.add_argument('--infill', type=str, default=None, choices=["infill","test"])
    arg_parser.add_argument('--create-images', type=str, default=None)
    arg_parser.add_argument('--create-video', action='store_true')
    arg_parser.add_argument('--create-report', action='store_true')
    arg_parser.add_argument('--eval-names', type=str_list, default='Output')
    arg_parser.add_argument('--eval-range', type=str_list, default=None)
    arg_parser.add_argument('--ts-range', type=str_list, default=None)
    arg_parser.add_argument('--out-channels', type=int, default=1)
    arg_parser.add_argument('--eval-threshold', type=float, default=None)
    arg_parser.add_argument('--smoothing-factor', type=int, default=1)
    arg_parser.add_argument('--channel-reduction-rate', type=int, default=1)
    arg_parser.add_argument('--fps', type=float, default=5)
    arg_parser.add_argument('--attention', action='store_true')
    arg_parser.add_argument('--disable-skip-layers', action='store_true')
    arg_parser.add_argument('--convert-to-netcdf', action='store_true')
    arg_parser.add_argument('--load-from-file', type=str, action=LoadFromFile)
    global_args(arg_parser,arg_file)
    globals()["weights"] = None
