import argparse

import torch
from torchvision import transforms
import opt
from evaluator import Evaluator
from net import PConvLSTM
from netcdfloader import SingleNetCDFDataLoader
from util.io import load_ckpt

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--data-type', type=str, default='pr')
arg_parser.add_argument('--evaluation-dir', type=str, default='evaluation/radolan-pn/')
arg_parser.add_argument('--snapshot-dir', type=str, default='snapshots/precipitation/radolan-prev-next/ckpt/200000.pth')
arg_parser.add_argument('--data-root-dir', type=str, default='../data/radolan-complete-scaled/')
arg_parser.add_argument('--mask-dir', type=str, default='masks/single_radar_fail.h5')
arg_parser.add_argument('--device', type=str, default='cpu')
arg_parser.add_argument('--partitions', type=str, default=2009)
arg_parser.add_argument('--infill', type=str)
arg_parser.add_argument('--prev-next', type=int, default=1)
arg_parser.add_argument('--encoding-layers', type=int, default=4)
arg_parser.add_argument('--pooling-layers', type=int, default=3)
arg_parser.add_argument('--image-size', type=int, default=512)
args = arg_parser.parse_args()

evaluator = Evaluator(args.evaluation_dir, args.mask_dir, args.data_root_dir + 'test_large/', args.data_type)
device = torch.device(args.device)

if args.infill:
    split = 'infill'
else:
    split = 'test'
dataset_val = SingleNetCDFDataLoader(args.data_root_dir, args.mask_dir, split, args.data_type, args.prev_next)

model = PConvLSTM(image_size=args.image_size, encoding_layers=args.encoding_layers, pooling_layers=args.pooling_layers,
                  input_channels=1 + args.prev_next)

load_ckpt(args.snapshot_dir, [('model', model)])

model.eval()
evaluator.infill(model, dataset_val, device, args.partitions)