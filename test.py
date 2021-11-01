import argparse

import torch
from torchvision import transforms
import opt
from evaluator import Evaluator
from net import PConvUNetPrecipitation, PConvUNetTemperature
from util.io import load_ckpt

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--data-type', type=str, default='pr')
arg_parser.add_argument('--evaluation-dir', type=str, default='evaluation/radolan-pn/')
arg_parser.add_argument('--snapshot-dir', type=str, default='snapshots/precipitation/radolan-prev-next/ckpt/200000.pth')
arg_parser.add_argument('--data-root-dir', type=str, default='../data/radolan-prev-next/')
arg_parser.add_argument('--mask-dir', type=str, default='masks/single_radar_fail.h5')
arg_parser.add_argument('--device', type=str, default='cpu')
arg_parser.add_argument('--image-size', type=str, default=512)
arg_parser.add_argument('--partitions', type=str, default=2009)
arg_parser.add_argument('--infill', type=str, default=False)
arg_parser.add_argument('--prev-next', type=str, default=True)
args = arg_parser.parse_args()

if args.prev_next:
    from netcdfloader import PrevNextImageNetCDFLoader as NetCDFloader
else:
    from netcdfloader import SingleImageNetCDFLoader as NetCDFloader

evaluator = Evaluator(args.evaluation_dir, args.mask_dir, args.data_root_dir + 'test_large/', args.data_type)
device = torch.device(args.device)

size = (args.image_size, args.image_size)
img_transform = transforms.Compose(
    [transforms.Resize(size=size), transforms.ToTensor(),
     transforms.Normalize(mean=opt.MEAN, std=opt.STD)])
mask_transform = transforms.Compose(
    [transforms.Resize(size=size), transforms.ToTensor()])

if args.infill:
    split = 'infill'
else:
    split = 'test'
dataset_val = NetCDFloader(args.data_root_dir, args.mask_dir, img_transform, mask_transform, split, args.data_type)

if args.data_type == 'pr':
    model = PConvUNetPrecipitation().to(device)
elif args.data_type == 'tas':
    model = PConvUNetTemperature().to(device)

load_ckpt(args.snapshot_dir, [('model', model)])

model.eval()
evaluator.infill(model, dataset_val, device, args.partitions)