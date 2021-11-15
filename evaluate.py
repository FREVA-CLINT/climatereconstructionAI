import argparse

import torch
from evaluator import Evaluator
from PConvLSTM import PConvLSTM
from netcdfloader import PrevNextNetCDFDataLoader
from util.io import load_ckpt

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--data-type', type=str, default='tas')
arg_parser.add_argument('--evaluation-dir', type=str, default='evaluation/')
arg_parser.add_argument('--snapshot-dir', type=str, default='snapshots/')
arg_parser.add_argument('--data-root-dir', type=str, default='../data/')
arg_parser.add_argument('--mask-dir', type=str, default='masks/')
arg_parser.add_argument('--device', type=str, default='cuda')
arg_parser.add_argument('--partitions', type=int, default=1)
arg_parser.add_argument('--prev-next', type=int, default=0)
arg_parser.add_argument('--encoding-layers', type=int, default=3)
arg_parser.add_argument('--pooling-layers', type=int, default=0)
arg_parser.add_argument('--image-size', type=int, default=72)
arg_parser.add_argument('--infill', type=str, default=None)
arg_parser.add_argument('--create-images', type=str, default='2017-07-12-14:00,2017-07-12-14:00')
arg_parser.add_argument('--create-video', action='store_true')
arg_parser.add_argument('--create-report', action='store_true')

args = arg_parser.parse_args()

evaluator = Evaluator(args.evaluation_dir, args.mask_dir, args.data_root_dir + 'test_large/', args.data_type)
device = torch.device(args.device)


if args.infill:
    dataset_val = PrevNextNetCDFDataLoader(args.data_root_dir, args.mask_dir, args.infill, args.data_type, args.prev_next)

    model = PConvLSTM(image_size=args.image_size, num_enc_dec_layers=args.encoding_layers, num_pool_layers=args.pooling_layers,
                      num_in_channels=1 + 2 * args.prev_next).to(device)

    load_ckpt(args.snapshot_dir, [('model', model)], device)

    model.eval()
    evaluator.infill(model, dataset_val, device, args.partitions)

if args.create_images:
    start_date = args.create_images.split(',')[0]
    end_date = args.create_images.split(',')[1]
    create_video = False
    if args.create_video:
        create_video = True
    evaluator.create_evaluation_images('image.nc', create_video, start_date, end_date)
    evaluator.create_evaluation_images('gt.nc', create_video, start_date, end_date)
    evaluator.create_evaluation_images('output.nc', create_video, start_date, end_date)
    evaluator.create_evaluation_images('output_comp.nc', create_video, start_date, end_date)

if args.create_report:
    evaluator.create_evaluation_report()