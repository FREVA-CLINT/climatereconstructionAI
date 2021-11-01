import argparse
import numpy as np
import os
import torch
import opt
from tensorboardX import SummaryWriter
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm
from loss import InpaintingLoss
from net import PConvUNetPrecipitation, PConvUNetTemperature
from net import VGG16FeatureExtractor
from util.io import load_ckpt, save_ckpt

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--data-type', type=str, default='pr')
arg_parser.add_argument('--log-dir', type=str, default='logs/')
arg_parser.add_argument('--snapshot-dir', type=str, default='snapshots/')
arg_parser.add_argument('--data-root-dir', type=str, default='../data/radolan-complete-scaled/')
arg_parser.add_argument('--mask-dir', type=str, default='masks/single_radar_fail.h5')
arg_parser.add_argument('--resume-dir', type=str, default='snapshots/')
arg_parser.add_argument('--device', type=str, default='cpu')
arg_parser.add_argument('--batch-size', type=str, default=4)
arg_parser.add_argument('--n-threads', type=str, default=64)
arg_parser.add_argument('--finetune', type=str, default=False)
arg_parser.add_argument('--lr', type=str, default=2e-4)
arg_parser.add_argument('--lr-finetune', type=str, default=5e-5)
arg_parser.add_argument('--resume', type=str, default=False)
arg_parser.add_argument('--prev-next', type=str, default=True)
arg_parser.add_argument('--max-iter', type=str, default=100000)
arg_parser.add_argument('--log-interval', type=str, default=10000)
arg_parser.add_argument('--save-model-interval', type=str, default=10000)
args = arg_parser.parse_args()

if args.prev_next:
    from netcdfloader import PrevNextImageNetCDFLoader as NetCDFloader
else:
    from netcdfloader import SingleImageNetCDFLoader as NetCDFloader


class InfiniteSampler(data.sampler.Sampler):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __iter__(self):
        return iter(self.loop())

    def __len__(self):
        return 2 ** 31

    def loop(self):
        i = 0
        order = np.random.permutation(self.num_samples)
        while True:
            yield order[i]
            i += 1
            if i >= self.num_samples:
                np.random.seed()
                order = np.random.permutation(self.num_samples)
                i = 0


parser = argparse.ArgumentParser()

torch.backends.cudnn.benchmark = True
device = torch.device(args.device)

if not os.path.exists(args.snapshot_dir):
    os.makedirs('{:s}/images'.format(args.snapshot_dir))
    os.makedirs('{:s}/ckpt'.format(args.snapshot_dir))

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
writer = SummaryWriter(log_dir=args.log_dir)

img_tf = transforms.Compose(
    [transforms.Normalize(mean=opt.MEAN, std=opt.STD)])
mask_tf = transforms.Compose(
    [transforms.ToTensor()])

dataset_train = NetCDFloader(args.data_root_dir, args.mask_dir, img_tf, mask_tf, 'train', args.data_type)
dataset_val = NetCDFloader(args.data_root_dir, args.mask_dir, img_tf, mask_tf, 'val', args.data_type)

iterator_train = iter(data.DataLoader(
    dataset_train, batch_size=args.batch_size,
    sampler=InfiniteSampler(len(dataset_train)),
    num_workers=args.n_threads))

if args.data_type == 'pr':
    model = PConvUNetPrecipitation().to(device)
elif args.data_type == 'tas':
    model = PConvUNetTemperature().to(device)

if args.finetune:
    lr = args.lr_finetune
    model.freeze_enc_bn = True
else:
    lr = args.lr

start_iter = 0
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
criterion = InpaintingLoss(VGG16FeatureExtractor()).to(device)

if args.resume:
    start_iter = load_ckpt(
        args.resume_dir, [('model', model)], [('optimizer', optimizer)])
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('Starting from iter ', start_iter)

for i in tqdm(range(start_iter, args.max_iter)):
    model.train()

    image, mask, gt = [x.to(device) for x in next(iterator_train)]

    output, _ = model(image, mask)
    loss_dict = criterion(image, mask, output, gt)

    loss = 0.0
    for key, coef in opt.LAMBDA_DICT.items():
        value = coef * loss_dict[key]
        loss += value
        if (i + 1) % args.log_interval == 0:
            writer.add_scalar('loss_{:s}'.format(key), value.item(), i + 1)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        save_ckpt('{:s}/ckpt/{:d}.pth'.format(args.snapshot_dir, i + 1),
                  [('model', model)], [('optimizer', optimizer)], i + 1)

writer.close()
