import argparse
import os
import torch
from torchvision.utils import make_grid, save_image

import opt
from tensorboardX import SummaryWriter
from torch.utils import data
from tqdm import tqdm
from MultiChannelPConvUNet import MultiChannelPConvUNet
from PConvLSTM import VGG16FeatureExtractor
from loss import PrevNextInpaintingLoss, PrecipitationInpaintingLoss
from util.io import load_ckpt, save_ckpt
from netcdfloader import PrevNextNetCDFDataLoader as NetCDFDataLoader, InfiniteSampler

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--data-type', type=str, default='tas')
arg_parser.add_argument('--log-dir', type=str, default='logs/')
arg_parser.add_argument('--snapshot-dir', type=str, default='snapshots/')
arg_parser.add_argument('--data-root-dir', type=str, default='../data/')
arg_parser.add_argument('--mask-dir', type=str, default='masks/')
arg_parser.add_argument('--resume', type=str)
arg_parser.add_argument('--device', type=str, default='cuda')
arg_parser.add_argument('--batch-size', type=int, default=18)
arg_parser.add_argument('--n-threads', type=int, default=64)
arg_parser.add_argument('--finetune', action='store_true')
arg_parser.add_argument('--lr', type=float, default=2e-4)
arg_parser.add_argument('--lr-finetune', type=float, default=5e-5)
arg_parser.add_argument('--max-iter', type=int, default=1000000)
arg_parser.add_argument('--log-interval', type=int, default=1000)
arg_parser.add_argument('--save-model-interval', type=int, default=50000)
arg_parser.add_argument('--prev-next', type=int, default=0)
arg_parser.add_argument('--lstm-steps', type=int, default=0)
arg_parser.add_argument('--encoding-layers', type=int, default=3)
arg_parser.add_argument('--pooling-layers', type=int, default=0)
arg_parser.add_argument('--image-size', type=int, default=72)
args = arg_parser.parse_args()

parser = argparse.ArgumentParser()

torch.backends.cudnn.benchmark = True
device = torch.device(args.device)

def evaluate(model, dataset, device, filename):
    image, mask, gt = zip(*[dataset[i] for i in range(8)])
    image = torch.stack(image)
    mask = torch.stack(mask)
    gt = torch.stack(gt)

    with torch.no_grad():
        output, _ = model(image.to(device), mask.to(device))

    # get mid indexed element
    mid_index = torch.tensor([(image.shape[1] // 2)], dtype=torch.long).to(device)
    image = torch.index_select(image, dim=1, index=mid_index)
    gt = torch.index_select(gt, dim=1, index=mid_index)
    mask = torch.index_select(mask, dim=1, index=mid_index)

    output = output.to(torch.device(device))
    output_comp = mask * image + (1 - mask) * output

    grid = make_grid(
        torch.cat(((image), mask, (output),
                   (output_comp), (gt)), dim=0))
    save_image(grid, filename)

if not os.path.exists(args.snapshot_dir):
    os.makedirs('{:s}/images'.format(args.snapshot_dir))
    os.makedirs('{:s}/ckpt'.format(args.snapshot_dir))

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
writer = SummaryWriter(log_dir=args.log_dir)

# define data set + iterator
dataset_train = NetCDFDataLoader(args.data_root_dir, args.mask_dir, 'train', args.data_type, args.prev_next)
dataset_val = NetCDFDataLoader(args.data_root_dir, args.mask_dir, 'val', args.data_type, args.prev_next)
iterator_train = iter(data.DataLoader(dataset_train, batch_size=args.batch_size,
                                      sampler=InfiniteSampler(len(dataset_train)),
                                      num_workers=args.n_threads))

# define network model
model = MultiChannelPConvUNet(image_size=args.image_size,
                              num_enc_dec_layers=args.encoding_layers,
                              num_pool_layers=args.pooling_layers,
                              num_in_channels=2*args.prev_next + 1).to(device)

# define learning rate
if args.finetune:
    lr = args.lr_finetune
    model.freeze_enc_bn = True
else:
    lr = args.lr

# define optimizer and loss functions
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
criterion = PrevNextInpaintingLoss(VGG16FeatureExtractor()).to(device)

# define start point
start_iter = 0
if args.resume:
    start_iter = load_ckpt(
        args.resume, [('model', model)], device, [('optimizer', optimizer)])
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('Starting from iter ', start_iter)

for i in tqdm(range(start_iter, args.max_iter)):
    # train model
    model.train()
    image, mask, gt = [x.to(device) for x in next(iterator_train)]
    output, _ = model(image, mask)
    loss_dict = criterion(image, mask, output, gt, device)

    # calculate loss function and apply backpropagation
    loss = 0.0
    for key, coef in opt.LAMBDA_DICT_IMG_INPAINTING.items():
        value = coef * loss_dict[key]
        loss += value
        if (i + 1) % args.log_interval == 0:
            writer.add_scalar('loss_{:s}'.format(key), value.item(), i + 1)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # save checkpoint
    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        save_ckpt('{:s}/ckpt/{:d}.pth'.format(args.snapshot_dir, i + 1),
                  [('model', model)], [('optimizer', optimizer)], i + 1)

    # create snapshot image
    if (i + 1) % args.log_interval == 0:
        model.eval()
        evaluate(model, dataset_val, device,
                 '{:s}/images/test_{:d}.jpg'.format(args.snapshot_dir, i + 1))

writer.close()

