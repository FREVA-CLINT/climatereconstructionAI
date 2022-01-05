import os
import torch
import sys

sys.path.append('./')

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.net import PConvLSTM
from utils.featurizer import VGG16FeatureExtractor
from utils.io import load_ckpt, save_ckpt
from utils.netcdfloader import NetCDFLoader, InfiniteSampler
from utils.evaluation import create_snapshot_image
from model.loss import InpaintingLoss, HoleLoss
import config as cfg

cfg.set_train_args()

if not os.path.exists(cfg.snapshot_dir):
    os.makedirs('{:s}/images'.format(cfg.snapshot_dir))
    os.makedirs('{:s}/ckpt'.format(cfg.snapshot_dir))

if not os.path.exists(cfg.log_dir):
    os.makedirs(cfg.log_dir)
writer = SummaryWriter(log_dir=cfg.log_dir)

# create data sets
dataset_train = NetCDFLoader(cfg.data_root_dir, cfg.img_names, cfg.mask_dir, cfg.mask_names, 'train', cfg.data_types,
                             cfg.lstm_steps, cfg.prev_next_steps)
dataset_val = NetCDFLoader(cfg.data_root_dir, cfg.img_names, cfg.mask_dir, cfg.mask_names, 'val', cfg.data_types,
                           cfg.lstm_steps, cfg.prev_next_steps)
iterator_train = iter(DataLoader(dataset_train, batch_size=cfg.batch_size,
                                 sampler=InfiniteSampler(len(dataset_train)),
                                 num_workers=cfg.n_threads))

# define network model
lstm = True
if cfg.lstm_steps == 0:
    lstm = False

model = PConvLSTM(radar_img_size=cfg.image_sizes[0],
                  radar_enc_dec_layers=cfg.encoding_layers[0],
                  radar_pool_layers=cfg.pooling_layers[0],
                  radar_in_channels=2*cfg.prev_next_steps + 1,
                  radar_out_channels=cfg.out_channels,
                  rea_img_sizes=cfg.image_sizes[1:],
                  rea_enc_layers=cfg.encoding_layers[1:],
                  rea_pool_layers=cfg.pooling_layers[1:],
                  rea_in_channels=(len(cfg.image_sizes) - 1) * [2*cfg.prev_next_steps + 1],
                  lstm=lstm).to(cfg.device)

# define learning rate
if cfg.finetune:
    lr = cfg.lr_finetune
    model.freeze_enc_bn = True
else:
    lr = cfg.lr

# define optimizer and loss functions
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
if cfg.loss_criterion == 1:
    criterion = HoleLoss().to(cfg.device)
    lambda_dict = cfg.LAMBDA_DICT_HOLE
else:
    criterion = InpaintingLoss(VGG16FeatureExtractor()).to(cfg.device)
    lambda_dict = cfg.LAMBDA_DICT_IMG_INPAINTING


# define start point
start_iter = 0
if cfg.resume_iter:
    start_iter = load_ckpt(
        '{}/ckpt/{}.pth'.format(cfg.snapshot_dir, cfg.resume_iter), [('model', model)], cfg.device, [('optimizer', optimizer)])
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('Starting from iter ', start_iter)

for i in tqdm(range(start_iter, cfg.max_iter)):
    # train model
    model.train()
    image, mask, gt, rea_images, rea_masks, rea_gts = [x for x in next(iterator_train)]
    image = image.to(cfg.device)
    mask = mask.to(cfg.device)
    gt = gt.to(cfg.device)
    if rea_images:
        rea_images = rea_images.to(cfg.device)
        rea_masks = rea_masks.to(cfg.device)
        rea_gts = rea_gts.to(cfg.device)
    output = model(image, mask, rea_images, rea_masks)

    # calculate loss function and apply backpropagation
    loss_dict = criterion(mask[:, cfg.lstm_steps, cfg.gt_channels, :, :],
                          output[:, cfg.lstm_steps, :, :, :],
                          gt[:, cfg.lstm_steps, cfg.gt_channels, :, :])
    loss = 0.0
    for key, factor in lambda_dict.items():
        value = factor * loss_dict[key]
        loss += value
        if cfg.log_interval and (i + 1) % cfg.log_interval == 0:
            writer.add_scalar('loss_{:s}'.format(key), value.item(), i + 1)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # save checkpoint
    if (i + 1) % cfg.save_model_interval == 0 or (i + 1) == cfg.max_iter:
        save_ckpt('{:s}/ckpt/{:d}.pth'.format(cfg.snapshot_dir, i + 1),
                  [('model', model)], [('optimizer', optimizer)], i + 1)

    # create snapshot image
    if cfg.save_snapshot_image and (i + 1) % cfg.log_interval == 0:
        model.eval()
        create_snapshot_image(model, dataset_val, '{:s}/images/iter_{:d}'.format(cfg.snapshot_dir, i + 1))

writer.close()
