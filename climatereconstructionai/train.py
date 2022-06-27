import os

import torch
import torch.multiprocessing
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import config as cfg
from .loss.get_loss import get_loss
from .loss.hole_loss import HoleLoss
from .loss.inpainting_loss import InpaintingLoss
from .model.net import CRAINet
from .utils.evaluation import create_snapshot_image
from .utils.featurizer import VGG16FeatureExtractor
from .utils.io import load_ckpt, save_ckpt
from .utils.netcdfloader import NetCDFLoader, InfiniteSampler, load_steadymask


def train(arg_file=None):
    torch.multiprocessing.set_sharing_strategy('file_system')
    print("* Number of GPUs: ", torch.cuda.device_count())

    cfg.set_train_args(arg_file)

    for subdir in ("", "/images", "/ckpt"):
        outdir = cfg.snapshot_dir + subdir
        if not os.path.exists(outdir):
            os.makedirs(outdir)

    if not os.path.exists(cfg.log_dir):
        os.makedirs(cfg.log_dir)
    writer = SummaryWriter(log_dir=cfg.log_dir)

    if cfg.lstm_steps:
        time_steps = cfg.lstm_steps
    elif cfg.gru_steps:
        time_steps = cfg.gru_steps
    elif cfg.channel_steps:
        time_steps = cfg.channel_steps
    else:
        time_steps = 0

    # create data sets
    dataset_train = NetCDFLoader(cfg.data_root_dir, cfg.img_names, cfg.mask_dir, cfg.mask_names, 'train',
                                 cfg.data_types, time_steps)
    dataset_val = NetCDFLoader(cfg.data_root_dir, cfg.img_names, cfg.mask_dir, cfg.mask_names, 'val', cfg.data_types,
                               time_steps)
    iterator_train = iter(DataLoader(dataset_train, batch_size=cfg.batch_size,
                                     sampler=InfiniteSampler(len(dataset_train)),
                                     num_workers=cfg.n_threads))
    iterator_val = iter(DataLoader(dataset_val, batch_size=cfg.batch_size,
                                   sampler=InfiniteSampler(len(dataset_val)),
                                   num_workers=cfg.n_threads))

    steady_mask = load_steadymask(cfg.mask_dir, cfg.steady_mask, cfg.data_types[0], cfg.device)

    # define network model
    if len(cfg.image_sizes) > 1:
        model = CRAINet(img_size=cfg.image_sizes[0],
                        enc_dec_layers=cfg.encoding_layers[0],
                        pool_layers=cfg.pooling_layers[0],
                        in_channels=2 * cfg.channel_steps + 1,
                        out_channels=cfg.out_channels,
                        fusion_img_size=cfg.image_sizes[1],
                        fusion_enc_layers=cfg.encoding_layers[1],
                        fusion_pool_layers=cfg.pooling_layers[1],
                        fusion_in_channels=(len(cfg.image_sizes) - 1) * (2 * cfg.channel_steps + 1)).to(cfg.device)
    else:
        model = CRAINet(img_size=cfg.image_sizes[0],
                        enc_dec_layers=cfg.encoding_layers[0],
                        pool_layers=cfg.pooling_layers[0],
                        in_channels=2 * cfg.channel_steps + 1,
                        out_channels=cfg.out_channels).to(cfg.device)

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
        if cfg.loss_criterion == 0:
            lambda_dict = cfg.LAMBDA_DICT_IMG_INPAINTING
        elif cfg.loss_criterion == 2:
            lambda_dict = cfg.LAMBDA_DICT_IMG_INPAINTING2
        else:
            raise ValueError("Unknown loss_criterion value.")

    if cfg.lr_scheduler_patience is not None:
        lr_scheduler = ReduceLROnPlateau(optimizer, 'min', patience=cfg.lr_scheduler_patience)

    # define start point
    start_iter = 0
    if cfg.resume_iter:
        start_iter = load_ckpt(
            '{}/ckpt/{}.pth'.format(cfg.snapshot_dir, cfg.resume_iter), [('model', model)], cfg.device,
            [('optimizer', optimizer)])
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Starting from iter ', start_iter)

    if cfg.multi_gpus:
        model = torch.nn.DataParallel(model)

    pbar = tqdm(range(start_iter, cfg.max_iter))
    for i in pbar:

        pbar.set_description("lr = {:.1e}".format(optimizer.param_groups[0]['lr']))

        # train model
        model.train()
        image, mask, gt, fusion_image, fusion_mask, fusion_gt = [x.to(cfg.device) for x in next(iterator_train)]
        output = model(image, mask, fusion_image, fusion_mask)

        train_loss = get_loss(criterion, lambda_dict, mask, steady_mask, output, gt, writer, i, "train")

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if cfg.log_interval and (i + 1) % cfg.log_interval == 0:

            model.eval()
            image, mask, gt, fusion_image, fusion_mask, fusion_gt = [x.to(cfg.device) for x in next(iterator_val)]
            with torch.no_grad():
                output = model(image, mask, fusion_image, fusion_mask)
            val_loss = get_loss(criterion, lambda_dict, mask, steady_mask, output, gt, writer, i, "val")
            if cfg.lr_scheduler_patience is not None:
                lr_scheduler.step(val_loss)

            # create snapshot image
            if cfg.save_snapshot_image:
                model.eval()
                create_snapshot_image(model, dataset_val, '{:s}/images/iter_{:d}'.format(cfg.snapshot_dir, i + 1))

        if (i + 1) % cfg.save_model_interval == 0 or (i + 1) == cfg.max_iter:
            save_ckpt('{:s}/ckpt/{:d}.pth'.format(cfg.snapshot_dir, i + 1),
                      [('model', model)], [('optimizer', optimizer)], i + 1)

    writer.close()


if __name__ == "__main__":
    train()
