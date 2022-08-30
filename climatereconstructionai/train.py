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
from .model.net import PConvLSTM
from .utils.evaluation import create_snapshot_image
from .utils.featurizer import VGG16FeatureExtractor
from .utils.io import load_ckpt, load_model, save_ckpt
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
        recurrent = True
        time_steps = cfg.lstm_steps
    elif cfg.gru_steps:
        recurrent = True
        time_steps = cfg.gru_steps
    elif cfg.prev_next_steps:
        recurrent = False
        time_steps = cfg.prev_next_steps
    else:
        recurrent = False
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
        model = PConvLSTM(radar_img_size=cfg.image_sizes[0],
                          radar_enc_dec_layers=cfg.encoding_layers[0],
                          radar_pool_layers=cfg.pooling_layers[0],
                          radar_in_channels=2 * cfg.prev_next_steps + 1,
                          radar_out_channels=cfg.out_channels,
                          rea_img_size=cfg.image_sizes[1],
                          rea_enc_layers=cfg.encoding_layers[1],
                          rea_pool_layers=cfg.pooling_layers[1],
                          rea_in_channels=(len(cfg.image_sizes) - 1) * (2 * cfg.prev_next_steps + 1),
                          recurrent=recurrent).to(cfg.device)
    else:
        model = PConvLSTM(radar_img_size=cfg.image_sizes[0],
                          radar_enc_dec_layers=cfg.encoding_layers[0],
                          radar_pool_layers=cfg.pooling_layers[0],
                          radar_in_channels=2 * cfg.prev_next_steps + 1,
                          radar_out_channels=cfg.out_channels,
                          recurrent=recurrent).to(cfg.device)

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
        ckpt_dict = load_ckpt('{}/ckpt/{}.pth'.format(cfg.snapshot_dir, cfg.resume_iter), cfg.device)
        start_iter = load_model(ckpt_dict, model, optimizer)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Starting from iter ', start_iter)

    if cfg.multi_gpus:
        model = torch.nn.DataParallel(model)

    i = cfg.max_iter - (cfg.n_final_models - 1) * cfg.final_models_interval
    final_models = range(i, cfg.max_iter + 1, cfg.final_models_interval)

    savelist = []
    pbar = tqdm(range(start_iter, cfg.max_iter))
    for i in pbar:

        n_iter = i + 1
        pbar.set_description("lr = {:.1e}".format(optimizer.param_groups[0]['lr']))

        # train model
        model.train()
        image, mask, gt, rea_images, rea_masks, rea_gts = [x.to(cfg.device) for x in next(iterator_train)]
        output = model(image, mask, rea_images, rea_masks)

        train_loss = get_loss(criterion, lambda_dict, mask, steady_mask, output, gt, writer, i, "train")

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if cfg.log_interval and n_iter % cfg.log_interval == 0:

            model.eval()
            image, mask, gt, rea_images, rea_masks, rea_gts = [x.to(cfg.device) for x in next(iterator_val)]
            with torch.no_grad():
                output = model(image, mask, rea_images, rea_masks)
            val_loss = get_loss(criterion, lambda_dict, mask, steady_mask, output, gt, writer, i, "val")
            if cfg.lr_scheduler_patience is not None:
                lr_scheduler.step(val_loss)

            # create snapshot image
            if cfg.save_snapshot_image:
                model.eval()
                create_snapshot_image(model, dataset_val, '{:s}/images/iter_{:d}'.format(cfg.snapshot_dir, n_iter))

        if n_iter % cfg.save_model_interval == 0:
            save_ckpt('{:s}/ckpt/{:d}.pth'.format(cfg.snapshot_dir, n_iter), [(n_iter, model, optimizer)])

        if n_iter in final_models:
            savelist.append((n_iter, model, optimizer))

    writer.close()
    save_ckpt('{:s}/ckpt/final.pth'.format(cfg.snapshot_dir), savelist)


if __name__ == "__main__":
    train()
