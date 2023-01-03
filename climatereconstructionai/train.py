import os

import torch
import torch.multiprocessing
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy
import numpy as np

from . import config as cfg
from .loss.get_loss import get_loss
from .loss.hole_loss import HoleLoss
from .loss.valid_loss import ValidLoss
from .loss.inpainting_loss import InpaintingLoss
from .model.net import CRAINet
from .utils.evaluation import create_snapshot_image
from .utils.featurizer import VGG16FeatureExtractor
from .utils.io import load_ckpt, load_model, save_ckpt
from .utils.netcdfloader import NetCDFLoader, InfiniteSampler, load_steadymask
from .utils.profiler import load_profiler


def train(arg_file=None):
    cfg.set_train_args(arg_file)

    print("* Number of GPUs: ", torch.cuda.device_count())
    torch.multiprocessing.set_sharing_strategy('file_system')

    np.random.seed(cfg.loop_random_seed)
    if cfg.cuda_random_seed is not None:
        torch.manual_seed(cfg.cuda_random_seed)

    if cfg.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False

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
    dataset_train = NetCDFLoader(cfg.data_root_dir, cfg.data_names, cfg.mask_dir, cfg.mask_names, 'train',
                                 cfg.data_types, time_steps)
    dataset_val = NetCDFLoader(cfg.data_root_dir, cfg.val_names, cfg.mask_dir, cfg.mask_names, 'val', cfg.data_types,
                               time_steps)
    iterator_train = iter(DataLoader(dataset_train, batch_size=cfg.batch_size,
                                     sampler=InfiniteSampler(len(dataset_train)),
                                     num_workers=cfg.n_threads))
    iterator_val = iter(DataLoader(dataset_val, batch_size=cfg.batch_size,
                                   sampler=InfiniteSampler(len(dataset_val)),
                                   num_workers=cfg.n_threads))

    steady_mask = load_steadymask(cfg.mask_dir, cfg.steady_masks, cfg.data_types, cfg.device)

    if cfg.target_data_indices == []:
        stat_target = None
    else:
        stat_target = {"mean": [dataset_train.img_mean[i] for i in cfg.target_data_indices],
                       "std": [dataset_train.img_std[i] for i in cfg.target_data_indices]}

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
                        fusion_in_channels=(len(cfg.image_sizes) - 1 - len(cfg.target_data_indices)) *
                        (2 * cfg.channel_steps + 1),
                        bounds=dataset_train.bounds).to(cfg.device)
    else:
        model = CRAINet(img_size=cfg.image_sizes[0],
                        enc_dec_layers=cfg.encoding_layers[0],
                        pool_layers=cfg.pooling_layers[0],
                        in_channels=2 * cfg.channel_steps + 1,
                        out_channels=cfg.out_channels,
                        bounds=dataset_train.bounds).to(cfg.device)

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
    elif cfg.loss_criterion == 3:
        criterion = ValidLoss().to(cfg.device)
        lambda_dict = cfg.LAMBDA_DICT_VALID
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

    prof = load_profiler(start_iter)

    if cfg.multi_gpus:
        model = torch.nn.DataParallel(model)

    i = cfg.max_iter - (cfg.n_final_models - 1) * cfg.final_models_interval
    final_models = range(i, cfg.max_iter + 1, cfg.final_models_interval)

    savelist = []
    pbar = tqdm(range(start_iter, cfg.max_iter))
    prof.start()
    for i in pbar:

        n_iter = i + 1
        lr_val = optimizer.param_groups[0]['lr']
        pbar.set_description("lr = {:.1e}".format(lr_val))

        # train model
        model.train()

        image, mask, gt = [x.to(cfg.device) for x in next(iterator_train)]
        output = model(image, mask)

        train_loss = get_loss(criterion, lambda_dict, mask, steady_mask, output, gt, writer, n_iter, "train")

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if cfg.log_interval and n_iter % cfg.log_interval == 0:

            model.eval()
            image, mask, gt = [x.to(cfg.device) for x in next(iterator_val)]
            with torch.no_grad():
                output = model(image, mask)
            val_loss = get_loss(criterion, lambda_dict, mask, steady_mask, output, gt, writer, n_iter, "val")

            writer.add_scalar('lr', lr_val, n_iter)

            if cfg.lr_scheduler_patience is not None:
                lr_scheduler.step(val_loss)

            # create snapshot image
            if cfg.save_snapshot_image:
                model.eval()
                create_snapshot_image(model, dataset_val, '{:s}/images/iter_{:d}'.format(cfg.snapshot_dir, n_iter))

        if n_iter % cfg.save_model_interval == 0:
            save_ckpt('{:s}/ckpt/{:d}.pth'.format(cfg.snapshot_dir, n_iter), stat_target,
                      [(str(n_iter), n_iter, model, optimizer)])

        if n_iter in final_models:
            savelist.append((str(n_iter), n_iter, copy.deepcopy(model), copy.deepcopy(optimizer)))
        prof.step()

    prof.stop()
    writer.close()
    save_ckpt('{:s}/ckpt/final.pth'.format(cfg.snapshot_dir), stat_target, savelist)


if __name__ == "__main__":
    train()
