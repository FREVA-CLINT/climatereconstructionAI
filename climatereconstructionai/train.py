import os

import torch
import torch.nn as nn
import torch.multiprocessing
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from climatereconstructionai.loss.gan_loss import DiscriminatorLoss, GeneratorLoss
from climatereconstructionai.model.discriminator import Discriminator
from climatereconstructionai.model.generator import Generator
from . import config as cfg
from .loss.get_loss import get_loss
from .loss.hole_loss import HoleLoss
from .loss.inpainting_loss import InpaintingLoss
from .model.net import PConvLSTM
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
    seed_size = 2

    # define network model
    if len(cfg.image_sizes) > 1:
        generator = Generator(img_size=cfg.image_sizes[0], in_channels=2 * cfg.prev_next_steps + 1, seed_size=seed_size)
        discriminator = Discriminator(img_size=cfg.image_sizes[0],
                                      in_channels=2 * cfg.prev_next_steps + 1).to(cfg.device)
    else:
        generator = Generator(img_size=cfg.image_sizes[0], in_channels=2 * cfg.prev_next_steps + 1, seed_size=seed_size)
        discriminator = Discriminator(img_size=cfg.image_sizes[0],
                                      in_channels=2 * cfg.prev_next_steps + 1).to(cfg.device)

    # define learning rate
    if cfg.finetune:
        lr = cfg.lr_finetune
        generator.freeze_enc_bn = True
        discriminator.freeze_enc_bn = True
    else:
        lr = cfg.lr

    # define optimizer and loss functions
    generator_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, generator.parameters()), lr=lr,
                                           betas=(0.5, 0.99))
    discriminator_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=lr,
                                               betas=(0.5, 0.99))

    if cfg.loss_criterion == 0:
        lambda_dict = cfg.LAMBDA_DICT_GAN
    elif cfg.loss_criterion == 2:
        lambda_dict = cfg.LAMBDA_DICT_IMG_INPAINTING2
    else:
        raise ValueError("Unknown loss_criterion value.")

    # define start point
    start_iter = 0
    if cfg.resume_iter:
        # load generator
        start_iter = load_ckpt('{}/ckpt/generator_{}.pth'.format(cfg.snapshot_dir, cfg.resume_iter),
                               [('model', generator)], cfg.device, [('optimizer', generator_optimizer)])
        # load discriminator
        load_ckpt('{}/ckpt/discriminator_{}.pth'.format(cfg.snapshot_dir, cfg.resume_iter),
                  [('model', discriminator)], cfg.device, [('optimizer', discriminator_optimizer)])
        for param_group in generator_optimizer.param_groups:
            param_group['lr'] = lr
        for param_group in discriminator_optimizer.param_groups:
            param_group['lr'] = lr
        print('Starting from iter ', start_iter)

    if cfg.multi_gpus:
        generator = torch.nn.DataParallel(generator)
        discriminator = torch.nn.DataParallel(discriminator)

    real_label = 1
    fake_label = 0

    criterion = nn.BCELoss()

    pbar = tqdm(range(start_iter, cfg.max_iter))
    for i in pbar:
        image, mask, gt, rea_images, rea_masks, rea_gts = [x.to(cfg.device) for x in next(iterator_train)]
        noise = torch.randn(cfg.batch_size, seed_size, 9, 9).to(cfg.device)

        pbar.set_description("lr = {:.1e}".format(generator_optimizer.param_groups[0]['lr']))

        # train model
        generator.train()
        discriminator.train()

        label = torch.full((cfg.batch_size, 1), real_label, dtype=torch.float, device=cfg.device)
        discriminator.zero_grad()

        # train with gt
        discr_gt = discriminator(gt[:, 0, :, :, :])
        discriminator_loss_real = criterion(discr_gt, label)
        discriminator_loss_real.backward()
        d_x = discr_gt.mean()

        #train with fake output
        label.fill_(fake_label)
        output = generator(noise)
        discr_output = discriminator(output.detach())
        discriminator_loss_fake = criterion(discr_output, label)
        discriminator_loss_fake.backward()
        d_g_z1 = discr_output.mean()

        discriminator_loss = discriminator_loss_real + discriminator_loss_fake
        discriminator_optimizer.step()

        discr_output = discriminator(output)
        label.fill_(real_label)
        generator_loss = criterion(discr_output, label)
        generator_loss.backward()
        d_g_z2 = output.mean()
        generator_optimizer.step()

        if cfg.log_interval and (i + 1) % cfg.log_interval == 0:
            writer.add_scalar('loss_gen', generator_loss, i + 1)
            writer.add_scalar('loss_dis', discriminator_loss, i + 1)
            writer.add_scalar('d_g_z1', d_g_z1, i + 1)
            writer.add_scalar('d_g_z2', d_g_z2, i + 1)
            generator.eval()

            # create snapshot image
            if cfg.save_snapshot_image:
                generator.eval()
                create_snapshot_image(generator, dataset_val, '{:s}/images/iter_{:d}'.format(cfg.snapshot_dir, i + 1))

        if (i + 1) % cfg.save_model_interval == 0 or (i + 1) == cfg.max_iter:
            save_ckpt('{:s}/ckpt/generator_{:d}.pth'.format(cfg.snapshot_dir, i + 1),
                      [('model', generator)], [('optimizer', generator_optimizer)], i + 1)
            save_ckpt('{:s}/ckpt/discriminator_{:d}.pth'.format(cfg.snapshot_dir, i + 1),
                      [('model', discriminator)], [('optimizer', discriminator_optimizer)], i + 1)

    writer.close()


if __name__ == "__main__":
    train()
