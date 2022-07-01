import os

import torch
import torch.nn as nn
import torch.multiprocessing
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from climatereconstructionai.loss.gan_loss import DiscriminatorLoss, GeneratorLoss
from climatereconstructionai.model.discriminator import PytorchDiscriminator as Discriminator
from climatereconstructionai.model.generator import Pytorch64x64Generator as Generator
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

    # define network model
    if len(cfg.image_sizes) > 1:
        generator = Generator(img_size=cfg.image_sizes[0], in_channels=2 * cfg.prev_next_steps + 1, seed_size=cfg.seed_size).to(cfg.device)
        discriminator = Discriminator(img_size=cfg.image_sizes[0],
                                      in_channels=2 * cfg.prev_next_steps + 1).to(cfg.device)
    else:
        generator = Generator(img_size=cfg.image_sizes[0], in_channels=2 * cfg.prev_next_steps + 1, seed_size=cfg.seed_size).to(cfg.device)
        discriminator = Discriminator(img_size=cfg.image_sizes[0],
                                      in_channels=2 * cfg.prev_next_steps + 1).to(cfg.device)

    # define learning rate
    if cfg.finetune:
        lr_gen = cfg.lr_gen_finetune
        lr_discr = cfg.lr_discr_finetune
        generator.freeze_enc_bn = True
        discriminator.freeze_enc_bn = True
    else:
        lr_gen = cfg.lr_gen
        lr_discr = cfg.lr_discr
    # define optimizer and loss functions
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=lr_gen, betas=(0.5, 0.999))
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr_discr, betas=(0.5, 0.999))

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

    real_label = 1.
    fake_label = 0.

    criterion = nn.MSELoss()

    # Training Loop

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    pbar = tqdm(range(start_iter, cfg.max_iter))

    print("Starting Training Loop...")
    # For each batch in the dataloader
    for i in pbar:
        image, mask, gt, rea_images, rea_masks, rea_gts = [x.to(cfg.device) for x in next(iterator_train)]
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        discriminator.zero_grad()
        # Format batch
        label = torch.full((cfg.batch_size,), real_label, dtype=torch.float, device=cfg.device)
        # Forward pass real batch through D
        output = discriminator(gt[:, 0, :, :, :]).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(cfg.batch_size, cfg.seed_size, 1, 1, device=cfg.device)
        # Generate fake image batch with G
        fake = generator(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = discriminator(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        discriminator_optimizer.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        generator.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = discriminator(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        generator_optimizer.step()

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        if cfg.log_interval and (i + 1) % cfg.log_interval == 0:
            writer.add_scalar('loss_gen', errG, i + 1)
            writer.add_scalar('loss_dis', errD, i + 1)
            writer.add_scalar('d_g_z1', D_G_z1, i + 1)
            writer.add_scalar('d_g_z2', D_G_z2, i + 1)
            generator.eval()

            print('[%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (i, errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

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
"""
    pbar = tqdm(range(start_iter, cfg.max_iter))
    for i in pbar:
        image, mask, gt, rea_images, rea_masks, rea_gts = [x.to(cfg.device) for x in next(iterator_train)]
        noise = torch.randn(cfg.batch_size, seed_size, 1, 1).to(cfg.device)

        pbar.set_description("lr = {:.1e}".format(generator_optimizer.param_groups[0]['lr']))

        # train model
        generator.train()
        discriminator.train()

        label = torch.full((cfg.batch_size, 1, 1, 1), real_label, dtype=torch.float, device=cfg.device)
        discriminator.zero_grad()

        # train with gt
        discr_gt = discriminator(gt[:, 0, :, :, :])
        discriminator_loss_real = criterion(discr_gt, label)
        discriminator_loss_real.backward()
        d_x = discr_gt.mean().item()

        #train with fake output
        label.fill_(fake_label)
        output = generator(noise)
        discr_output = discriminator(output.detach())
        discriminator_loss_fake = criterion(discr_output, label)
        discriminator_loss_fake.backward()
        d_g_z1 = discr_output.mean().item()

        discriminator_loss = discriminator_loss_real + discriminator_loss_fake
        discriminator_optimizer.step()

        discr_output = discriminator(output)
        label.fill_(real_label)
        generator_loss = criterion(discr_output, label)
        generator_loss.backward()
        d_g_z2 = discr_output.mean().item()
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
"""

if __name__ == "__main__":
    train()
