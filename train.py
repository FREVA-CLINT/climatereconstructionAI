import argparse
import numpy as np
import os
import torch
from tensorboardX import SummaryWriter
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm

import opt
import local_settings
from evaluation import evaluate
from loss import InpaintingLoss
from net import PConvUNetPercipitation, PConvUNetTemperature
from net import VGG16FeatureExtractor
from places2 import Places2
from util.io import load_ckpt
from util.io import save_ckpt

import torch.multiprocessing as mp


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
device = torch.device(local_settings.device)

if not os.path.exists(local_settings.save_dir):
    os.makedirs('{:s}/images'.format(local_settings.save_dir))
    os.makedirs('{:s}/ckpt'.format(local_settings.save_dir))

if not os.path.exists(local_settings.log_dir):
    os.makedirs(local_settings.log_dir)
writer = SummaryWriter(log_dir=local_settings.log_dir)

img_tf = transforms.Compose(
    [transforms.Normalize(mean=opt.MEAN, std=opt.STD)])
mask_tf = transforms.Compose(
    [transforms.ToTensor()])

dataset_train = Places2(local_settings.data_root_train_dir, local_settings.mask_train_dir, img_tf, mask_tf, 'train')
dataset_val = Places2(local_settings.data_root_train_dir, local_settings.mask_train_dir, img_tf, mask_tf, 'val')
dataset_test = Places2(local_settings.data_root_train_dir, local_settings.mask_train_dir, img_tf, mask_tf, 'test')


iterator_train = iter(data.DataLoader(
    dataset_train, batch_size=local_settings.batch_size,
    sampler=InfiniteSampler(len(dataset_train)),
    num_workers=local_settings.n_threads))

if local_settings.data_type == 'pr':
    model = PConvUNetPercipitation().to(device)
elif local_settings.data_type == 'tas':
    model = PConvUNetTemperature().to(device)

if local_settings.finetune:
    lr = local_settings.lr_finetune
    model.freeze_enc_bn = True
else:
    lr = local_settings.lr

start_iter = 0
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
criterion = InpaintingLoss(VGG16FeatureExtractor()).to(device)

if local_settings.resume:
    start_iter = load_ckpt(
        local_settings.resume_dir, [('model', model)], [('optimizer', optimizer)])
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('Starting from iter ', start_iter)

for i in tqdm(range(start_iter, local_settings.max_iter)):
    model.train()

    image, mask, gt = [x.to(device) for x in next(iterator_train)]

    output, _ = model(image, mask)
    loss_dict = criterion(image, mask, output, gt)

    loss = 0.0
    for key, coef in opt.LAMBDA_DICT.items():
        value = coef * loss_dict[key]
        loss += value
        if (i + 1) % local_settings.log_interval == 0:
            writer.add_scalar('loss_{:s}'.format(key), value.item(), i + 1)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (i + 1) % local_settings.save_model_interval == 0 or (i + 1) == local_settings.max_iter:
        save_ckpt('{:s}/ckpt/{:d}.pth'.format(local_settings.save_dir, i + 1),
                  [('model', model)], [('optimizer', optimizer)], i + 1)

    if (i + 1) % local_settings.vis_interval == 0:
        model.eval()
        #evaluate(model, dataset_val, device,
        #         '{:s}/images/test_{:d}.jpg'.format(local_settings.save_dir, i + 1))
        evaluate(model, dataset_test, device,
                 '{:s}/images/testing_{:d}.jpg'.format(local_settings.save_dir, i + 1))
writer.close()
