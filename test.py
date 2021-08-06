import argparse
import torch
from torchvision import transforms

import opt
import local_settings
from places2 import Places2
from evaluation import evaluate
from net import PConvUNetPercipitation, PConvUNetTemperature
from util.io import load_ckpt

device = torch.device(local_settings.device)

size = (local_settings.image_size, local_settings.image_size)
img_transform = transforms.Compose(
    [transforms.Resize(size=size), transforms.ToTensor(),
     transforms.Normalize(mean=opt.MEAN, std=opt.STD)])
print(img_transform)
mask_transform = transforms.Compose(
    [transforms.Resize(size=size), transforms.ToTensor()])

if local_settings.infill:
    split = 'infill'
else:
    split = 'test'
dataset_val = Places2(local_settings.data_root_test_dir, local_settings.mask_test_dir, img_transform, mask_transform, split)

if local_settings.data_type == 'pr':
    model = PConvUNetPercipitation().to(device)
elif local_settings.data_type == 'tas':
    model = PConvUNetTemperature().to(device)

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=local_settings.lr)
load_ckpt(local_settings.snapshot_dir, [('model', model)], [('optimizer', optimizer)])

model.eval()
evaluate(model, dataset_val, device, 'result.jpg')
