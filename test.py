import argparse
import torch
from torchvision import transforms
import time

import opt
import local_settings
from evaluation2 import evaluate2
from places2 import Places2
from evaluation import evaluate
from net import PConvUNetPrecipitation, PConvUNetTemperature
from util.io import load_ckpt
start = time.time()

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
    model = PConvUNetPrecipitation().to(device)
elif local_settings.data_type == 'tas':
    model = PConvUNetTemperature().to(device)

load_ckpt(local_settings.snapshot_dir, [('model', model)])

model.eval()
evaluate2(model, dataset_val, device, local_settings.test_image_dir + 'result', local_settings.partitions)
end = time.time()
print(start - end)