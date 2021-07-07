import argparse
import torch
from torchvision import transforms

import opt
#from evaluation3 import evaluate3
from places2 import Places2
from evaluation2 import evaluate2
from net import PConvUNet
from util.io import load_ckpt

parser = argparse.ArgumentParser()
# training options
parser.add_argument('--root', type=str, default='./data')
parser.add_argument('--mask_root', type=str, default='./mask')
parser.add_argument('--snapshot', type=str, default='')
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--partitions', type=int, default=1)
#parser.add_argument('--ensemble', type=int, default='')
args = parser.parse_args()

device = torch.device('cuda')

size = (args.image_size, args.image_size)
img_transform = transforms.Compose(
    [transforms.Resize(size=size), transforms.ToTensor(),
     transforms.Normalize(mean=opt.MEAN, std=opt.STD)])
print(img_transform)
mask_transform = transforms.Compose(
    [transforms.Resize(size=size), transforms.ToTensor()])

dataset_val = Places2(args.root, args.mask_root, img_transform, mask_transform, 'test')

model = PConvUNet().to(device)
load_ckpt(args.snapshot, [('model', model)])

model.eval()
evaluate2(model, dataset_val, device, 'result.jpg', args.partitions)
