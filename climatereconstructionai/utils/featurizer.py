import torch
from torch import nn
from torchvision import models


class VGG16FeatureExtractor(nn.Module):
    def __init__(self, device):
        super().__init__()

        vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.enc_1 = nn.Sequential(*vgg16.features[:5]).to(device)
        self.enc_2 = nn.Sequential(*vgg16.features[5:10]).to(device)
        self.enc_3 = nn.Sequential(*vgg16.features[10:17]).to(device)

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [torch.cat([image] * 3, 1)]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]
