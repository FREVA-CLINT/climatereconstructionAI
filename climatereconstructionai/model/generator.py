import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, img_size, in_channels, seed_size):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.Conv2d(seed_size, img_size * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(img_size * 2),
            nn.ReLU(True),
            # state size. (ngf*8) x 9 x 9
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(img_size * 2, img_size, 3, 1, 1, bias=False),
            nn.BatchNorm2d(img_size),
            nn.ReLU(True),
            # state size. (ngf*4) x 18 x 18
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(img_size, img_size, 3, 1, 1, bias=False),
            nn.BatchNorm2d(img_size),
            nn.ReLU(True),
            # state size. (ngf*2) x 36 x 36
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(img_size, in_channels, 3, 1, 1, bias=False),
            nn.Tanh()
            # state size. (ngf) x 72 x 72
        )
        self.weights_init()

    def forward(self, input):
        return self.main(input)

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)