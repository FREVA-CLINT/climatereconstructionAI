import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, radar_img_size=512, radar_enc_dec_layers=4, radar_pool_layers=4, radar_in_channels=1,
                 radar_out_channels=1,
                 rea_img_size=None, rea_enc_layers=None, rea_pool_layers=None, rea_in_channels=0,
                 recurrent=True):
        super().__init__()

        self.freeze_enc_bn = False
        self.net_depth = radar_enc_dec_layers + radar_pool_layers

        # define encoding layers
        net = []
        net.append(
            nn.Conv2d(
                2*radar_in_channels,
                radar_img_size // (2 ** (radar_enc_dec_layers - 1)), (7, 7), (1, 1), (3, 3)))
        net.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        for i in range(1, radar_enc_dec_layers):
            if i == radar_enc_dec_layers - 1:
                net.append(
                    nn.Conv2d(
                        radar_img_size // (2 ** (radar_enc_dec_layers - i)),
                        radar_img_size // (2 ** (radar_enc_dec_layers - i - 1)),
                        (3, 3), (2, 2), (1, 1)))
            else:
                net.append(nn.Conv2d(
                    radar_img_size // (2 ** (radar_enc_dec_layers - i)),
                    radar_img_size // (2 ** (radar_enc_dec_layers - i - 1)),
                    (5, 5), (2, 2), (2, 2)))
            net.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        # define encoding pooling layers
        for i in range(radar_pool_layers):
            net.append(nn.Conv2d(
                radar_img_size,
                radar_img_size,
                (3, 3), (2, 2), (1, 1)))
            net.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        net.append(nn.Linear(radar_img_size * ((radar_img_size // (2 ** (self.net_depth - 1))) ** 2), radar_img_size))
        net.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        net.append(nn.Linear(radar_img_size, 1))
        net.append(nn.Sigmoid())
        self.conv = nn.Sequential(*net)

    def forward(self, input, mask):
        h = torch.cat([input[:, 0, :, :, :], mask[:, 0, :, :, :]], dim=1)
        for net in self.conv:
            if isinstance(net, nn.Linear):
                h = torch.flatten(h, start_dim=1)
            h = net(h)
        return h
