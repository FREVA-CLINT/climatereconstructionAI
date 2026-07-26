import torch
from torch import nn


class KLDivLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, data_dict):

        mu, logvar = data_dict['latent_dist']
        loss_dict = {'kldiv': -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())}

        return loss_dict
