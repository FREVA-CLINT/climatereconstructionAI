import torch
import torch.nn.functional as F
from torch import nn
from .utils import conv_variance


class GaussLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.Gauss = nn.GaussianNLLLoss()


    def forward(self, data_dict):
        loss_dict = {
            'gauss': 0.0
        } 

        output = data_dict['gauss']
        gt = data_dict['gt']

        loss_dict['gauss'] += self.Gauss(output[:,0],gt,output[:,1])

        return loss_dict