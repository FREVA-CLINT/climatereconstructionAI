import torch
from torch import nn

from .utils import total_variation_loss


class TotalVariationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, data_dict):
        loss_dict = {
            'tv': 0.0
        }

        output_comp = data_dict['comp']

        # calculate loss for all channels
        for channel in range(output_comp.shape[1]):
            output_comp_ch = torch.unsqueeze(output_comp[:, channel, :, :], dim=1)
            loss_dict['tv'] += total_variation_loss(output_comp_ch)
        return loss_dict
