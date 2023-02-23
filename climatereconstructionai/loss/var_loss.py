import torch
import torch.nn.functional as F
from torch import nn
from .utils import conv_variance


class VarLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()


    def forward(self, mask, output, gt):
        loss_dict = {
            'valid': 0.0,
            'valid_std': 0.0
        } 

        # calculate loss for all channels
        for channel in range(output.shape[1]):
            # only select first channel

            mask_ch = torch.unsqueeze(mask[:, channel, :, :], dim=1)
            gt_ch = torch.unsqueeze(gt[:, channel, :, :], dim=1)
            output_ch = torch.unsqueeze(output[:, channel, :, :], dim=1)


            # define different loss functions from output and output_comp
            loss_dict['valid_std'] += self.l1(conv_variance(output_ch), conv_variance(gt_ch))
            loss_dict['valid'] += self.l1(mask_ch * output_ch, mask_ch * gt_ch)

            if loss_dict['valid_std'].isnan().sum()>0:
                pass
        return loss_dict