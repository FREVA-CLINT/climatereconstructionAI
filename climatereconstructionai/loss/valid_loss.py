import torch
from torch import nn


class ValidLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, data_dict):
        loss_dict = {
            'valid': 0.0
        }
        output = data_dict['output']
        gt = data_dict['gt']
        mask = data_dict['mask']

        # calculate loss for all channels
        for channel in range(output.shape[1]):
            # only select first channel
            mask_ch = torch.unsqueeze(mask[:, channel, :, :], dim=1)
            gt_ch = torch.unsqueeze(gt[:, channel, :, :], dim=1)
            output_ch = torch.unsqueeze(output[:, channel, :, :], dim=1)

            # define different loss functions from output and output_comp
            loss_dict['valid'] += self.l1(mask_ch * output_ch, mask_ch * gt_ch)
        return loss_dict