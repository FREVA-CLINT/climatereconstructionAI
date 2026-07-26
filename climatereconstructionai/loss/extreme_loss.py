import torch
import torch.nn as nn


class ExtremeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l2 = nn.MSELoss()
        self.sm = nn.Softmax(dim=0)

    def forward(self, data_dict):
        loss_dict = {
            '-extreme': 0.0,
            '+extreme': 0.0,
        }

        output = data_dict['output']
        gt = data_dict['gt']

        # calculate loss for all channels
        for channel in range(output.shape[1]):

            gt_ch = torch.unsqueeze(gt[:, channel, :, :], dim=1)
            output_ch = torch.unsqueeze(output[:, channel, :, :], dim=1)
            loss_dict['-extreme'] += self.l2(self.sm(-output_ch), self.sm(-gt_ch))
            loss_dict['+extreme'] += self.l2(self.sm(output_ch), self.sm(gt_ch))

        return loss_dict
