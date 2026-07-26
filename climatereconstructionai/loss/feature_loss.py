import torch
import torch.nn as nn

from .utils import gram_matrix

class FeatureLoss(nn.Module):
    def __init__(self, extractor, devices):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.extractor = {f'{device}': extractor(device) for device in devices}

    def forward(self, data_dict):
        loss_dict = {
            'prc': 0.0,
            'style': 0.0,
        }

        output_comp = data_dict['comp']
        output = data_dict['output']
        gt = data_dict['gt']

        extractor = self.extractor[str(output.device)]

        # calculate loss for all channels
        for channel in range(output.shape[1]):

            gt_ch = torch.unsqueeze(gt[:, channel, :, :], dim=1)
            output_ch = torch.unsqueeze(output[:, channel, :, :], dim=1)
            output_comp_ch = torch.unsqueeze(output_comp[:, channel, :, :], dim=1)

            # define different loss function from features from output and output_comp
            feat_output = extractor(output_ch)
            feat_output_comp = extractor(output_comp_ch)
            feat_gt = extractor(gt_ch)
            for i in range(len(feat_gt)):
                loss_dict['prc'] += self.l1(feat_output[i], feat_gt[i])
                loss_dict['prc'] += self.l1(feat_output_comp[i], feat_gt[i])
                loss_dict['style'] += self.l1(gram_matrix(feat_output[i]),
                                              gram_matrix(feat_gt[i]))
                loss_dict['style'] += self.l1(gram_matrix(feat_output_comp[i]),
                                              gram_matrix(feat_gt[i]))

        return loss_dict
