import torch
import torch.nn as nn
import sys

from .utils import gram_matrix, total_variation_loss


class InpaintingLoss(nn.Module):
    def __init__(self, extractor=None):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.extractor = extractor

    def forward(self, mask, output, gt):
        loss_dict = {
            'hole': 0.0,
            'valid': 0.0,
            'prc': 0.0,
            'style': 0.0,
            'tv': 0.0
        }

        # create output_comp
        output_comp = mask * gt + (1 - mask) * output

        # calculate loss for all channels
        for channel in range(output.shape[1]):
            # only select first channel
            mask_ch = torch.unsqueeze(mask[:, channel, :, :], dim=1)
            gt_ch = torch.unsqueeze(gt[:, channel, :, :], dim=1)
            output_ch = torch.unsqueeze(output[:, channel, :, :], dim=1)
            output_comp_ch = torch.unsqueeze(output_comp[:, channel, :, :], dim=1)

            # define different loss functions from output and output_comp
            loss_dict['hole'] += self.l1((1 - mask_ch) * output_ch, (1 - mask_ch) * gt_ch)
            loss_dict['valid'] += self.l1(mask_ch * output_ch, mask_ch * gt_ch)

            # define different loss function from features from output and output_comp
            if self.extractor:
                feat_output = self.extractor(torch.cat([output_ch] * 3, 1))
                feat_output_comp = self.extractor(torch.cat([output_comp_ch] * 3, 1))
                feat_gt = self.extractor(torch.cat([gt_ch] * 3, 1))
                for i in range(len(feat_gt)):
                    loss_dict['prc'] += self.l1(feat_output[i], feat_gt[i])
                    loss_dict['prc'] += self.l1(feat_output_comp[i], feat_gt[i])
                    loss_dict['style'] += self.l1(gram_matrix(feat_output[i]),
                                                  gram_matrix(feat_gt[i]))
                    loss_dict['style'] += self.l1(gram_matrix(feat_output_comp[i]),
                                                  gram_matrix(feat_gt[i]))
            else:
                loss_dict['prc'] += self.l1(output_ch, gt_ch)
                loss_dict['prc'] += self.l1(output_comp_ch, gt_ch)

            loss_dict['tv'] += total_variation_loss(output_comp_ch)

        return loss_dict
