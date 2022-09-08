import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import config as cfg
from ..utils.masked_batchnorm import MaskedBatchNorm2d
from ..utils.weights import weights_init


def bound_pad(input, padding):
    input = F.pad(input, (0, 0, padding[2], 0), "constant", float(input[:, :, 0, :].mean()))
    input = F.pad(input, (0, 0, 0, padding[3]), "constant", float(input[:, :, -1, :].mean()))
    input = F.pad(input, (padding[0], padding[1], 0, 0), mode="circular")

    return input


class PConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, padding, dilation, groups, bias, activation, bn):
        super().__init__()

        self.padding = 2 * padding
        if cfg.global_padding:
            self.trans_pad = bound_pad
        else:
            self.trans_pad = F.pad

        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel, stride, 0, dilation, groups, bias)
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel, stride, 0, dilation, groups, False)

        if cfg.weights:
            self.input_conv.apply(weights_init(cfg.weights, random_seed=cfg.weights_random_seed))
        torch.nn.init.constant_(self.mask_conv.weight, 1.0)

        if activation:
            self.activation = activation
        if bn:
            if cfg.masked_bn:
                self.bn = MaskedBatchNorm2d(out_channels)
            else:
                self.bn = nn.BatchNorm2d(out_channels)

        # exclude mask gradients from backpropagation
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, input, mask):

        pad_input = self.trans_pad(input, self.padding)
        pad_mask = self.trans_pad(mask, self.padding)

        output = self.input_conv(pad_input * pad_mask)

        if self.input_conv.bias is not None:
            output_bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(output)
        else:
            output_bias = torch.zeros_like(output)

        with torch.no_grad():
            output_mask = self.mask_conv(pad_mask)

        no_update_holes = output_mask == 0
        mask_sum = output_mask.masked_fill_(no_update_holes, 1.0)
        output_pre = (output - output_bias) / mask_sum + output_bias
        output = output_pre.masked_fill_(no_update_holes, 0.0)

        new_mask = torch.ones_like(output)
        new_mask = new_mask.masked_fill_(no_update_holes, 0.0)

        if hasattr(self, 'bn'):
            if cfg.masked_bn:
                output = self.bn(output, new_mask)
            else:
                output = self.bn(output)

        if hasattr(self, 'activation'):
            output = self.activation(output)

        return output, new_mask
