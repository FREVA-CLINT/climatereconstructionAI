import torch
import torch.nn as nn
import sys
import logging

from .. import config as cfg
from ..utils.weights import weights_init


class PConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, padding, dilation, groups, bias, activation, bn):
        super().__init__()

        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel, stride, padding, dilation, groups, bias)
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel, stride, padding, dilation, groups, False)

        if cfg.weights:
            self.input_conv.apply(weights_init(cfg.weights,random_seed=cfg.random_seed))
        torch.nn.init.constant_(self.mask_conv.weight, 1.0)

        if activation:
            self.activation = activation
        if bn:
            self.bn = nn.BatchNorm2d(out_channels)

        # exclude mask gradients from backpropagation
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, input, mask):
        output = self.input_conv(input * mask)

        if cfg.verbose > 1:
            logging.info("* size of the masks: {}".format(mask.size()))
            logging.info("* % of non-zeros in the masks: {:.5f}".format(100.*mask.count_nonzero().item()/mask.numel()))

        if self.input_conv.bias is not None:
            output_bias = (self.input_conv.bias).view(1, -1, 1, 1).expand_as(output)
        else:
            output_bias = torch.zeros_like(output)

        with torch.no_grad():
            output_mask = self.mask_conv(mask)

        no_update_holes = output_mask == 0
        mask_sum = output_mask.masked_fill_(no_update_holes, 1.0)
        output_pre = (output - output_bias) / mask_sum + output_bias
        output = output_pre.masked_fill_(no_update_holes, 0.0)

        new_mask = torch.ones_like(output)
        new_mask = new_mask.masked_fill_(no_update_holes, 0.0)

        if hasattr(self, 'bn'):
            output = self.bn(output)
        if hasattr(self, 'activation'):
            output = self.activation(output)

        return output, new_mask
