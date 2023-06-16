import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import config as cfg
from ..utils.masked_batchnorm import MaskedBatchNorm2d
from ..utils.weights import weights_init


def sequence_to_batch(input):
    return torch.reshape(input, (-1, input.shape[2], input.shape[3], input.shape[4]))


def batch_to_sequence(input, batch_size):
    return torch.reshape(input,
                         (batch_size, cfg.n_recurrent_steps, input.shape[1], input.shape[2], input.shape[3]))


def bound_pad(input, padding):
    input = F.pad(input, (0, 0, padding[2], 0), "constant", 0.)
    input = F.pad(input, (0, 0, 0, padding[3]), "constant", 0.)
    input = F.pad(input, (padding[0], padding[1], 0, 0), mode="circular")

    return input


class PConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, img_size, kernel, stride, padding, dilation, groups, bias, activation,
                 bn):
        super().__init__()

        self.padding = 2 * padding
        if cfg.global_padding:
            self.trans_pad = bound_pad
        else:
            self.trans_pad = F.pad

        if cfg.lstm_steps:
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.img_size = img_size
            self.input_conv = nn.Conv2d(in_channels + out_channels, 4 * out_channels, kernel, (1, 1), 0, dilation,
                                        groups, bias)
            self.max_pool = nn.MaxPool2d(kernel, stride)
            self.Wci = nn.Parameter(torch.zeros(1, out_channels, *img_size))
            self.Wcf = nn.Parameter(torch.zeros(1, out_channels, *img_size))
            self.Wco = nn.Parameter(torch.zeros(1, out_channels, *img_size))
        else:
            self.input_conv = nn.Conv2d(in_channels, out_channels, kernel, stride, 0, dilation, groups, bias)

        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel, stride, 0, dilation, groups, False)

        if cfg.weights:
            self.input_conv.apply(weights_init(cfg.weights))
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

    def forward(self, inputs, mask, lstm_state=None):

        pad_input = self.trans_pad(inputs, self.padding)
        pad_mask = self.trans_pad(mask, self.padding)

        if cfg.lstm_steps:
            pad_input = batch_to_sequence(pad_input, cfg.batch_size)
            pad_mask = batch_to_sequence(pad_mask, cfg.batch_size)

            lstm_steps = pad_input.shape[1]
            next_hs = []

            if lstm_state is None:
                batch_size = pad_input.shape[0]
                h = torch.zeros((batch_size, self.out_channels, *self.img_size), dtype=torch.float).to(cfg.device)
                mem_cell = torch.zeros((batch_size, self.out_channels, *self.img_size), dtype=torch.float).to(
                    cfg.device)
            else:
                h, mem_cell = lstm_state

            # iterate through time steps
            for i in range(lstm_steps):
                h = self.trans_pad(h, self.padding)
                input = pad_input[:, i, :, :, :] * pad_mask[:, i, :, :, :]

                input_memory = torch.cat([input, h], dim=1)
                gates = self.input_conv(input_memory)
                # lstm convolution
                input, forget, cell, output = torch.split(gates, self.out_channels, dim=1)
                input = torch.sigmoid(input + self.Wci * mem_cell)
                forget = torch.sigmoid(forget + self.Wcf * mem_cell)
                mem_cell = forget * mem_cell + input * torch.tanh(cell)
                output = torch.sigmoid(output + self.Wco * mem_cell)
                h = output * torch.tanh(mem_cell)
                next_hs.append(h)
            output = torch.stack(next_hs, dim=1)
            output = sequence_to_batch(output)
            pad_mask = sequence_to_batch(pad_mask)
            output = self.trans_pad(output, self.padding)
            output = self.max_pool(output)

            lstm_state = (h, mem_cell)
        else:
            output = self.input_conv(pad_input * pad_mask)
            lstm_state = None

        if self.input_conv.bias is not None:
            if cfg.lstm_steps:
                output_bias = self.input_conv.bias[0].view(1, -1, 1, 1).expand_as(output)
            else:
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

        return output, new_mask, lstm_state
