import torch
import torch.nn as nn
import sys

sys.path.append('./')
import config as cfg


class ConvLSTMBlock(nn.Module):
    def __init__(self, in_channels, out_channels, image_size, kernel, stride, padding, dilation, groups):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.image_size = image_size

        self.lstm_conv = nn.Conv2d(in_channels + out_channels, 4 * out_channels, kernel, stride, padding, dilation,
                                   groups, True)

        self.Wci = nn.Parameter(torch.zeros(1, out_channels, image_size, image_size)).to(cfg.device)
        self.Wcf = nn.Parameter(torch.zeros(1, out_channels, image_size, image_size)).to(cfg.device)
        self.Wco = nn.Parameter(torch.zeros(1, out_channels, image_size, image_size)).to(cfg.device)

    def forward(self, inputs, lstm_state=None):
        lstm_steps = inputs.shape[1]
        next_hs = []

        if lstm_state is None:
            batch_size = inputs.shape[0]
            h = torch.zeros((batch_size, self.out_channels, self.image_size,
                             self.image_size), dtype=torch.float).to(cfg.device)
            mem_cell = torch.zeros((batch_size, self.out_channels, self.image_size,
                                    self.image_size), dtype=torch.float).to(cfg.device)
        else:
            h, mem_cell = lstm_state

        # iterate through time steps
        for i in range(lstm_steps):
            input = inputs[:, i, :, :, :]
            input_memory = torch.cat([input, h], dim=1)
            gates = self.lstm_conv(input_memory)
            # lstm convolution
            input, forget, cell, output = torch.split(gates, self.out_channels, dim=1)
            input = torch.sigmoid(input + self.Wci * mem_cell)
            forget = torch.sigmoid(forget + self.Wcf * mem_cell)
            mem_cell = forget * mem_cell + input * torch.tanh(cell)
            output = torch.sigmoid(output + self.Wco * mem_cell)
            h = output * torch.tanh(mem_cell)
            next_hs.append(h)
        return torch.stack(next_hs, dim=1), (h, mem_cell)


class PConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, padding, dilation, groups, bias, activation, bn):
        super().__init__()

        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel, stride, padding, dilation, groups, bias)
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel, stride, padding, dilation, groups, False)

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

        return output, new_mask