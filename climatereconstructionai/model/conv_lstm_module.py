import torch
import torch.nn as nn

from .. import config as cfg
from ..utils.weights import weights_init


class ConvLSTMBlock(nn.Module):
    def __init__(self, in_channels, out_channels, image_size, kernel, stride, padding, dilation, groups):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.image_size = image_size

        self.lstm_conv = nn.Conv2d(in_channels + out_channels, 4 * out_channels, kernel, stride, padding, dilation,
                                   groups, True)

        if cfg.weights:
            self.lstm_conv.apply(weights_init(cfg.weights))

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
