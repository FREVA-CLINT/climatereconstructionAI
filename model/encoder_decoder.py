import torch
import torch.nn as nn
import sys
import torch.nn.functional as F

sys.path.append('./')
import config as cfg
from model.conv_lstm_module import ConvLSTMBlock
from model.partial_conv_module import PConvBlock


def lstm_to_batch(input):
    return torch.reshape(input, (-1, input.shape[2], input.shape[3], input.shape[4]))


def batch_to_lstm(input, batch_size):
    return torch.reshape(input,
                         (batch_size, 2 * cfg.lstm_steps + 1, input.shape[1], input.shape[2], input.shape[3]))


class EncoderBlock(nn.Module):
    def __init__(self, conv_config, kernel, stride, activation, dilation=(1, 1), groups=1,
                 lstm=False):
        super().__init__()
        padding = kernel[0] // 2, kernel[1] // 2
        self.partial_conv = PConvBlock(conv_config['in_channels'], conv_config['out_channels'], kernel,
                                       stride, padding, dilation, groups, False, activation, True)

        if lstm:
            self.lstm_conv = ConvLSTMBlock(conv_config['out_channels'], conv_config['out_channels'],
                                           conv_config['img_size'] // 2, kernel, (1, 1), padding, (1, 1), groups)

    def forward(self, input, mask, lstm_state=None):
        batch_size = input.shape[0]

        input = lstm_to_batch(input)
        mask = lstm_to_batch(mask)

        # apply partial convolution
        output, mask = self.partial_conv(input, mask)

        output = batch_to_lstm(output, batch_size)
        mask = batch_to_lstm(mask, batch_size)

        # apply LSTM convolution
        if hasattr(self, 'lstm_conv'):
            output, lstm_state = self.lstm_conv(output, lstm_state)

        return output, mask, lstm_state


class DecoderBlock(nn.Module):
    def __init__(self, conv_config, kernel, stride, activation, dilation=(1, 1), groups=1,
                 lstm=False, bias=False, bn=True):
        super().__init__()
        padding = kernel[0] // 2, kernel[1] // 2
        self.partial_conv = PConvBlock(conv_config['in_channels'] + conv_config['skip_channels'],
                                       conv_config['out_channels'], kernel, stride, padding, dilation, groups, bias,
                                       activation, bn)
        if lstm:
            self.lstm_conv = ConvLSTMBlock(conv_config['in_channels'], conv_config['in_channels'],
                                           conv_config['img_size'] // 2, kernel, (1, 1), padding, (1, 1), groups)

    def forward(self, input, skip_input, mask, skip_mask, lstm_state=None):
        # apply LSTM convolution
        if hasattr(self, 'lstm_conv'):
            input, lstm_state = self.lstm_conv(input, lstm_state)

        batch_size = input.shape[0]

        input = lstm_to_batch(input)
        mask = lstm_to_batch(mask)
        skip_input = lstm_to_batch(skip_input)
        skip_mask = lstm_to_batch(skip_mask)

        # interpolate input and mask
        h = F.interpolate(input, scale_factor=2, mode='nearest')
        h_mask = F.interpolate(mask, scale_factor=2, mode='nearest')

        if cfg.skip_layers:
            # skip layers: pass results from encoding layers to decoding layers
            h = torch.cat([h, skip_input], dim=1)
            h_mask = torch.cat([h_mask, skip_mask], dim=1)

        # apply partial convolution
        output, mask = self.partial_conv(h, h_mask)

        output = batch_to_lstm(output, batch_size)
        mask = batch_to_lstm(mask, batch_size)

        return output, mask, lstm_state