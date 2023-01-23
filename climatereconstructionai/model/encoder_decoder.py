import torch
import torch.nn as nn

from .conv_lstm_module import ConvLSTMBlock
from .partial_conv_module import PConvBlock
from .traj_gru_module import TrajGRUBlock
from .. import config as cfg


def sequence_to_batch(input):
    return torch.reshape(input, (-1, input.shape[2], input.shape[3], input.shape[4]))


def batch_to_sequence(input, batch_size):
    if cfg.lstm_steps:
        steps = cfg.lstm_steps
    elif cfg.gru_steps:
        steps = cfg.gru_steps
    else:
        steps = 0
    return torch.reshape(input,
                         (batch_size, 2 * steps + 1, input.shape[1], input.shape[2], input.shape[3]))


class EncoderBlock(nn.Module):
    def __init__(self, conv_config, kernel, stride, activation, dilation=(1, 1), groups=1):
        super().__init__()
        padding = kernel[0] // 2, kernel[1] // 2
        self.partial_conv = PConvBlock(conv_config['in_channels'], conv_config['out_channels'], kernel,
                                       stride, padding, dilation, groups, False, activation, conv_config['bn'])

        if cfg.lstm_steps:
            self.recurrent_conv = ConvLSTMBlock(conv_config['out_channels'], conv_config['out_channels'],
                                                conv_config['rec_size'], kernel, (1, 1), padding, (1, 1), groups)
        elif cfg.gru_steps:
            self.recurrent_conv = TrajGRUBlock(conv_config['out_channels'], conv_config['out_channels'],
                                               conv_config['rec_size'])

    def forward(self, input, mask, recurrent_state=None):
        batch_size = input.shape[0]

        input = sequence_to_batch(input)
        mask = sequence_to_batch(mask)

        # apply partial convolution
        output, mask = self.partial_conv(input, mask)

        output = batch_to_sequence(output, batch_size)
        mask = batch_to_sequence(mask, batch_size)

        # apply LSTM convolution
        if hasattr(self, 'recurrent_conv'):
            output, recurrent_state = self.recurrent_conv(output, recurrent_state)

        return output, mask, recurrent_state


class DecoderBlock(nn.Module):
    def __init__(self, conv_config, kernel, stride, activation, dilation=(1, 1), groups=1, bias=False):
        super().__init__()
        padding = kernel[0] // 2, kernel[1] // 2
        self.partial_conv = PConvBlock(conv_config['in_channels'] + conv_config['skip_channels'],
                                       conv_config['out_channels'], kernel, stride, padding, dilation, groups, bias,
                                       activation, conv_config['bn'])
        if cfg.lstm_steps:
            self.recurrent_conv = ConvLSTMBlock(conv_config['in_channels'], conv_config['in_channels'],
                                                conv_config['rec_size'], kernel, (1, 1), padding, (1, 1), groups)
        elif cfg.gru_steps:
            self.recurrent_conv = TrajGRUBlock(conv_config['in_channels'], conv_config['in_channels'],
                                               conv_config['rec_size'])

    def forward(self, input, skip_input, mask, skip_mask, recurrent_state=None):
        # apply LSTM convolution
        if hasattr(self, 'recurrent_conv'):
            input, recurrent_state = self.recurrent_conv(input, recurrent_state)

        batch_size = input.shape[0]

        input = sequence_to_batch(input)
        mask = sequence_to_batch(mask)
        skip_input = sequence_to_batch(skip_input)
        skip_mask = sequence_to_batch(skip_mask)

        # interpolate input and mask
        m = nn.Upsample(skip_input.size()[-2:], mode='nearest')
        h = m(input)
        h_mask = m(mask)

        if cfg.skip_layers:
            # skip layers: pass results from encoding layers to decoding layers
            h = torch.cat([h, skip_input], dim=1)
            h_mask = torch.cat([h_mask, skip_mask], dim=1)

        # apply partial convolution
        output, mask = self.partial_conv(h, h_mask)

        output = batch_to_sequence(output, batch_size)
        mask = batch_to_sequence(mask, batch_size)

        return output, mask, recurrent_state
