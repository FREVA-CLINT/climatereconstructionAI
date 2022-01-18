import torch
import torch.nn as nn
import sys
import torch.nn.functional as F

sys.path.append('./')
import config as cfg
from model.pconv_lstm_module import PConvBlock, ConvLSTMBlock


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, image_size, kernel, stride, activation, dilation=(1, 1), groups=1,
                 lstm=False):
        super().__init__()
        padding = kernel[0] // 2, kernel[1] // 2
        self.partial_conv = PConvBlock(in_channels, out_channels, kernel, stride, padding, dilation, groups, False,
                                       activation, True)

        if lstm:
            self.lstm_conv = ConvLSTMBlock(out_channels, out_channels, image_size // 2, kernel, (1, 1), padding, (1, 1),
                                           groups)

    def forward(self, input, mask, lstm_state=None):
        batch_size = input.shape[0]

        input = torch.reshape(input, (-1, input.shape[2], input.shape[3], input.shape[4]))
        mask = torch.reshape(mask, (-1, mask.shape[2], mask.shape[3], mask.shape[4]))
        output, mask = self.partial_conv(input, mask)

        output = torch.reshape(output,
                               (batch_size, 2 * cfg.lstm_steps + 1, output.shape[1], output.shape[2], output.shape[3]))
        mask = torch.reshape(mask, (batch_size, 2 * cfg.lstm_steps + 1, mask.shape[1], mask.shape[2], mask.shape[3]))

        if hasattr(self, 'lstm_conv'):
            output, lstm_state = self.lstm_conv(output, lstm_state)

        return output, mask, lstm_state


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, image_size, kernel, stride, activation, dilation=(1, 1), groups=1,
                 lstm=False, bias=False, bn=True, init_in_channels=1, init_out_channels=1):
        super().__init__()
        padding = kernel[0] // 2, kernel[1] // 2
        self.partial_conv = PConvBlock(in_channels, out_channels, kernel, stride, padding, dilation, groups, bias,
                                       activation, bn)

        if lstm:
            self.lstm_conv = ConvLSTMBlock(in_channels - (out_channels + init_in_channels - init_out_channels),
                                           in_channels - (out_channels + init_in_channels - init_out_channels),
                                           image_size // 2, kernel, (1, 1), padding, (1, 1), groups)

    def forward(self, input, skip_input, mask, skip_mask, lstm_state=None):
        batch_size = input.shape[0]

        if hasattr(self, 'lstm_conv'):
            input, lstm_state = self.lstm_conv(input, lstm_state)

        input = torch.reshape(input, (-1, input.shape[2], input.shape[3], input.shape[4]))
        mask = torch.reshape(mask, (-1, mask.shape[2], mask.shape[3], mask.shape[4]))

        # interpolate input and mask
        h = F.interpolate(input, scale_factor=2, mode='nearest')
        h_mask = F.interpolate(mask, scale_factor=2, mode='nearest')

        # skip layers: pass results from encoding layers to decoding layers
        skip_input = torch.reshape(skip_input, (-1, skip_input.shape[2], skip_input.shape[3], skip_input.shape[4]))
        skip_mask = torch.reshape(skip_mask, (-1, skip_mask.shape[2], skip_mask.shape[3], skip_mask.shape[4]))
        h = torch.cat([h, skip_input], dim=1)
        h_mask = torch.cat([h_mask, skip_mask], dim=1)

        output, mask = self.partial_conv(h, h_mask)

        output = torch.reshape(output,
                               (batch_size, 2 * cfg.lstm_steps + 1, output.shape[1], output.shape[2], output.shape[3]))
        mask = torch.reshape(mask, (batch_size, 2 * cfg.lstm_steps + 1, mask.shape[1], mask.shape[2], mask.shape[3]))

        return output, mask, lstm_state