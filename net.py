import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    return init_fun


class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]


class ConvLSTMBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, dilation, groups, bias):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_input_gate = nn.Conv2d(in_channels + out_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_forget_gate = nn.Conv2d(in_channels + out_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_output_gate = nn.Conv2d(in_channels + out_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_gate_gate = nn.Conv2d(in_channels + out_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        self.conv_input_gate.apply(weights_init('kaiming'))
        self.conv_forget_gate.apply(weights_init('kaiming'))
        self.conv_output_gate.apply(weights_init('kaiming'))
        self.conv_gate_gate.apply(weights_init('kaiming'))

    def forward(self, input, lstm_state):
        h_prev, cell = lstm_state
        combined_input = torch.cat([input, h_prev], dim=1)

        input = torch.sigmoid(self.conv_input_gate(combined_input))
        forget = torch.sigmoid(self.conv_input_gate(combined_input))
        output = torch.sigmoid(self.conv_input_gate(combined_input))
        gate = torch.tanh(self.conv_input_gate(combined_input))

        cell_next = forget * cell + input * gate
        h = output * torch.tanh(cell_next)

        return h, cell_next

    def init_lstm_state(self, device, batch_size, image_size, depth, encode):
        if encode:
            return (torch.zeros(batch_size, self.out_channels, image_size // (2 ** depth), image_size // (2 ** depth)).to(device),
                    torch.zeros(batch_size, self.out_channels, image_size // (2 ** (depth + 1)), image_size // (2 ** (depth + 1))).to(device))
        else:
            return (torch.zeros(batch_size, self.out_channels, image_size // (2 ** depth), image_size // (2 ** depth)).to(device),
                    torch.zeros(batch_size, self.out_channels, image_size // (2 ** depth), image_size // (2 ** depth)).to(device))

class PConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1), dilation=(1,1), groups=1, bias=True):
        super().__init__()
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.input_conv = ConvLSTMBlock(in_channels, out_channels, kernel_size,
                                    stride, self.padding, dilation, groups, bias)
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                   stride, self.padding, dilation, groups, False)

        torch.nn.init.constant_(self.mask_conv.weight, 1.0)

        # mask is not updated
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, input, lstm_state, mask):
        output, lstm_state = self.input_conv(input * mask, lstm_state)
        if self.input_conv.conv_input_gate.bias is not None and False:
            output_bias = self.input_conv.conv_input_gate.bias.view(1, -1, 1, 1).expand_as(output)
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

        return output, lstm_state, new_mask


class PConvBlockActivation(nn.Module):
    def __init__(self, in_channels, out_channels, bn=True, sample='none-3', activ='relu',
                 conv_bias=False):
        super().__init__()
        if sample == 'down-5':
            self.conv = PConvBlock(in_channels, out_channels, (5,5), (2,2), bias=conv_bias)
        elif sample == 'down-7':
            self.conv = PConvBlock(in_channels, out_channels, (7,7), (2,2), bias=conv_bias)
        elif sample == 'down-3':
            self.conv = PConvBlock(in_channels, out_channels, (3,3), (2,2), bias=conv_bias)
        else:
            self.conv = PConvBlock(in_channels, out_channels, (3,3), (1,1), bias=conv_bias)

        if bn:
            self.bn = nn.BatchNorm2d(out_channels)
        if activ == 'relu':
            self.activation = nn.ReLU()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, input, lstm_state, input_mask):
        h, lstm_state, h_mask = self.conv(input, lstm_state, input_mask)
        if hasattr(self, 'bn'):
            h = self.bn(h)
        if hasattr(self, 'activation'):
            h = self.activation(h)
        return h, lstm_state, h_mask


class PConvLSTM(nn.Module):
    def __init__(self, image_size=512, num_enc_dec_layers=4, num_pool_layers=4, num_in_channels=1, upsampling_mode='nearest'):
        super().__init__()
        # adjust encoding layers if it doesn't blend with image size
        if image_size % (2**(num_enc_dec_layers-1)) != 0:
            for i in range(num_enc_dec_layers):
                if image_size % (2**(num_enc_dec_layers-i-1)) == 0:
                    num_enc_dec_layers -= i
                    break
            print("WARNING: Number of encoding layers doesn't blend with image size. Using {} encoding layers instead.".format(num_enc_dec_layers))

        self.freeze_enc_bn = False
        self.upsampling_mode = upsampling_mode
        self.num_enc_dec_layers = num_enc_dec_layers
        self.num_pool_layers = num_pool_layers
        self.num_in_channels = num_in_channels
        self.net_depth = num_enc_dec_layers + num_pool_layers

        # define encoding layers
        self.encoding_layers = []
        self.encoding_layers.append(PConvBlockActivation(self.num_in_channels, image_size // (2 ** (self.num_enc_dec_layers - 1)), bn=False, sample='down-7'))
        for i in range(1, self.num_enc_dec_layers):
            if i == self.num_enc_dec_layers-1:
                sample='down-3'
            else:
                sample='down-5'
            self.encoding_layers.append(PConvBlockActivation(image_size // (2 ** (self.num_enc_dec_layers - i)),
                                                             image_size // (2 ** (self.num_enc_dec_layers - i - 1)), sample=sample))

        # define ecoding pooling layers
        for i in range(self.num_pool_layers):
            self.encoding_layers.append(PConvBlockActivation(image_size, image_size, sample='down-3'))
        self.encoding_layers = nn.ModuleList(self.encoding_layers)

        # define decoding pooling layers
        self.decoding_layers = []
        for i in range(self.num_pool_layers):
            self.decoding_layers.append(PConvBlockActivation(image_size + image_size, image_size, activ='leaky'))

        # define decoding layers
        for i in range(1, self.num_enc_dec_layers):
            self.decoding_layers.append(PConvBlockActivation(image_size // (2 ** (i - 1)) + image_size // (2 ** i), image_size // (2 ** i), activ='leaky'))
        self.decoding_layers.append(PConvBlockActivation(image_size // (2 ** (self.num_enc_dec_layers - 1)) + self.num_in_channels, 1,
                                                         bn=False, activ=None, conv_bias=True))
        self.decoding_layers = nn.ModuleList(self.decoding_layers)

    def forward(self, input, lstm_states, input_mask):
        # get the number of time steps for LSTM
        num_time_steps = input.shape[1]
        hs = [input]
        hs_mask = [input_mask]

        # forward pass encoding layers
        for i in range(self.net_depth):
            hs_inner = []
            hs_mask_inner = []

            for j in range(num_time_steps):
                h, cell, h_mask = self.encoding_layers[i](input=hs[i][:,j,:,:,:],
                                                          lstm_state=lstm_states[i],
                                                          input_mask=hs_mask[i][:,j,:,:,:])
                hs_inner.append(h)
                hs_mask_inner.append(h_mask)

            hs.append(torch.stack(hs_inner, dim=1))
            hs_mask.append(torch.stack(hs_mask_inner, dim=1))

        # get output from last encoding layer
        h_sequence, h_mask_sequence = hs[self.net_depth], hs_mask[self.net_depth]

        # forward pass decoding layers
        for i in range(self.net_depth):
            hs_inner = []
            hs_mask_inner = []
            for j in range(num_time_steps):
                # interpolate input and mask
                h = F.interpolate(h_sequence[:,j,:,:,:], scale_factor=2, mode=self.upsampling_mode)
                h_mask = F.interpolate(h_mask_sequence[:,j,:,:,:], scale_factor=2, mode='nearest')

                # U-Net -> pass results from encoding layers to decoding layers
                h = torch.cat([h, hs[self.net_depth - i - 1][:,j,:,:,:]], dim=1)
                h_mask = torch.cat([h_mask, hs_mask[self.net_depth - i - 1][:,j,:,:,:]], dim=1)
                h, cell, h_mask = self.decoding_layers[i](input=h,
                                                          lstm_state=lstm_states[self.net_depth + i],
                                                          input_mask=h_mask)
                hs_inner.append(h)
                hs_mask_inner.append(h_mask)

            h_sequence = torch.stack(hs_inner, dim=1)
            h_mask_sequence = torch.stack(hs_mask_inner, dim=1)

        # return output from last decoding layer
        return h_sequence

    def init_lstm_states(self, device, batch_size, image_size):
        init_states = []
        # encoding layers
        for i in range(self.num_enc_dec_layers):
            init_states.append(self.encoding_layers[i].conv.input_conv.init_lstm_state(device, batch_size, image_size, i, True))
        # pooling layers
        for i in range(self.num_enc_dec_layers, self.net_depth):
            init_states.append(self.encoding_layers[i].conv.input_conv.init_lstm_state(device, batch_size, image_size, i, True))
        for i in range(self.num_pool_layers):
            init_states.append(self.decoding_layers[i].conv.input_conv.init_lstm_state(device, batch_size, image_size, self.net_depth - i - 1, False))
        # decoding layers
        for i in range(self.num_pool_layers, self.net_depth):
            init_states.append(self.decoding_layers[i].conv.input_conv.init_lstm_state(device, batch_size, image_size, self.net_depth - i - 1, False))
        return init_states

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_enc_bn:
            for i in range(self.num_enc_dec_layers):
                if isinstance(self.encoding_layers[i].bn, nn.BatchNorm2d):
                    self.encoding_layers[i].eval()

