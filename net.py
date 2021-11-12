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
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels + out_channels, out_channels*4, kernel_size, stride, padding, dilation, groups, bias)
        self.conv.apply(weights_init('kaiming'))

    def forward(self, input, current_state):
        h_cur, c_cur = current_state
        combined = torch.cat([input, h_cur], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.out_channels, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        return (torch.zeros(batch_size, self.out_channels, image_size, image_size, device=self.conv.weight.device),
                torch.zeros(batch_size, self.out_channels, image_size, image_size, device=self.conv.weight.device))


class PConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.input_conv = ConvLSTMBlock(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups, bias)
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                   stride, padding, dilation, groups, False)

        torch.nn.init.constant_(self.mask_conv.weight, 1.0)

        # mask is not updated
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, input, hidden_state, mask):
        output, hidden_state = self.input_conv(input * mask, hidden_state)
        if self.input_conv.conv.bias is not None:
            output_bias = self.input_conv.conv.bias.view(1, -1, 1, 1).expand_as(
                output)
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

        return output, hidden_state, new_mask


class PConvBlockActivation(nn.Module):
    def __init__(self, in_channels, out_channels, bn=True, sample='none-3', activ='relu',
                 conv_bias=False):
        super().__init__()
        if sample == 'down-5':
            self.conv = PConvBlock(in_channels, out_channels, 5, 2, 2, bias=conv_bias)
        elif sample == 'down-7':
            self.conv = PConvBlock(in_channels, out_channels, 7, 2, 3, bias=conv_bias)
        elif sample == 'down-3':
            self.conv = PConvBlock(in_channels, out_channels, 3, 2, 1, bias=conv_bias)
        else:
            self.conv = PConvBlock(in_channels, out_channels, 3, 1, 1, bias=conv_bias)

        if bn:
            self.bn = nn.BatchNorm2d(out_channels)
        if activ == 'relu':
            self.activation = nn.ReLU()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, input, hidden_state, input_mask):
        h, hidden_state, h_mask = self.conv(input, hidden_state, input_mask)
        if hasattr(self, 'bn'):
            h = self.bn(h)
        if hasattr(self, 'activation'):
            h = self.activation(h)
        return h, hidden_state, h_mask


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
        self.num_enc_layers = self.num_dec_layers = num_enc_dec_layers
        self.num_pool_layers = num_pool_layers

        # define encoding layers
        self.encoding_layers = []
        self.encoding_layers.append(PConvBlockActivation(num_in_channels, num_in_channels))
        for i in range(1, self.num_enc_layers):
            if i == self.num_enc_layers-1:
                sample='down-3'
            else:
                sample='down-5'
            self.encoding_layers.append(PConvBlockActivation(image_size // (2 ** (self.num_enc_layers - i)),
                                                     image_size // (2**(self.num_enc_layers-i-1)), sample=sample))
        # define ecoding pooling layers
        for i in range(self.num_pool_layers):
            self.encoding_layers.append(PConvBlockActivation(image_size, image_size, sample='down-3'))
        self.encoding_layers = nn.ModuleList(self.encoding_layers)

        # define decoding pooling layers
        self.decoding_layers = []
        for i in range(self.num_pool_layers):
            self.decoding_layers.append(PConvBlockActivation(image_size + image_size, image_size, activ='leaky'))
        self.decoding_layers = nn.ModuleList(self.decoding_layers)
        # define decoding layers
        for i in range(1, self.num_dec_layers):
            self.decoding_layers.append(PConvBlockActivation(image_size // (2 ** (i - 1)) + image_size // (2 ** i), image_size // (2 ** i), activ='leaky'))
        self.decoding_layers.append(PConvBlockActivation(num_in_channels, num_in_channels,
                                                         bn=False, activ=None, conv_bias=True))
        self.decoding_layers = nn.ModuleList(self.decoding_layers)

    def forward(self, input, hidden_states, input_mask):
        # get the number of time steps for LSTM
        num_time_steps = input.shape[1]

        hs = [input]
        hs_mask = [input_mask]
        next_hidden = [hidden_states]

        # forward pass encoding layers
        for i in range(self.num_enc_layers):
            print(hidden_states[i].__len__())
            h, h_hidden = hidden_states[i]
            hs_inner = []
            hs_mask_inner = []

            for j in range(num_time_steps):
                print(hs[i][:,j,:,:,:].shape)
                print([h, h_hidden].__len__())
                print(hs_mask[i][:,j,:,:,:].shape)
                h, h_hidden, h_mask = self.encoding_layers[i](input=hs[i][:,j,:,:,:],
                                                              hidden_state=hidden_states[i],
                                                              input_mask=hs_mask[i][:,j,:,:,:])
                hs_inner.append(h)
                hs_mask_inner.append(h_mask)

            hs.append(torch.stack(hs_inner, dim=1))
            next_hidden.append([h, h_hidden])
            hs_mask.append(torch.stack(hs_mask_inner, dim=1))

        # get current states
        h, h_hidden, h_mask = hs[self.num_enc_layers], next_hidden[self.num_enc_layers], hs_mask[self.num_enc_layers]

        # forward pass decoding layers
        for i in range(self.num_dec_layers):
            # interpolate input, hidden state and mask
            h = F.interpolate(h, scale_factor=2, mode=self.upsampling_mode)
            h_cur, c_cur = h_hidden
            h_cur = F.interpolate(h_cur, scale_factor=2, mode=self.upsampling_mode)
            c_cur = F.interpolate(c_cur, scale_factor=2, mode=self.upsampling_mode)
            h_mask = F.interpolate(
                h_mask, scale_factor=2, mode='nearest')

            # U-Net -> pass results from encoding layers to decoding layers
            print(h.shape)
            print(hs[self.num_enc_layers - i].shape)
            h = torch.cat([h, hs[self.num_enc_layers - i]], dim=1)
            h_prev, c_prev = next_hidden[self.num_enc_layers - i]
            h_prev, c_prev = torch.cat([h_cur, h_prev], dim=1), torch.cat([c_cur, c_prev], dim=1)
            h_mask = torch.cat([h_mask, hs_mask[self.num_enc_layers - i]])

            hs_inner = []
            hs_mask_inner = []

            for j in range(num_time_steps):
                h, h_hidden, h_mask = self.encoding_layers[i](input=h[:,j,:,:,:],
                                                              hidden_state=[h_prev, c_prev],
                                                              input_mask=h_mask)
                hs_inner.append(h)
                hs_mask_inner.append(h_mask)

            hs.append(torch.stack(hs_inner, dim=1))
            hs_mask.append(torch.stack(hs_mask_inner, dim=1))

        return hs, next_hidden, hs_mask

    def init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_enc_layers):
            init_states.append(self.encoding_layers[i].conv.input_conv.init_hidden(batch_size, image_size))
        for i in range(self.num_dec_layers):
            init_states.append(self.decoding_layers[i].conv.input_conv.init_hidden(batch_size, image_size))
        return init_states

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_enc_bn:
            for i in range(self.num_enc_layers):
                if isinstance(self.encoding_layers[i].bn, nn.BatchNorm2d):
                    self.encoding_layers[i].eval()

