import torch
import torch.nn as nn
import torch.nn.functional as F


class PConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, dilation=(1, 1), groups=1, bias=True):
        super().__init__()
        padding = kernel[0] // 2, kernel[1] // 2
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel, stride, padding, dilation, groups, bias)
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel, stride, padding, dilation, groups, False)

        torch.nn.init.constant_(self.mask_conv.weight, 1.0)

        # exclude mask gradients from backpropagation
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, input, mask):
        output = self.input_conv(input * mask)
        if self.input_conv.bias is not None:
            output_bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(output)
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


class MultiChannelPConvActivationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=(3, 3), stride=(1, 1),
                 activation=None, bn=True, bias=False):
        super().__init__()
        self.conv = PConvBlock(in_channels, out_channels, kernel, stride, bias=bias)

        if bn:
            self.bn = nn.BatchNorm2d(out_channels)
        if activation:
            self.activation = activation

    def forward(self, input, input_mask):
        h, h_mask = self.conv(input, input_mask)
        if hasattr(self, 'bn'):
            h = self.bn(h)
        if hasattr(self, 'activation'):
            h = self.activation(h)
        return h, h_mask


class MultiChannelPConvUNet(nn.Module):
    def __init__(self, image_size=512, num_enc_dec_layers=4, num_pool_layers=4, num_in_channels=1):
        super().__init__()

        # adjust encoding layers if it doesn't blend with image size
        if image_size % (2 ** (num_enc_dec_layers + num_pool_layers - 1)) != 0:
            num_enc_dec_layers = num_enc_dec_layers + num_pool_layers
            num_pool_layers = 0
            for i in range(num_enc_dec_layers):
                if image_size % (2 ** (num_enc_dec_layers - i - 1)) == 0:
                    num_enc_dec_layers -= i
                    break
            print("WARNING: Number of encoding layers doesn't match with image size. Using {} encoding and" +
                  " 0 pooling layers layers instead.".format(num_enc_dec_layers))

        self.freeze_enc_bn = False
        self.num_enc_dec_layers = num_enc_dec_layers
        self.num_pool_layers = num_pool_layers
        self.num_in_channels = num_in_channels
        self.net_depth = num_enc_dec_layers + num_pool_layers

        # define encoding layers
        self.encoding_layers = []
        self.encoding_layers.append(
            MultiChannelPConvActivationBlock(self.num_in_channels, image_size // (2 ** (self.num_enc_dec_layers - 1)),
                                             (7, 7), (2, 2), nn.ReLU()))
        for i in range(1, self.num_enc_dec_layers):
            if i == self.num_enc_dec_layers - 1:
                self.encoding_layers.append(MultiChannelPConvActivationBlock(image_size // (2 ** (self.num_enc_dec_layers - i)),
                                                                             image_size // (2 ** (self.num_enc_dec_layers - i - 1)),
                                                                             (3, 3), (2, 2), nn.ReLU()))
            else:
                self.encoding_layers.append(MultiChannelPConvActivationBlock(image_size // (2 ** (self.num_enc_dec_layers - i)),
                                                                             image_size // (2 ** (self.num_enc_dec_layers - i - 1)),
                                                                             (5, 5), (2, 2), nn.ReLU()))

        # define ecoding pooling layers
        for i in range(self.num_pool_layers):
            self.encoding_layers.append(MultiChannelPConvActivationBlock(image_size, image_size, (3, 3), (2, 2), nn.ReLU()))
        self.encoding_layers = nn.ModuleList(self.encoding_layers)

        # define decoding pooling layers
        self.decoding_layers = []
        for i in range(self.num_pool_layers):
            self.decoding_layers.append(MultiChannelPConvActivationBlock(image_size + image_size, image_size,
                                                                         (3, 3), (1, 1), nn.LeakyReLU()))

        # define decoding layers
        for i in range(1, self.num_enc_dec_layers):
            self.decoding_layers.append(
                MultiChannelPConvActivationBlock(image_size // (2 ** (i - 1)) + image_size // (2 ** i), image_size // (2 ** i),
                                                 (3, 3), (1, 1), nn.LeakyReLU()))
        self.decoding_layers.append(
            MultiChannelPConvActivationBlock(image_size // (2 ** (self.num_enc_dec_layers - 1)) + self.num_in_channels, 1,
                                             (3, 3), (1, 1), bn=False, bias=True))
        self.decoding_layers = nn.ModuleList(self.decoding_layers)

    def forward(self, input, input_mask):
        hs = [input]
        hs_mask = [input_mask]

        # forward pass encoding layers
        for i in range(self.net_depth):
            h, h_mask = self.encoding_layers[i](input=hs[i],
                                                input_mask=hs_mask[i])
            hs.append(h)
            hs_mask.append(h_mask)

        # get current states
        h, h_mask = hs[self.net_depth], hs_mask[self.net_depth]

        # forward pass decoding layers
        for i in range(self.net_depth):
            # interpolate encoder output and mask
            h = F.interpolate(h, scale_factor=2, mode='nearest')
            h_mask = F.interpolate(h_mask, scale_factor=2, mode='nearest')

            # U-Net -> pass results from encoding layers to decoding layers
            h = torch.cat([h, hs[self.net_depth - i - 1]], dim=1)
            h_mask = torch.cat([h_mask, hs_mask[self.net_depth - i - 1]], dim=1)

            h, h_mask = self.decoding_layers[i](input=h,
                                                input_mask=h_mask)

        return h

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_enc_bn:
            for i in range(self.net_depth):
                if isinstance(self.encoding_layers[i].bn, nn.BatchNorm2d):
                    self.encoding_layers[i].eval()

