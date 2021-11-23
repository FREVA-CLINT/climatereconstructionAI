import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLSTMBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, padding, dilation, groups, bias):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lstm_conv = nn.Conv2d(in_channels + out_channels, 4*out_channels, kernel, stride, padding, dilation, groups, bias)
        self.mem_cell_conv = nn.Conv2d(in_channels // 2, out_channels, kernel, (1,1), padding, dilation, groups, bias)
        self.bias = self.lstm_conv.bias
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, padding, dilation, groups, bias)

    def forward(self, input, lstm_state=None):
        if lstm_state is None:
            b_w_h = input.shape[0], input.shape[2], input.shape[3]
            h = torch.zeros((b_w_h[0], self.out_channels, b_w_h[1],
                             b_w_h[2]), dtype=torch.float).to(torch.device('cpu'))
            mem_cell = torch.zeros((b_w_h[0], self.out_channels, b_w_h[1] // 2,
                                    b_w_h[2] // 2), dtype=torch.float).to(torch.device('cpu'))
        else:
            h, mem_cell = lstm_state

        combined_input = torch.cat([input, h], dim=1)
        combined_output = self.lstm_conv(combined_input)

        input, forget, output, gate = torch.split(combined_output, self.out_channels, dim=1)

        input = torch.sigmoid(input)
        forget = torch.sigmoid(forget)
        output = torch.sigmoid(output)
        gate = torch.tanh(gate)

        next_mem_cell = forget * mem_cell + input * gate
        next_h = output * torch.tanh(next_mem_cell)
        return next_h, (next_h, next_mem_cell)


class PConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, in_channels_mask, kernel, stride, dilation=(1, 1), groups=1, bias=True):
        super().__init__()
        padding = kernel[0] // 2, kernel[1] // 2
        self.input_conv = ConvLSTMBlock(in_channels, out_channels, kernel, stride, padding, dilation, groups, False)

        self.mask_conv = nn.Conv2d(in_channels_mask, out_channels, kernel, stride, padding, dilation, groups, False)

        torch.nn.init.constant_(self.mask_conv.weight, 1.0)

        # exclude mask gradients from backpropagation
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, input, lstm_state, mask):
        output, lstm_state = self.input_conv(input * mask, lstm_state)
        if False:# self.input_conv.bias is not None:
            output_bias = torch.sum(self.input_conv.bias).view(1, -1, 1, 1).expand_as(output)
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


class PConvLSTMActivationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, in_channels_mask, kernel=(3, 3), stride=(1, 1),
                 activation=None, bn=True, bias=False):
        super().__init__()
        self.conv = PConvBlock(in_channels, out_channels, in_channels_mask, kernel, stride, bias=bias)

        if bn:
            self.bn = nn.BatchNorm2d(out_channels)
        if activation:
            self.activation = activation

    def forward(self, input, lstm_state, input_mask):
        h, lstm_state, h_mask = self.conv(input, lstm_state, input_mask)
        if hasattr(self, 'bn'):
            h = self.bn(h)
        if hasattr(self, 'activation'):
            h = self.activation(h)
        return h, lstm_state, h_mask


class PConvLSTM(nn.Module):
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
            PConvLSTMActivationBlock(self.num_in_channels, image_size // (2 ** (self.num_enc_dec_layers - 1)),self.num_in_channels,
                                     (7, 7), (2, 2), nn.ReLU()))
        for i in range(1, self.num_enc_dec_layers):
            if i == self.num_enc_dec_layers - 1:
                self.encoding_layers.append(PConvLSTMActivationBlock(image_size // (2 ** (self.num_enc_dec_layers - i)),
                                                                     image_size // (2 ** (
                                                                             self.num_enc_dec_layers - i - 1)),image_size // (2 ** (self.num_enc_dec_layers - i)),
                                                                     (3, 3), (2, 2), nn.ReLU()))
            else:
                self.encoding_layers.append(PConvLSTMActivationBlock(image_size // (2 ** (self.num_enc_dec_layers - i)),
                                                                     image_size // (2 ** (
                                                                             self.num_enc_dec_layers - i - 1)),image_size // (2 ** (self.num_enc_dec_layers - i)),
                                                                     (5, 5), (2, 2), nn.ReLU()))

        # define ecoding pooling layers
        for i in range(self.num_pool_layers):
            self.encoding_layers.append(PConvLSTMActivationBlock(image_size, image_size, (3, 3), (2, 2), nn.ReLU()))
        self.encoding_layers = nn.ModuleList(self.encoding_layers)

        # define decoding pooling layers
        self.decoding_layers = []
        for i in range(self.num_pool_layers):
            self.decoding_layers.append(PConvLSTMActivationBlock(image_size + image_size, image_size,
                                                                 (3, 3), (1, 1), nn.LeakyReLU()))

        # define decoding layers
        for i in range(1, self.num_enc_dec_layers):
            self.decoding_layers.append(
                PConvLSTMActivationBlock(image_size // (2 ** (i - 1)) + image_size // (2 ** (i - 1)), image_size // (2 ** i), image_size // (2 ** (i - 1)) + image_size // (2 ** i),
                                         (3, 3), (1, 1), nn.LeakyReLU()))
        self.decoding_layers.append(
            PConvLSTMActivationBlock(image_size // (2 ** (self.num_enc_dec_layers - 1)) + self.num_in_channels + 17, 1,image_size // (2 ** (self.num_enc_dec_layers - 1)) + self.num_in_channels,
                                     (3, 3), (1, 1), bn=False, bias=True))
        self.decoding_layers = nn.ModuleList(self.decoding_layers)
        self.final_conv = nn.Conv2d(1, 6, (3,3), (1,1), (1, 1), (1,1))

    def forward(self, input, input_mask, device):
        # get the number of time steps for LSTM
        num_time_steps = input.shape[1]

        # init hidden states
        lstm_states = []

        hs = [input]
        hs_mask = [input_mask]

        # forward pass encoding layers
        for i in range(self.net_depth):
            hs_inner = []
            lstm_states_inner = []
            hs_mask_inner = []
            for j in range(num_time_steps):
                h, lstm_state, h_mask = self.encoding_layers[i](input=hs[i][:, j, :, :, :],
                                                                lstm_state=None,
                                                                input_mask=hs_mask[i][:, j, :, :, :])
                hs_inner.append(h)
                lstm_states_inner.append(lstm_state)
                hs_mask_inner.append(h_mask)

            hs.append(torch.stack(hs_inner, dim=1))
            lstm_states.append(lstm_states_inner)
            hs_mask.append(torch.stack(hs_mask_inner, dim=1))
        # get output from last encoding layer
        h_sequence, h_mask_sequence = hs[self.net_depth], hs_mask[self.net_depth]
        # forward pass decoding layers
        for i in range(self.net_depth):
            hs_inner = []
            hs_mask_inner = []
            for j in range(num_time_steps):
                # interpolate input and mask
                h = F.interpolate(h_sequence[:, j, :, :, :], scale_factor=2, mode='nearest')
                h_mask = F.interpolate(h_mask_sequence[:, j, :, :, :], scale_factor=2, mode='nearest')

                lstm_state_h, lstm_state_c = lstm_states[self.net_depth - 1 - i][j]
                lstm_state_h = F.interpolate(lstm_state_h, scale_factor=2, mode='nearest')
                lstm_state_c = F.interpolate(lstm_state_c, scale_factor=2, mode='nearest')
                lstm_state_c = self.decoding_layers[i].conv.input_conv.mem_cell_conv(lstm_state_c)

                # U-Net -> pass results from encoding layers to decoding layers
                h = torch.cat([h, hs[self.net_depth - i - 1][:, j, :, :, :]], dim=1)
                h_mask = torch.cat([h_mask, hs_mask[self.net_depth - i - 1][:, j, :, :, :]], dim=1)
                h, lstm_state, h_mask = self.decoding_layers[i](input=h,
                                                                lstm_state=(lstm_state_h, lstm_state_c),
                                                                input_mask=h_mask)
                hs_inner.append(h)
                hs_mask_inner.append(h_mask)

            h_sequence = torch.stack(hs_inner, dim=1)
            h_mask_sequence = torch.stack(hs_mask_inner, dim=1)

        # return last element of output from last decoding layer
        return self.final_conv(h_sequence[:, num_time_steps - 1, :, :, :]), h_sequence[:, num_time_steps - 1, :, :, :]

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_enc_bn:
            for i in range(self.net_depth):
                if isinstance(self.encoding_layers[i].bn, nn.BatchNorm2d):
                    self.encoding_layers[i].eval()
