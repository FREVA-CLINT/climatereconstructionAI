import torch
import torch.nn as nn
import torch.nn.functional as F
import config as cfg


class ConvLSTMBlock(nn.Module):
    def __init__(self, in_channels, out_channels, image_size, kernel, stride, padding, dilation, groups, bias):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.image_size = image_size

        self.lstm_conv = nn.Conv2d(in_channels + out_channels, 4 * out_channels, kernel, stride, padding, dilation,
                                   groups, bias)

    def forward(self, input, lstm_state=None):
        if lstm_state is None:
            batch_size = input.shape[0]
            h = torch.zeros((batch_size, self.out_channels, self.image_size,
                            self.image_size), dtype=torch.float).to(cfg.device)
            mem_cell = torch.zeros((batch_size, self.out_channels, self.image_size,
                                    self.image_size), dtype=torch.float).to(cfg.device)
        else:
            h, mem_cell = lstm_state
        combined_input = torch.cat([input, h], dim=1)
        combined_output = self.lstm_conv(combined_input)

        input, forget, gate, output = torch.split(combined_output, self.out_channels, dim=1)
        input = torch.sigmoid(input)
        forget = torch.sigmoid(forget)
        output = torch.sigmoid(output)
        gate = torch.tanh(gate)

        next_mem_cell = forget * mem_cell + input * gate
        next_h = output * torch.tanh(next_mem_cell)

        return next_h, (next_h, next_mem_cell)


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


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, image_size, kernel, stride, activation, dilation=(1, 1), groups=1,
                 lstm=False):
        super().__init__()
        padding = kernel[0] // 2, kernel[1] // 2
        self.partial_conv = PConvBlock(in_channels, out_channels, kernel, stride, padding, dilation, groups, False,
                                       activation, True)

        if lstm:
            self.lstm_conv = ConvLSTMBlock(out_channels, out_channels, image_size // 2, kernel, (1, 1), padding, (1, 1),
                                           groups, False)
            self.mem_cell_conv = nn.Conv2d(in_channels, out_channels, kernel, (1, 1), padding, dilation, groups, False)

    def forward(self, input, mask, lstm_state=None):
        output, mask = self.partial_conv(input, mask)

        if hasattr(self, 'lstm_conv'):
            output, lstm_state = self.lstm_conv(output, lstm_state)
        if hasattr(self.partial_conv, 'bn'):
            output = self.partial_conv.bn(output)
        if hasattr(self.partial_conv, 'activation'):
            output = self.partial_conv.activation(output)
        return output, mask, lstm_state


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, image_size, kernel, stride, activation, dilation=(1, 1), groups=1,
                 lstm=False, bias=False, bn=True):
        super().__init__()
        padding = kernel[0] // 2, kernel[1] // 2
        self.partial_conv = PConvBlock(in_channels, out_channels, kernel, stride, padding, dilation, groups, bias,
                                       activation, bn)

        if lstm:
            self.lstm_conv = ConvLSTMBlock(in_channels - out_channels, in_channels, image_size, kernel, (1, 1), padding,
                                           (1, 1), groups, bias)
            self.mem_cell_conv = nn.Conv2d(in_channels - out_channels, in_channels, kernel, (1,1), padding, dilation, groups, False)

    def forward(self, input, mask, lstm_state=None):
        if hasattr(self, 'lstm_conv'):
            output, lstm_state = self.lstm_conv(input, lstm_state)

        output, mask = self.partial_conv(input, mask)
        if hasattr(self.partial_conv, 'bn'):
            output = self.partial_conv.bn(output)
        if hasattr(self.partial_conv, 'activation'):
            output = self.partial_conv.activation(output)
        return output, mask, lstm_state


class PConvLSTM(nn.Module):
    def __init__(self, image_size=512, num_enc_dec_layers=4, num_pool_layers=4, num_in_channels=1, lstm=True):
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
        self.lstm = lstm

        # define encoding layers
        encoding_layers = []
        encoding_layers.append(
            EncoderBlock(
                in_channels=self.num_in_channels,
                out_channels=image_size // (2 ** (self.num_enc_dec_layers - 1)),
                image_size=image_size,
                kernel=(7, 7), stride=(2, 2), activation=nn.ReLU(), lstm=lstm))
        for i in range(1, self.num_enc_dec_layers):
            if i == self.num_enc_dec_layers - 1:
                encoding_layers.append(EncoderBlock(
                    in_channels=image_size // (2 ** (self.num_enc_dec_layers - i)),
                    out_channels=image_size // (2 ** (self.num_enc_dec_layers - i - 1)),
                    image_size=image_size // (2 ** i),
                    kernel=(3, 3), stride=(2, 2), activation=nn.ReLU(), lstm=lstm))
            else:
                encoding_layers.append(EncoderBlock(
                    in_channels=image_size // (2 ** (self.num_enc_dec_layers - i)),
                    out_channels=image_size // (2 ** (self.num_enc_dec_layers - i - 1)),
                    image_size=image_size // (2 ** i),
                    kernel=(5, 5), stride=(2, 2), activation=nn.ReLU(), lstm=lstm))
        # define encoding pooling layers
        for i in range(self.num_pool_layers):
            encoding_layers.append(EncoderBlock(
                in_channels=image_size,
                out_channels=image_size,
                image_size=image_size // (2 ** (self.num_enc_dec_layers + i + 1)),
                kernel=(3, 3), stride=(2, 2), activation=nn.ReLU(), lstm=lstm))
        self.encoder = nn.ModuleList(encoding_layers)

        # define decoding pooling layers
        decoding_layers = []
        for i in range(self.num_pool_layers):
            decoding_layers.append(DecoderBlock(
                in_channels=image_size + image_size,
                out_channels=image_size,
                image_size=image_size // (2 ** (self.net_depth - i - 1)),
                kernel=(3, 3), stride=(1, 1), activation=nn.LeakyReLU(), lstm=lstm))

        # define decoding layers
        for i in range(1, self.num_enc_dec_layers):
            decoding_layers.append(
                DecoderBlock(
                    in_channels=image_size // (2 ** (i - 1)) + image_size // (2 ** i),
                    out_channels=image_size // (2 ** i),
                    image_size=image_size // (2 ** (self.net_depth - self.num_pool_layers - i)),
                    kernel=(3, 3), stride=(1, 1), activation=nn.LeakyReLU(), lstm=lstm))

        decoding_layers.append(
            DecoderBlock(
                in_channels=image_size // (2 ** (self.num_enc_dec_layers - 1)) + self.num_in_channels,
                out_channels=1,
                image_size=image_size,
                kernel=(3, 3), stride=(1, 1), activation=None, lstm=lstm, bn=False, bias=True))
        self.decoder = nn.ModuleList(decoding_layers)

    def forward(self, input, input_mask):
        # get the number of time steps for LSTM
        num_time_steps = input.shape[1]

        # create lists for skip layers
        hs = [input]
        hs_mask = [input_mask]
        lstm_states = []

        # forward pass encoding layers
        for i in range(self.net_depth):
            hs_inner = []
            lstm_states_inner = []
            hs_mask_inner = []
            for j in range(num_time_steps):
                h, h_mask, lstm_state = self.encoder[i](hs[i][:, j, :, :, :],
                                                        hs_mask[i][:, j, :, :, :],
                                                        None)
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
                lstm_state_h, lstm_state_c = None, None

                if self.lstm:
                    # interpolate hidden state
                    lstm_state_h, lstm_state_c = lstm_states[self.net_depth - 1 - i][j]
                    lstm_state_h = F.interpolate(lstm_state_h, scale_factor=2, mode='nearest')
                    lstm_state_c = F.interpolate(lstm_state_c, scale_factor=2, mode='nearest')
                    lstm_state_c = self.decoder[i].mem_cell_conv(lstm_state_c)

                # skip layers: pass results from encoding layers to decoding layers
                h = torch.cat([h, hs[self.net_depth - i - 1][:, j, :, :, :]], dim=1)
                h_mask = torch.cat([h_mask, hs_mask[self.net_depth - i - 1][:, j, :, :, :]], dim=1)
                h, h_mask, lstm_state = self.decoder[i](h,
                                                        h_mask,
                                                        (lstm_state_h, lstm_state_c))
                hs_inner.append(h)
                hs_mask_inner.append(h_mask)

            h_sequence = torch.stack(hs_inner, dim=1)
            h_mask_sequence = torch.stack(hs_mask_inner, dim=1)

        # return last element of output from last decoding layer
        return h_sequence[:, num_time_steps - 1, :, :, :]

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_enc_bn:
            for i in range(self.net_depth):
                if isinstance(self.encoder[i].bn, nn.BatchNorm2d):
                    self.encoder[i].eval()
