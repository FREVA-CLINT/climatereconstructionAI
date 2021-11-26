import torch
import torch.nn as nn
import torch.nn.functional as F
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

        if hasattr(self.partial_conv, 'bn'):
            output = self.partial_conv.bn(output)
        if hasattr(self.partial_conv, 'activation'):
            output = self.partial_conv.activation(output)

        output = torch.reshape(output,
                               (batch_size, cfg.lstm_steps + 1, output.shape[1], output.shape[2], output.shape[3]))
        mask = torch.reshape(mask, (batch_size, cfg.lstm_steps + 1, mask.shape[1], mask.shape[2], mask.shape[3]))

        if hasattr(self, 'lstm_conv'):
            output, lstm_state = self.lstm_conv(output, lstm_state)

        return output, mask, lstm_state


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, image_size, kernel, stride, activation, dilation=(1, 1), groups=1,
                 lstm=False, bias=False, bn=True):
        super().__init__()
        padding = kernel[0] // 2, kernel[1] // 2
        self.partial_conv = PConvBlock(in_channels, out_channels, kernel, stride, padding, dilation, groups, bias,
                                       activation, bn)

        if lstm:
            self.lstm_conv = ConvLSTMBlock(in_channels - out_channels, in_channels - out_channels, image_size // 2,
                                           kernel, (1, 1), padding,
                                           (1, 1), groups)

    def forward(self, input, skip_input, mask, skip_mask, lstm_state=None):
        batch_size = input.shape[0]

        if hasattr(self, 'lstm_conv'):
            output, lstm_state = self.lstm_conv(input, lstm_state)

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

        if hasattr(self.partial_conv, 'bn'):
            output = self.partial_conv.bn(output)
        if hasattr(self.partial_conv, 'activation'):
            output = self.partial_conv.activation(output)

        output = torch.reshape(output,
                               (batch_size, cfg.lstm_steps + 1, output.shape[1], output.shape[2], output.shape[3]))
        mask = torch.reshape(mask, (batch_size, cfg.lstm_steps + 1, mask.shape[1], mask.shape[2], mask.shape[3]))

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
                image_size=image_size // (2 ** (self.num_enc_dec_layers + i)),
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
            h, h_mask, lstm_state = self.encoder[i](hs[i],
                                                    hs_mask[i],
                                                    None)
            hs.append(h)
            lstm_states.append(lstm_state)
            hs_mask.append(h_mask)

        # get output from last encoding layer
        h, h_mask = hs[self.net_depth], hs_mask[self.net_depth]
        # forward pass decoding layers
        for i in range(self.net_depth):
            lstm_state_h, lstm_state_c = None, None

            if self.lstm:
                lstm_state_h, lstm_state_c = lstm_states[self.net_depth - 1 - i]

            h, h_mask, lstm_state = self.decoder[i](h, hs[self.net_depth - i - 1],
                                                    h_mask, hs_mask[self.net_depth - i - 1],
                                                    (lstm_state_h, lstm_state_c))

        # return last element of output from last decoding layer
        return h

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_enc_bn:
            for i in range(self.net_depth):
                if isinstance(self.encoder[i].bn, nn.BatchNorm2d):
                    self.encoder[i].eval()
