import torch
import torch.nn as nn
import torch.nn.functional as F
import config as cfg


class MaxChannelPool(nn.MaxPool1d):
    def forward(self, input):
        n, c, w, h = input.size()
        input = input.view(n, c, w * h).permute(0, 2, 1)
        pooled = F.max_pool1d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.ceil_mode,
            self.return_indices,
        )
        _, _, c = pooled.size()
        pooled = pooled.permute(0, 2, 1)
        return pooled.view(n, c, w, h)


class AvgChannelPool(nn.AvgPool1d):
    def forward(self, input):
        n, c, w, h = input.size()
        input = input.view(n, c, w * h).permute(0, 2, 1)
        pooled = F.max_pool1d(
            input,
            self.kernel_size,
            self.stride,
            self.padding
        )
        _, _, c = pooled.size()
        pooled = pooled.permute(0, 2, 1)
        return pooled.view(n, c, w, h)


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

        if hasattr(self.partial_conv, 'bn'):
            output = self.partial_conv.bn(output)
        if hasattr(self.partial_conv, 'activation'):
            output = self.partial_conv.activation(output)

        output = torch.reshape(output,
                               (batch_size, 2 * cfg.lstm_steps + 1, output.shape[1], output.shape[2], output.shape[3]))
        mask = torch.reshape(mask, (batch_size, 2 * cfg.lstm_steps + 1, mask.shape[1], mask.shape[2], mask.shape[3]))

        return output, mask, lstm_state


class PConvLSTM(nn.Module):
    def __init__(self, radar_img_size=512, radar_enc_dec_layers=4, radar_pool_layers=4, radar_in_channels=1,
                 radar_out_channels=1,
                 rea_img_size=[], rea_enc_layers=[], rea_pool_layers=[], rea_in_channels=[],
                 lstm=True):
        super().__init__()

        assert len(rea_img_size) == len(rea_enc_layers) == len(rea_pool_layers) == len(rea_in_channels)

        self.freeze_enc_bn = False
        self.radar_enc_dec_layers = radar_enc_dec_layers
        self.radar_pool_layers = radar_pool_layers
        self.rea_enc_layers = rea_enc_layers
        self.rea_pool_layers = rea_pool_layers
        self.net_depth = radar_enc_dec_layers + radar_pool_layers
        self.lstm = lstm

        # define encoding layers
        encoding_layers = []
        encoding_layers.append(
            EncoderBlock(
                in_channels=radar_in_channels,
                out_channels=radar_img_size // (2 ** (self.radar_enc_dec_layers - 1)),
                image_size=radar_img_size,
                kernel=(7, 7), stride=(2, 2), activation=nn.ReLU(), lstm=lstm))
        for i in range(1, self.radar_enc_dec_layers):
            if i == self.radar_enc_dec_layers - 1:
                encoding_layers.append(EncoderBlock(
                    in_channels=radar_img_size // (2 ** (self.radar_enc_dec_layers - i)),
                    out_channels=radar_img_size // (2 ** (self.radar_enc_dec_layers - i - 1)),
                    image_size=radar_img_size // (2 ** i),
                    kernel=(3, 3), stride=(2, 2), activation=nn.ReLU(), lstm=lstm))
            else:
                encoding_layers.append(EncoderBlock(
                    in_channels=radar_img_size // (2 ** (self.radar_enc_dec_layers - i)),
                    out_channels=radar_img_size // (2 ** (self.radar_enc_dec_layers - i - 1)),
                    image_size=radar_img_size // (2 ** i),
                    kernel=(5, 5), stride=(2, 2), activation=nn.ReLU(), lstm=lstm))
        # define encoding pooling layers
        for i in range(self.radar_pool_layers):
            encoding_layers.append(EncoderBlock(
                in_channels=radar_img_size,
                out_channels=radar_img_size,
                image_size=radar_img_size // (2 ** (self.radar_enc_dec_layers + i)),
                kernel=(3, 3), stride=(2, 2), activation=nn.ReLU(), lstm=lstm))
        self.encoder = nn.ModuleList(encoding_layers)

        # define encoders for additional information
        self.attention_extractors = []
        for i in range(len(rea_img_size)):
            attention_extractor = {}
            rea_encoding_layers = []
            rea_encoding_layers.append(
                EncoderBlock(
                    in_channels=rea_in_channels[i],
                    out_channels=rea_img_size[i] // (2 ** (rea_enc_layers[i] - 1)),
                    image_size=rea_img_size[i],
                    kernel=(7, 7), stride=(2, 2), activation=nn.ReLU(), lstm=lstm))
            for j in range(1, rea_enc_layers[i]):
                if i == self.radar_enc_dec_layers - 1:
                    rea_encoding_layers.append(EncoderBlock(
                        in_channels=rea_img_size[i] // (2 ** (rea_enc_layers[i] - j)),
                        out_channels=rea_img_size[i] // (2 ** (rea_enc_layers[i] - j - 1)),
                        image_size=rea_img_size[i] // (2 ** j),
                        kernel=(3, 3), stride=(2, 2), activation=nn.ReLU(), lstm=lstm))
                else:
                    rea_encoding_layers.append(EncoderBlock(
                        in_channels=rea_img_size[i] // (2 ** (rea_enc_layers[i] - j)),
                        out_channels=rea_img_size[i] // (2 ** (rea_enc_layers[i] - j - 1)),
                        image_size=rea_img_size[i] // (2 ** j),
                        kernel=(5, 5), stride=(2, 2), activation=nn.ReLU(), lstm=lstm))
            # define encoding pooling layers
            for j in range(rea_pool_layers[i]):
                rea_encoding_layers.append(EncoderBlock(
                    in_channels=rea_img_size[i],
                    out_channels=rea_img_size[i],
                    image_size=rea_img_size[i] // (2 ** (rea_enc_layers[i] + j)),
                    kernel=(3, 3), stride=(2, 2), activation=nn.ReLU(), lstm=lstm))
            attention_extractor['encoding_layers'] = nn.ModuleList(rea_encoding_layers).to(cfg.device)
            attention_extractor['mlp_image'] = nn.Sequential(
                nn.Conv2d(in_channels=rea_img_size[i], out_channels=rea_img_size[i] // cfg.channel_reduction_rate,
                          kernel_size=(1, 1)),
                nn.ReLU(),
                nn.Conv2d(in_channels=rea_img_size[i] // cfg.channel_reduction_rate, out_channels=rea_img_size[i],
                          kernel_size=(1, 1)),
            ).to(cfg.device)
            attention_extractor['mlp_mask'] = nn.Sequential(
                nn.Conv2d(in_channels=rea_img_size[i], out_channels=rea_img_size[i] // cfg.channel_reduction_rate,
                          kernel_size=(1, 1)),
                nn.ReLU(),
                nn.Conv2d(in_channels=rea_img_size[i] // cfg.channel_reduction_rate, out_channels=rea_img_size[i],
                          kernel_size=(1, 1)),
            ).to(cfg.device)
            self.attention_extractors.append(attention_extractor)
        # add fusion layer if extractors exist
        if self.attention_extractors:
            self.max_pool = MaxChannelPool(kernel_size=radar_img_size)
            self.avg_pool = AvgChannelPool(kernel_size=radar_img_size)
            self.spatial_attention_img = nn.Sequential(
                nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(3, 3), padding=(1, 1)),
                nn.Sigmoid()
            )
            self.spatial_attention_mask = nn.Sequential(
                nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(3, 3), padding=(1, 1)),
                nn.Sigmoid()
            )
            rea_channels = len(rea_img_size) * radar_img_size
        else:
            rea_channels = 0

        # define decoding pooling layers
        decoding_layers = []
        for i in range(self.radar_pool_layers):
            decoding_layers.append(DecoderBlock(
                in_channels=radar_img_size + radar_img_size + rea_channels,
                out_channels=radar_img_size,
                image_size=radar_img_size // (2 ** (self.net_depth - i - 1)),
                kernel=(3, 3), stride=(1, 1), activation=nn.LeakyReLU(), lstm=lstm))

        # define decoding layers
        for i in range(1, self.radar_enc_dec_layers):
            decoding_layers.append(
                DecoderBlock(
                    in_channels=radar_img_size // (2 ** (i - 1)) + radar_img_size // (2 ** i),
                    out_channels=radar_img_size // (2 ** i),
                    image_size=radar_img_size // (2 ** (self.net_depth - self.radar_pool_layers - i)),
                    kernel=(3, 3), stride=(1, 1), activation=nn.LeakyReLU(), lstm=lstm))

        decoding_layers.append(
            DecoderBlock(
                in_channels=radar_img_size // (2 ** (self.radar_enc_dec_layers - 1)) + radar_in_channels,
                out_channels=radar_out_channels,
                image_size=radar_img_size,
                kernel=(3, 3), stride=(1, 1), activation=None, lstm=lstm, bn=False, bias=True,
                init_in_channels=radar_in_channels, init_out_channels=radar_out_channels))
        self.decoder = nn.ModuleList(decoding_layers)

    def forward(self, input, input_mask):
        # create lists for skip layers
        h = input[:, 0, :, :, :, :]
        h_mask = input_mask[:, 0, :, :, :, :]
        hs = [h]
        hs_mask = [h_mask]
        lstm_states = []

        # forward pass encoding layers
        for i in range(self.net_depth):
            h, h_mask, lstm_state = self.encoder[i](hs[i],
                                                    hs_mask[i],
                                                    None)
            hs.append(h)
            lstm_states.append(lstm_state)
            hs_mask.append(h_mask)

        # reverse all hidden states
        for i in range(self.net_depth):
            hs[i] = torch.flip(hs[i], (1,))
            hs_mask[i] = torch.flip(hs_mask[i], (1,))

        # forward fusion data
        if self.attention_extractors:

            # calculate channel attention
            h_channel_attentions = []
            h_mask_channel_attentions = []
            for i in range(len(self.attention_extractors)):
                h_channel_attention, h_mask_channel_attention = self.forward_channel_attention(
                    self.attention_extractors[i], input[:, i + 1, :, :, :, :], input_mask[:, i + 1, :, :, :, :])
                h_channel_attentions.append(h_channel_attention * h)
                h_mask_channel_attentions.append(h_mask_channel_attention * h_mask)

            # calculate spatial attention
            h_spatial_attention = torch.unsqueeze(self.spatial_attention_img(
                torch.cat([torch.max(h[:, 0, :, :, :], keepdim=True, dim=1)[0],
                           torch.mean(h[:, 0, :, :, :], keepdim=True, dim=1)], dim=1)
            ), dim=1)
            h_mask_spatial_attention = torch.unsqueeze(self.spatial_attention_mask(
                torch.cat([torch.max(h_mask[:, 0, :, :, :], keepdim=True, dim=1)[0],
                           torch.mean(h_mask[:, 0, :, :, :], keepdim=True, dim=1)], dim=1)
            ), dim=1)

            h_total_attentions = []
            h_mask_total_attentions = []
            for i in range(len(self.attention_extractors)):
                h_total_attention = h_channel_attentions[i] * h_spatial_attention
                h_total_attentions.append(h_total_attention)

                h_mask_total_attention = h_mask_channel_attentions[i] * h_mask_spatial_attention
                h_mask_total_attentions.append(h_mask_total_attention)
            h_total_attentions = torch.cat(h_total_attentions, dim=2)
            h_mask_total_attentions = torch.cat(h_mask_total_attentions, dim=2)

            h = torch.cat([h, h_total_attentions], dim=2)
            h_mask = torch.cat([h_mask, h_mask_total_attentions], dim=2)

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

    def forward_channel_attention(self, extractor, fusion_input, fusion_input_mask):
        fusion, fusion_mask = fusion_input, fusion_input_mask
        for i in range(len(extractor['encoding_layers'])):
            fusion, fusion_mask, lstm_state = extractor['encoding_layers'][i](fusion,
                                                                              fusion_mask,
                                                                              None)
        fusion_max = F.max_pool2d(fusion[:, 0, :, :, :], fusion.shape[3])
        fusion_avg = F.avg_pool2d(fusion[:, 0, :, :, :], fusion.shape[3])

        fusion_mask_max = F.max_pool2d(fusion_mask[:, 0, :, :, :], fusion_mask.shape[3])
        fusion_mask_avg = F.avg_pool2d(fusion_mask[:, 0, :, :, :], fusion_mask.shape[3])

        fusion_attention = torch.sigmoid(
            extractor['mlp_image'](fusion_max) + extractor['mlp_image'](fusion_avg)
        )
        fusion_mask_attention = torch.sigmoid(
            extractor['mlp_mask'](fusion_mask_max) + extractor['mlp_mask'](fusion_mask_avg)
        )

        return fusion * torch.unsqueeze(fusion_attention, dim=1), \
               fusion_mask * torch.unsqueeze(fusion_mask_attention, dim=1)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_enc_bn:
            for i in range(self.net_depth):
                if isinstance(self.encoder[i].partial_conv.bn, nn.BatchNorm2d):
                    self.encoder[i].eval()
