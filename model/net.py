import torch
import torch.nn as nn
import sys

sys.path.append('./')
from model.attention_module import ConvolutionalAttentionBlock
from model.encoder_decoder import EncoderBlock, DecoderBlock


class PConvLSTM(nn.Module):
    def __init__(self, radar_img_size=512, radar_enc_dec_layers=4, radar_pool_layers=4, radar_in_channels=1,
                 radar_out_channels=1,
                 rea_img_sizes=[], rea_enc_layers=[], rea_pool_layers=[], rea_in_channels=[],
                 lstm=True):
        super().__init__()

        self.freeze_enc_bn = False
        self.radar_enc_dec_layers = radar_enc_dec_layers
        self.radar_pool_layers = radar_pool_layers
        self.rea_enc_layers = rea_enc_layers
        self.rea_pool_layers = rea_pool_layers
        self.net_depth = radar_enc_dec_layers + radar_pool_layers
        self.lstm = lstm

        # define encoding blocks for attention channels
        if rea_img_sizes:
            self.attention_block = ConvolutionalAttentionBlock(rea_img_sizes, rea_enc_layers,
                                                               rea_pool_layers, rea_in_channels, lstm)
        attention_channels = len(rea_img_sizes) * radar_img_size

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

        # define decoding pooling layers
        decoding_layers = []
        for i in range(self.radar_pool_layers):
            if i == 0:
                in_channels = radar_img_size + radar_img_size + attention_channels
            else:
                in_channels = radar_img_size + radar_img_size
            decoding_layers.append(DecoderBlock(
                in_channels=in_channels,
                out_channels=radar_img_size,
                image_size=radar_img_size // (2 ** (self.net_depth - i - 1)),
                kernel=(3, 3), stride=(1, 1), activation=nn.LeakyReLU(), lstm=lstm))

        # define decoding layers
        for i in range(1, self.radar_enc_dec_layers):
            if i == 1 and self.radar_pool_layers == 0:
                in_channels = radar_img_size // (2 ** (i - 1)) + radar_img_size // (2 ** i) + attention_channels
            else:
                in_channels = radar_img_size // (2 ** (i - 1)) + radar_img_size // (2 ** i)
            decoding_layers.append(
                DecoderBlock(
                    in_channels=in_channels,
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

        # forward attention block
        if hasattr(self, 'attention_block'):
            h, h_mask = self.attention_block(input[:, 1:, :, :, :, :], input_mask[:, 1:, :, :, :, :], h, h_mask)

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
                if isinstance(self.encoder[i].partial_conv.bn, nn.BatchNorm2d):
                    self.encoder[i].eval()
