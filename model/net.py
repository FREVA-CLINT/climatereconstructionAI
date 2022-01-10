import torch
import torch.nn as nn
import sys

sys.path.append('./')
from model.attention_module import AttentionEncoderBlock
from model.encoder_decoder import EncoderBlock, DecoderBlock
import config as cfg


class PConvLSTM(nn.Module):
    def __init__(self, radar_img_size=512, radar_enc_dec_layers=4, radar_pool_layers=4, radar_in_channels=1,
                 radar_out_channels=1,
                 rea_img_size=256, rea_enc_layers=256, rea_pool_layers=256, rea_in_channels=0,
                 lstm=True):
        super().__init__()

        self.freeze_enc_bn = False
        self.radar_enc_dec_layers = radar_enc_dec_layers
        self.radar_pool_layers = radar_pool_layers
        self.net_depth = radar_enc_dec_layers + radar_pool_layers
        self.rea_enc_layers = rea_enc_layers
        self.rea_pool_layers = rea_pool_layers
        self.attention_depth = rea_enc_layers + rea_pool_layers
        self.lstm = lstm

        attention_enc_channels = [0] * self.net_depth
        attention_dec_channels = [0] * self.net_depth
        if rea_img_size:
            if not cfg.attention:
                for i in range(self.net_depth):
                    if radar_img_size // (2 ** i) == rea_img_size:
                        attention_enc_channels[i] = rea_in_channels
                attention_dec_channels = attention_enc_channels
            else:
                attention_module_layers = []
                for i in range(self.rea_enc_layers):
                    if i == 0:
                        in_channels = rea_in_channels
                    else:
                        in_channels = rea_img_size // (2 ** (self.rea_enc_layers - i))
                    attention_module_layers.append(AttentionEncoderBlock(
                        in_channels=in_channels,
                        out_channels=rea_img_size // (2 ** (self.rea_enc_layers - i - 1)),
                        image_size=rea_img_size // (2 ** i),
                        kernel=(5, 5), stride=(2, 2), activation=nn.ReLU(), lstm=lstm))
                    if i == self.attention_depth - 1:
                        attention_dec_channels[self.net_depth - 1] = rea_img_size + rea_img_size // 2
                    elif i == 0:
                        attention_dec_channels[self.net_depth - self.attention_depth - 1] = 0
                    else:
                        attention_dec_channels[self.net_depth - (self.attention_depth - i) - 1] = rea_img_size // (2 ** (self.rea_enc_layers - i - 1))

                # define encoding pooling layers
                for i in range(self.rea_pool_layers):
                    attention_module_layers.append(AttentionEncoderBlock(
                        in_channels=rea_img_size,
                        out_channels=rea_img_size,
                        image_size=rea_img_size // (2 ** (self.rea_enc_layers + i)),
                        kernel=(3, 3), stride=(2, 2), activation=nn.ReLU(), lstm=lstm))
                    if i == self.attention_depth - self.rea_enc_layers - 1:
                        attention_dec_channels[self.net_depth - 1] = rea_img_size + rea_img_size
                    elif i == 0 and self.rea_enc_layers == 0:
                        attention_dec_channels[self.net_depth - self.attention_depth - 1] = 0
                    else:
                        attention_dec_channels[self.net_depth - (self.attention_depth - i) - 1] = rea_img_size // (
                                    2 ** (self.rea_enc_layers - i - 1))
                self.attention_module = nn.ModuleList(attention_module_layers)
                print(attention_dec_channels)

        # define encoding layers
        encoding_layers = []
        encoding_layers.append(
            EncoderBlock(
                in_channels=radar_in_channels + attention_enc_channels[0],
                out_channels=radar_img_size // (2 ** (self.radar_enc_dec_layers - 1)),
                image_size=radar_img_size,
                kernel=(7, 7), stride=(2, 2), activation=nn.ReLU(), lstm=lstm))
        for i in range(1, self.radar_enc_dec_layers):
            if i == self.radar_enc_dec_layers - 1:
                encoding_layers.append(EncoderBlock(
                    in_channels=radar_img_size // (2 ** (self.radar_enc_dec_layers - i)) + attention_enc_channels[i],
                    out_channels=radar_img_size // (2 ** (self.radar_enc_dec_layers - i - 1)),
                    image_size=radar_img_size // (2 ** i),
                    kernel=(3, 3), stride=(2, 2), activation=nn.ReLU(), lstm=lstm))
            else:
                encoding_layers.append(EncoderBlock(
                    in_channels=radar_img_size // (2 ** (self.radar_enc_dec_layers - i)) + attention_enc_channels[i],
                    out_channels=radar_img_size // (2 ** (self.radar_enc_dec_layers - i - 1)),
                    image_size=radar_img_size // (2 ** i),
                    kernel=(5, 5), stride=(2, 2), activation=nn.ReLU(), lstm=lstm))
        # define encoding pooling layers
        for i in range(self.radar_pool_layers):
            encoding_layers.append(EncoderBlock(
                in_channels=radar_img_size + attention_enc_channels[self.radar_enc_dec_layers + i],
                out_channels=radar_img_size,
                image_size=radar_img_size // (2 ** (self.radar_enc_dec_layers + i)),
                kernel=(3, 3), stride=(2, 2), activation=nn.ReLU(), lstm=lstm))
        self.encoder = nn.ModuleList(encoding_layers)

        # define decoding pooling layers
        decoding_layers = []
        for i in range(self.radar_pool_layers):
            decoding_layers.append(DecoderBlock(
                in_channels=radar_img_size + radar_img_size + attention_dec_channels[self.net_depth - i - 1],
                out_channels=radar_img_size,
                image_size=radar_img_size // (2 ** (self.net_depth - i - 1)),
                kernel=(3, 3), stride=(1, 1), activation=nn.LeakyReLU(), lstm=lstm))

        # define decoding layers
        for i in range(1, self.radar_enc_dec_layers):
            decoding_layers.append(
                DecoderBlock(
                    in_channels=radar_img_size // (2 ** (i - 1)) + radar_img_size // (2 ** i)
                                + attention_dec_channels[self.net_depth - self.radar_pool_layers - i],
                    out_channels=radar_img_size // (2 ** i),
                    image_size=radar_img_size // (2 ** (self.net_depth - self.radar_pool_layers - i)),
                    kernel=(3, 3), stride=(1, 1), activation=nn.LeakyReLU(), lstm=lstm))
        decoding_layers.append(
            DecoderBlock(
                in_channels=radar_img_size // (2 ** (self.radar_enc_dec_layers - 1)) + radar_in_channels
                            + attention_dec_channels[0],
                out_channels=radar_out_channels,
                image_size=radar_img_size,
                kernel=(3, 3), stride=(1, 1), activation=None, lstm=lstm, bn=False, bias=True,
                init_in_channels=radar_in_channels, init_out_channels=radar_out_channels))
        self.decoder = nn.ModuleList(decoding_layers)

    def forward(self, input, input_mask, rea_input, rea_input_mask):
        # create lists for skip connections
        h = input
        h_mask = input_mask
        hs = [h]
        hs_mask = [h_mask]
        lstm_states = []

        h_rea = rea_input
        h_rea_mask = rea_input_mask
        attentions = []
        attentions_mask = []

        # forward pass encoding layers
        for i in range(self.net_depth):
            if len(h_rea.size()) != 0 and h.shape[3] == h_rea.shape[3]:
                if not cfg.attention:
                    hs[i] = torch.cat([hs[i], h_rea], dim=2)
                    hs_mask[i] = torch.cat([hs_mask[i], h_rea_mask], dim=2)

            h, h_mask, lstm_state = self.encoder[i](hs[i],
                                                    hs_mask[i],
                                                    None)
            # save hidden states for skip connections
            hs.append(h)
            lstm_states.append(lstm_state)
            hs_mask.append(h_mask)

            # execute attention module if configured
            if cfg.attention and i >= (self.net_depth - self.attention_depth):
                rea_index = i - (self.net_depth - self.attention_depth)
                h_rea, h_rea_mask, rea_lstm_state, attention, attention_mask = \
                    self.attention_module[rea_index](h_rea,
                                                     h_rea_mask,
                                                     None,
                                                     h,
                                                     h_mask)
                attentions.append(attention)
                attentions_mask.append(attention_mask)
        # concat attentions
        if attentions:
            for i in range(self.attention_depth):
                hs[i + (self.net_depth - self.attention_depth) + 1] = torch.cat([hs[i + (self.net_depth - self.attention_depth) + 1], attentions[i]], dim=2)
                hs_mask[i + (self.net_depth - self.attention_depth) + 1] = torch.cat([hs_mask[i + (self.net_depth - self.attention_depth) + 1], attentions_mask[i]], dim=2)


        # reverse all hidden states
        for i in range(self.net_depth):
            hs[i] = torch.flip(hs[i], (1,))
            hs_mask[i] = torch.flip(hs_mask[i], (1,))

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
                if isinstance(self.encoder[i].partial_conv.bn, nn.BatchNorm2d):
                    self.encoder[i].eval()
