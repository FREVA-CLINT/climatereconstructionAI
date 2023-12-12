import torch
import torch.nn as nn

from .attention_module import AttentionEncoderBlock
from .conv_configs import init_enc_conv_configs, init_dec_conv_configs, \
    init_enc_conv_configs_orig, init_dec_conv_configs_orig
from .encoder_decoder import EncoderBlock, DecoderBlock
from .bounds_scaler import constrain_bounds
from .. import config as cfg


class GaussActivation(nn.Module):
    def __init__(self, activation_mu=nn.Identity()):
        super().__init__() 
        self.activation_mu = activation_mu
        self.activation_std = nn.Softplus()

    def forward(self, input):
        mu = self.activation_mu(input[:,[0]])
        std = self.activation_std(input[:,[1]])
        
        return torch.concat((mu,std),1)

class CRAINet(nn.Module):
    def __init__(self, img_size_source, img_size_target, enc_dec_layers=4, pool_layers=4, in_channels=1, out_channels=1,
                 conv_factor=16,fusion_img_size=None, fusion_enc_layers=None, fusion_pool_layers=None, fusion_in_channels=0,
                 bounds=None, n_filters=None, skip_layers=True, upsampling_mode='linear',attention=False, n_channel_steps=1, predict_residual=True,
                 recurrent_steps=0, lstm_steps=0, gru_steps=0, dropout=0, pixel_shuffle=False):

        super().__init__()

        self.bounds = bounds
        self.freeze_enc_bn = False
        self.net_depth = enc_dec_layers + pool_layers
        self.n_channel_steps = n_channel_steps
        self.predict_residual = predict_residual
        self.attention=attention
        self.recurrent_steps = recurrent_steps
        self.lstm_steps = lstm_steps
        self.gru_steps = gru_steps
        self.in_channels = in_channels

        if 'gauss' in cfg.lambda_dict:
            out_channels = out_channels*2
        
        self.out_channels = out_channels

        # initialize channel inputs and outputs and image size for encoder and decoder
        if n_filters is None:
            enc_conv_configs = init_enc_conv_configs(conv_factor, img_size_target, enc_dec_layers,
                                                     pool_layers, in_channels)
            dec_conv_configs = init_dec_conv_configs(conv_factor, img_size_target, enc_dec_layers,
                                                     pool_layers, in_channels,
                                                     out_channels, skip_layers)
        else:
            enc_conv_configs = init_enc_conv_configs_orig(img_size_target, enc_dec_layers,
                                                          out_channels, n_filters)
            dec_conv_configs = init_dec_conv_configs_orig(img_size_target, enc_dec_layers,
                                                          out_channels, n_filters)

        if ((torch.tensor(img_size_target)-torch.tensor(img_size_source))>0).any():
            self.upsample = nn.Upsample(size=(img_size_target), mode=upsampling_mode)
        else:
            self.upsample = nn.Identity()

        if attention:
            self.attention_depth = fusion_enc_layers + fusion_pool_layers
            attention_enc_conv_configs = init_enc_conv_configs(conv_factor, fusion_img_size, fusion_enc_layers,
                                                               fusion_pool_layers, fusion_in_channels)

            attention_layers = []
            for i in range(self.attention_depth):
                if i < fusion_enc_layers:
                    kernel = (5, 5)
                else:
                    kernel = (3, 3)
                attention_layers.append(AttentionEncoderBlock(
                    conv_config=attention_enc_conv_configs[i],
                    kernel=kernel, stride=(2, 2), activation=nn.ReLU()))

                # adjust skip channels for decoder
                if i != self.attention_depth - 1:
                    dec_conv_configs[i]['out_channels'] += \
                        attention_enc_conv_configs[self.attention_depth - i - 1]['in_channels']
                dec_conv_configs[i]['skip_channels'] += \
                    skip_layers * attention_enc_conv_configs[self.attention_depth - i - 1]['in_channels']
                dec_conv_configs[i]['in_channels'] += \
                    attention_enc_conv_configs[self.attention_depth - i - 1]['out_channels']

            self.attention_module = nn.ModuleList(attention_layers)

        elif fusion_img_size is not None:
            self.channel_fusion_depth = fusion_enc_layers + fusion_pool_layers
            enc_conv_configs[self.net_depth - self.channel_fusion_depth]['in_channels'] += fusion_in_channels
            dec_conv_configs[self.channel_fusion_depth - 1]['skip_channels'] += skip_layers * fusion_in_channels

        # define encoding layers
        encoding_layers = []
        for i in range(0, self.net_depth):
            encoding_layers.append(EncoderBlock(
                conv_config=enc_conv_configs[i],
                kernel=enc_conv_configs[i]['kernel'], stride=(2, 2), activation=nn.ReLU()))
        self.encoder = nn.ModuleList(encoding_layers)

        # define decoding layers
        decoding_layers = []
        for i in range(self.net_depth):
            if i == self.net_depth - 1:
                activation = None
                bias = True
                if hasattr(cfg, 'lambda_dict'):
                    if 'gauss' in cfg.lambda_dict:
                        if cfg.lambda_dict['gauss']>0:
                            activation = GaussActivation()
                            bias = True
            else:
                activation = nn.LeakyReLU()
                bias = False
            decoding_layers.append(DecoderBlock(
                conv_config=dec_conv_configs[i],
                kernel=dec_conv_configs[i]['kernel'], stride=(1, 1), activation=activation, bias=bias, skip_layers=skip_layers))
        self.decoder = nn.ModuleList(decoding_layers)

        if bounds is not None:
            self.binder = constrain_bounds(bounds)
        else:
            self.binder = None

    def forward(self, input, input_mask=None):
        # create lists for skip connections
        # We split the inputs in case we use the attention module with different image dimension
        h_index = self.n_channel_steps
        
        output_mask = True
        if input_mask is None:
            input_mask = torch.ones_like(input, device=input.device)
            output_mask=False

        input = torch.stack([self.upsample(input[:,k]) for k in range(cfg.n_recurrent_steps)],dim=1)
        input_mask = torch.stack([self.upsample(input_mask[:,k]) for k in range(cfg.n_recurrent_steps)],dim=1)

        if self.predict_residual:
            input_base = input.clone()

        hs = [input[:, :, :h_index, :, :]]
        hs_mask = [input_mask[:, :, :h_index, :, :]]
        recurrent_states = []

        fusion_input = input[:, :, h_index:, :, :]
        fusion_input_mask = input_mask[:, :, h_index:, :, :]
        h_fusion = fusion_input
        h_fusion_mask = fusion_input_mask
        hs_fusion = []
        hs_fusion_mask = []
        recurrent_fusion_states = []

        # forward pass encoding layers
        for i in range(self.net_depth):
            if h_fusion.size()[1] != 0 and hs[i].shape[3] == h_fusion.shape[3]:
                if not self.attention:
                    hs[i] = torch.cat([hs[i], h_fusion], dim=2)
                    hs_mask[i] = torch.cat([hs_mask[i], h_fusion_mask], dim=2)

            h, h_mask, recurrent_state = self.encoder[i](hs[i],
                                                         hs_mask[i],
                                                         None)

            # execute attention module if configured
            if self.attention and i >= (self.net_depth - self.attention_depth):
                attention_index = i - (self.net_depth - self.attention_depth)
                h_fusion, h_fusion_mask, attention_recurrent_state, attention = \
                    self.attention_module[attention_index](h_fusion,
                                                           h_fusion_mask,
                                                           None,
                                                           h)
                hs_fusion.append(attention)
                hs_fusion_mask.append(h_fusion_mask)
                recurrent_fusion_states.append(attention_recurrent_state)

            # save hidden states for skip connections
            hs.append(h)
            recurrent_states.append(recurrent_state)
            hs_mask.append(h_mask)

        # concat attentions
        if self.attention:
            hs[self.net_depth - self.attention_depth] = torch.cat(
                [hs[self.net_depth - self.attention_depth], fusion_input], dim=2)
            hs_mask[self.net_depth - self.attention_depth] = torch.cat(
                [hs_mask[self.net_depth - self.attention_depth], fusion_input_mask], dim=2)
            for i in range(self.attention_depth):
                hs[i + (self.net_depth - self.attention_depth) + 1] = torch.cat(
                    [hs[i + (self.net_depth - self.attention_depth) + 1], hs_fusion[i]], dim=2)
                hs_mask[i + (self.net_depth - self.attention_depth) + 1] = torch.cat(
                    [hs_mask[i + (self.net_depth - self.attention_depth) + 1], hs_fusion_mask[i]], dim=2)

                if self.lstm_steps:
                    recurrent_state_h, recurrent_state_c = recurrent_states[i + (self.net_depth - self.attention_depth)]
                    recurrent_fusion_state_h, recurrent_fusion_state_c = recurrent_fusion_states[i]
                    recurrent_state_h = torch.cat([recurrent_state_h, recurrent_fusion_state_h], dim=1)
                    recurrent_state_c = torch.cat([recurrent_state_c, recurrent_fusion_state_c], dim=1)
                    recurrent_states[i + (self.net_depth - self.attention_depth)] = (recurrent_state_h,
                                                                                     recurrent_state_c)
                elif self.gru_steps:
                    recurrent_state_h = recurrent_states[i + (self.net_depth - self.attention_depth)]
                    recurrent_fusion_state_h = recurrent_fusion_states[i]
                    recurrent_state_h = torch.cat([recurrent_state_h, recurrent_fusion_state_h], dim=1)
                    recurrent_states[i + (self.net_depth - self.attention_depth)] = recurrent_state_h

        # # reverse all hidden states
        # if cfg.recurrent_steps:
        #     for i in range(self.net_depth):
        #         hs[i] = torch.flip(hs[i], (1,))
        #         hs_mask[i] = torch.flip(hs_mask[i], (1,))

        h, h_mask = hs[self.net_depth], hs_mask[self.net_depth]

        # forward pass decoding layers
        for i in range(self.net_depth):
            if self.recurrent_steps:
                h, h_mask, recurrent_state = self.decoder[i](h, hs[self.net_depth - i - 1],
                                                             h_mask, hs_mask[self.net_depth - i - 1],
                                                             recurrent_states[self.net_depth - 1 - i])
            else:
                h, h_mask, recurrent_state = self.decoder[i](h, hs[self.net_depth - i - 1],
                                                             h_mask, hs_mask[self.net_depth - i - 1],
                                                             None)
        if self.predict_residual:
            if self.in_channels != self.out_channels:
                h[:,:,:self.in_channels] = h[:,:,:self.in_channels] + input_base
            else:
                h+=input_base
 

        if self.bounds is not None:
            h = self.binder.scale(h)
        
        # return last element of output from last decoding layer
        if output_mask:
            return h, input_mask
        else:
            return h

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_enc_bn:
            for i in range(self.net_depth):
                if hasattr(self.encoder[i].partial_conv, "bn"):
                    if isinstance(self.encoder[i].partial_conv.bn, nn.BatchNorm2d):
                        self.encoder[i].eval()
