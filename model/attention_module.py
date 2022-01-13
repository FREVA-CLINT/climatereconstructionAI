import torch
from torch import nn
import torch.nn.functional as F
import sys

sys.path.append('./')
from model.encoder_decoder import EncoderBlock
import config as cfg


class AttentionEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, image_size, kernel, stride, activation, lstm):
        super().__init__()
        self.partial_conv_enc = EncoderBlock(in_channels=in_channels, out_channels=out_channels, image_size=image_size,
                                             kernel=kernel, stride=stride, activation=activation, lstm=lstm)
        self.channel_attention_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=out_channels, out_features=out_channels // cfg.channel_reduction_rate),
            nn.ReLU(),
            nn.Linear(in_features=out_channels // cfg.channel_reduction_rate, out_features=out_channels),
        )
        self.spatial_attention_block = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(7, 7), padding=(3, 3)),
            nn.Sigmoid()
        )

    def forward(self, h_rea, h_rea_mask, rea_lstm_state, h, h_mask):
        h_rea, h_rea_mask, rea_lstm_state = self.partial_conv_enc(h_rea, h_rea_mask, rea_lstm_state)

        # channel attention
        channel_attention = self.forward_channel_attention(h_rea, h_rea_mask)
        # spatial attention
        spatial_attention = torch.unsqueeze(self.spatial_attention_block(
            torch.cat([torch.max(h[:, 0, :, :, :], keepdim=True, dim=1)[0],
                       torch.mean(h[:, 0, :, :, :], keepdim=True, dim=1)], dim=1)
        ), dim=1)
        attention = channel_attention * spatial_attention
        attention_mask = torch.ones_like(attention)

        return h_rea, h_rea_mask, rea_lstm_state, attention, attention_mask

    def forward_channel_attention(self, input, input_mask):
        attention_max = F.max_pool2d(input[:, 0, :, :, :], input.shape[3])
        attention_avg = F.avg_pool2d(input[:, 0, :, :, :], input.shape[3])
        total_attention = torch.sigmoid(
            self.channel_attention_block(attention_max) + self.channel_attention_block(attention_avg)
        )
        return input * torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(total_attention, dim=2), dim=2), dim=1)
