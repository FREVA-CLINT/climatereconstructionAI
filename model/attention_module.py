import torch
from torch import nn
import torch.nn.functional as F
import sys

sys.path.append('./')
from model.encoder_decoder import EncoderBlock, lstm_to_batch, batch_to_lstm
import config as cfg


class AttentionEncoderBlock(nn.Module):
    def __init__(self, conv_config, kernel, stride, activation, lstm):
        super().__init__()
        self.partial_conv_enc = EncoderBlock(conv_config=conv_config, kernel=kernel, stride=stride,
                                             activation=activation, lstm=lstm)
        self.channel_attention_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=conv_config['in_channels'],
                      out_features=conv_config['in_channels'] // cfg.channel_reduction_rate),
            nn.ReLU(),
            nn.Linear(in_features=conv_config['in_channels'] // cfg.channel_reduction_rate,
                      out_features=conv_config['in_channels']),
        )
        self.spatial_attention_block = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(7, 7), padding=(3, 3)),
            nn.Sigmoid()
        )

    def forward(self, h_rea, h_rea_mask, rea_lstm_state, h, h_mask):
        batch_size = h_rea.shape[0]

        # convert lstm steps to batch dimension
        h_rea = lstm_to_batch(h_rea)
        h_rea_mask = lstm_to_batch(h_rea_mask)
        h = lstm_to_batch(h)
        h_mask = lstm_to_batch(h_mask)

        # channel attention
        channel_attention = self.forward_channel_attention(h_rea, h_rea_mask)
        # spatial attention
        spatial_attention = self.spatial_attention_block(
            torch.cat([torch.max(h, keepdim=True, dim=1)[0],
                       torch.mean(h, keepdim=True, dim=1)], dim=1)
        )
        attention = channel_attention * spatial_attention
        attention_mask = torch.ones_like(attention)

        # convert batches to lstm dimension
        h_rea = batch_to_lstm(h_rea, batch_size)
        h_rea_mask = batch_to_lstm(h_rea_mask, batch_size)
        attention = batch_to_lstm(attention, batch_size)
        attention_mask = batch_to_lstm(attention_mask, batch_size)

        h_rea, h_rea_mask, rea_lstm_state = self.partial_conv_enc(h_rea, h_rea_mask, rea_lstm_state)

        return h_rea, h_rea_mask, rea_lstm_state, attention, attention_mask

    def forward_channel_attention(self, input, input_mask):
        attention_max = F.max_pool2d(input, input.shape[2])
        attention_avg = F.avg_pool2d(input, input.shape[2])
        total_attention = torch.sigmoid(
            self.channel_attention_block(attention_max) + self.channel_attention_block(attention_avg)
        )
        return input * torch.unsqueeze(torch.unsqueeze(total_attention, dim=2), dim=2)
