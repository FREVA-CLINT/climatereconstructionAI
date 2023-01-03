import torch
import torch.nn.functional as F
from torch import nn

from .encoder_decoder import EncoderBlock, sequence_to_batch, batch_to_sequence
from .. import config as cfg


class AttentionEncoderBlock(nn.Module):
    def __init__(self, conv_config, kernel, stride, activation):
        super().__init__()
        self.partial_conv_enc = EncoderBlock(conv_config=conv_config, kernel=kernel, stride=stride,
                                             activation=activation)
        self.channel_attention_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=conv_config['out_channels'],
                      out_features=conv_config['out_channels'] // cfg.channel_reduction_rate),
            nn.ReLU(),
            nn.Linear(in_features=conv_config['out_channels'] // cfg.channel_reduction_rate,
                      out_features=conv_config['out_channels']),
        )
        self.spatial_attention_block = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(7, 7), padding=(3, 3)),
            nn.Sigmoid()
        )

    def forward(self, h_fusion, h_fusion_mask, recurrent_fusion_state, h):
        h_fusion, h_fusion_mask, recurrent_fusion_state = self.partial_conv_enc(h_fusion, h_fusion_mask,
                                                                                recurrent_fusion_state)
        batch_size = h_fusion.shape[0]

        # convert sequence steps to batch dimension
        h_fusion = sequence_to_batch(h_fusion)
        h = sequence_to_batch(h)

        # channel attention
        channel_attention = self.forward_channel_attention(h_fusion)
        # spatial attention
        spatial_attention = self.spatial_attention_block(
            torch.cat([torch.max(h, keepdim=True, dim=1)[0],
                       torch.mean(h, keepdim=True, dim=1)], dim=1)
        )
        attention = channel_attention * spatial_attention

        # convert batches to sequence dimension
        h_fusion = batch_to_sequence(h_fusion, batch_size)
        attention = batch_to_sequence(attention, batch_size)

        return h_fusion, h_fusion_mask, recurrent_fusion_state, attention

    def forward_channel_attention(self, input):
        attention_max = F.max_pool2d(input, input.shape[2])
        attention_avg = F.avg_pool2d(input, input.shape[2])
        total_attention = torch.sigmoid(
            self.channel_attention_block(attention_max) + self.channel_attention_block(attention_avg)
        )
        return input * torch.unsqueeze(torch.unsqueeze(total_attention, dim=2), dim=2)
