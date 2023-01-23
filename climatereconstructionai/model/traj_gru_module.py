import torch
import torch.nn.functional as F
from torch import nn

from .. import config as cfg


def wrap(input, flow):
    B, C, H, W = input.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1).to(cfg.device)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W).to(cfg.device)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()
    vgrid = grid + flow

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)
    output = torch.nn.functional.grid_sample(input, vgrid)
    return output


class TrajGRUBlock(nn.Module):
    def __init__(self, in_channels, out_channels, img_size, zoneout=0.0, L=5,
                 i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_padding=(1, 1), i2h_dilation=(1, 1),
                 h2h_kernel=(5, 5), h2h_dilation=(1, 1), activation=torch.tanh):
        super().__init__()
        self.out_channels = out_channels

        self.h2h_kernel = h2h_kernel
        assert (self.h2h_kernel[0] % 2 == 1) and (self.h2h_kernel[1] % 2 == 1), \
            "Only support odd number, get h2h_kernel= %s" % str(h2h_kernel)
        self.h2h_padding = (h2h_dilation[0] * (h2h_kernel[0] - 1) // 2,
                            h2h_dilation[1] * (h2h_kernel[1] - 1) // 2)
        self.h2h_dilation = h2h_dilation
        self.i2h_kernel = i2h_kernel
        self.i2h_stride = i2h_stride
        self.i2h_padding = i2h_padding
        self.i2h_dilation = i2h_dilation
        self.activation = activation
        i2h_dilate_ksize_h = 1 + (self.i2h_kernel[0] - 1) * self.i2h_dilation[0]
        i2h_dilate_ksize_w = 1 + (self.i2h_kernel[1] - 1) * self.i2h_dilation[1]
        self.state_height = (img_size[0] + 2 * self.i2h_padding[0] - i2h_dilate_ksize_h) // self.i2h_stride[0] + 1
        self.state_width = (img_size[1] + 2 * self.i2h_padding[1] - i2h_dilate_ksize_w) // self.i2h_stride[1] + 1
        self.L = L
        self.zoneout = zoneout

        self.i2h = nn.Conv2d(in_channels=in_channels,
                             out_channels=self.out_channels * 3,
                             kernel_size=self.i2h_kernel,
                             stride=self.i2h_stride,
                             padding=self.i2h_padding,
                             dilation=self.i2h_dilation)

        self.i2f_conv1 = nn.Conv2d(in_channels=in_channels,
                                   out_channels=32,
                                   kernel_size=(5, 5),
                                   stride=1,
                                   padding=(2, 2),
                                   dilation=(1, 1))

        self.h2f_conv1 = nn.Conv2d(in_channels=self.out_channels,
                                   out_channels=32,
                                   kernel_size=(5, 5),
                                   stride=1,
                                   padding=(2, 2),
                                   dilation=(1, 1))

        self.flows_conv = nn.Conv2d(in_channels=32,
                                    out_channels=self.L * 2,
                                    kernel_size=(5, 5),
                                    stride=1,
                                    padding=(2, 2))

        self.ret = nn.Conv2d(in_channels=self.out_channels * self.L,
                             out_channels=self.out_channels * 3,
                             kernel_size=(1, 1),
                             stride=1)

    def flow_generator(self, inputs, states):
        if inputs is not None:
            i2f_conv1 = self.i2f_conv1(inputs)
        else:
            i2f_conv1 = None
        h2f_conv1 = self.h2f_conv1(states)
        f_conv1 = i2f_conv1 + h2f_conv1 if i2f_conv1 is not None else h2f_conv1
        f_conv1 = self.activation(f_conv1)

        flows = self.flows_conv(f_conv1)
        flows = torch.split(flows, 2, dim=1)
        return flows

    def forward(self, inputs, gru_state=None):
        batch_size = inputs.shape[0]
        gru_steps = inputs.shape[1]

        if gru_state is None:
            gru_state = torch.zeros((batch_size, self.out_channels, self.state_height,
                                     self.state_width), dtype=torch.float).to(cfg.device)

        i2h = self.i2h(torch.reshape(inputs, (-1, inputs.shape[2], inputs.shape[3], inputs.shape[4])))
        i2h = torch.reshape(i2h, (gru_steps, batch_size, i2h.size(1), i2h.size(2), i2h.size(3)))
        i2h_slice = torch.split(i2h, self.out_channels, dim=2)

        prev_h = gru_state
        outputs = []
        for i in range(gru_steps):
            flows = self.flow_generator(inputs[:, i, :, :, :], prev_h)
            wrapped_data = []
            for j in range(len(flows)):
                flow = flows[j]
                wrapped_data.append(wrap(prev_h, -flow))
            wrapped_data = torch.cat(wrapped_data, dim=1)
            h2h = self.ret(wrapped_data)
            h2h_slice = torch.split(h2h, self.out_channels, dim=1)
            if i2h_slice is not None:
                reset_gate = torch.sigmoid(i2h_slice[0][i, ...] + h2h_slice[0])
                update_gate = torch.sigmoid(i2h_slice[1][i, ...] + h2h_slice[1])
                new_mem = self.activation(i2h_slice[2][i, ...] + reset_gate * h2h_slice[2])
            else:
                reset_gate = torch.sigmoid(h2h_slice[0])
                update_gate = torch.sigmoid(h2h_slice[1])
                new_mem = self.activation(reset_gate * h2h_slice[2])
            next_h = update_gate * prev_h + (1 - update_gate) * new_mem
            if self.zoneout > 0.0:
                mask = F.dropout2d(torch.zeros_like(prev_h), p=self.zoneout)
                next_h = torch.where(mask, next_h, prev_h)
            outputs.append(next_h)
            prev_h = next_h

        return torch.stack(outputs, dim=1), next_h
