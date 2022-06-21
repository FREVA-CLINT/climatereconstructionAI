import torch
from torch import nn
import torch.nn.functional as F
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


class BaseConvRNN(nn.Module):
    def __init__(self, out_channels, b_h_w,
                 h2h_kernel=(3, 3), h2h_dilate=(1, 1),
                 i2h_kernel=(3, 3), i2h_stride=(1, 1),
                 i2h_pad=(1, 1), i2h_dilate=(1, 1),
                 act_type=torch.tanh):
        super(BaseConvRNN, self).__init__()
        self.out_channels = out_channels
        self._h2h_kernel = h2h_kernel
        assert (self._h2h_kernel[0] % 2 == 1) and (self._h2h_kernel[1] % 2 == 1), \
            "Only support odd number, get h2h_kernel= %s" % str(h2h_kernel)
        self._h2h_pad = (h2h_dilate[0] * (h2h_kernel[0] - 1) // 2,
                         h2h_dilate[1] * (h2h_kernel[1] - 1) // 2)
        self._h2h_dilate = h2h_dilate
        self._i2h_kernel = i2h_kernel
        self._i2h_stride = i2h_stride
        self._i2h_pad = i2h_pad
        self._i2h_dilate = i2h_dilate
        self._act_type = act_type
        assert len(b_h_w) == 3
        i2h_dilate_ksize_h = 1 + (self._i2h_kernel[0] - 1) * self._i2h_dilate[0]
        i2h_dilate_ksize_w = 1 + (self._i2h_kernel[1] - 1) * self._i2h_dilate[1]
        self._batch_size, self._height, self._width = b_h_w
        self._state_height = (self._height + 2 * self._i2h_pad[0] - i2h_dilate_ksize_h) \
                             // self._i2h_stride[0] + 1
        self._state_width = (self._width + 2 * self._i2h_pad[1] - i2h_dilate_ksize_w) \
                            // self._i2h_stride[1] + 1
        self._curr_states = None
        self._counter = 0


class TrajGRUBlock(BaseConvRNN):
    # b_h_w: input feature map size
    def __init__(self, in_channels, out_channels, b_h_w, zoneout=0.0, L=5,
                 i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                 h2h_kernel=(5, 5), h2h_dilate=(1, 1)):
        super(TrajGRUBlock, self).__init__(out_channels=out_channels,
                                           b_h_w=b_h_w,
                                           h2h_kernel=h2h_kernel,
                                           h2h_dilate=h2h_dilate,
                                           i2h_kernel=i2h_kernel,
                                           i2h_pad=i2h_pad,
                                           i2h_stride=i2h_stride)
        self._L = L
        self._zoneout = zoneout

        self.i2h = nn.Conv2d(in_channels=in_channels,
                             out_channels=self.out_channels * 3,
                             kernel_size=self._i2h_kernel,
                             stride=self._i2h_stride,
                             padding=self._i2h_pad,
                             dilation=self._i2h_dilate)

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
                                    out_channels=self._L * 2,
                                    kernel_size=(5, 5),
                                    stride=1,
                                    padding=(2, 2))

        self.ret = nn.Conv2d(in_channels=self.out_channels * self._L,
                             out_channels=self.out_channels * 3,
                             kernel_size=(1, 1),
                             stride=1)

    def _flow_generator(self, inputs, states):
        if inputs is not None:
            i2f_conv1 = self.i2f_conv1(inputs)
        else:
            i2f_conv1 = None
        h2f_conv1 = self.h2f_conv1(states)
        f_conv1 = i2f_conv1 + h2f_conv1 if i2f_conv1 is not None else h2f_conv1
        f_conv1 = self._act_type(f_conv1)

        flows = self.flows_conv(f_conv1)
        flows = torch.split(flows, 2, dim=1)
        return flows

    def forward(self, inputs, states=None):
        # switch batch and sequence dimensions
        inputs = inputs.transpose(0, 1)

        if states is None:
            states = torch.zeros((inputs.size(1), self.out_channels, self._state_height,
                                  self._state_width), dtype=torch.float).to(cfg.device)

        S, B, C, H, W = inputs.size()
        i2h = self.i2h(torch.reshape(inputs, (-1, C, H, W)))
        i2h = torch.reshape(i2h, (S, B, i2h.size(1), i2h.size(2), i2h.size(3)))
        i2h_slice = torch.split(i2h, self.out_channels, dim=2)

        prev_h = states
        outputs = []
        for i in range(S):
            flows = self._flow_generator(inputs[i, ...], prev_h)
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
                new_mem = self._act_type(i2h_slice[2][i, ...] + reset_gate * h2h_slice[2])
            else:
                reset_gate = torch.sigmoid(h2h_slice[0])
                update_gate = torch.sigmoid(h2h_slice[1])
                new_mem = self._act_type(reset_gate * h2h_slice[2])
            next_h = update_gate * prev_h + (1 - update_gate) * new_mem
            if self._zoneout > 0.0:
                mask = F.dropout2d(torch.zeros_like(prev_h), p=self._zoneout)
                next_h = torch.where(mask, next_h, prev_h)
            outputs.append(next_h)
            prev_h = next_h

        # switch batch and sequence dimensions
        outputs = torch.stack(outputs)
        outputs = outputs.transpose(0, 1)
        return outputs, next_h
