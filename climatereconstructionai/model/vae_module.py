import torch
import torch.nn as nn
import torch.nn.functional as F

class VAEBlock(nn.Module):
    def __init__(self, conv_config, n_steps, z_dim):
        super().__init__()

        self.h_shape = [-1, n_steps, conv_config['out_channels']] + conv_config['rec_size']
        self.h_dim = self.h_shape[1] * self.h_shape[2] * self.h_shape[3] * self.h_shape[4]
        self.efc1 = nn.Linear(self.h_dim, z_dim)
        self.efc2 = nn.Linear(self.h_dim, z_dim)
        self.dfc1 = nn.Linear(z_dim, self.h_dim)


    def forward(self, input):

        input = input.view(-1, self.h_dim)
        mu = self.efc1(input)
        logvar = self.efc2(input)
        std = torch.exp(logvar * 0.5)
        eps = torch.randn_like(std)
        z = mu + std * eps
        output = F.relu(self.dfc1(z))
        output = output.view(self.h_shape)

        return output, (mu, logvar)
