import torch
import torch.nn as nn


def total_variation_loss(image):
    # shift one pixel and get difference (for both x and y direction)
    loss = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) + \
           torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
    return loss


class GeneratorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, discr_output):
        loss_dict = {}
        loss_dict['gan'] = self.mse_loss(0, discr_output)
        return loss_dict


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.BCELoss()

    def forward(self, prediction, label):
        loss = self.mse_loss(prediction, label)
        return loss
