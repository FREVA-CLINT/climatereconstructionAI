import torch
import torch.nn.functional as F


def gram_matrix(feat):
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram


def total_variation_loss(image):
    loss = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) +\
        torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
    return loss


def conv_variance(data, k=11):
    weights = torch.ones((1, 1, k, k), dtype=torch.float, device=data.device) / (k * k)

    with torch.no_grad():
        exp = F.conv2d(torch.pow(data, 2), weights, padding='valid')
        exp2 = torch.pow(F.conv2d(data, weights, padding='valid'), 2)

    return (exp - exp2)
