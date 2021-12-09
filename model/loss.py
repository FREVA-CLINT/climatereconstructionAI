import torch
import torch.nn as nn
import torch.nn.functional as F


def gram_matrix(feat):
    # https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram


def total_variation_loss(image):
    # shift one pixel and get difference (for both x and y direction)
    loss = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) + \
           torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
    return loss


class InpaintingLoss(nn.Module):
    def __init__(self, extractor=None):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.extractor = extractor

    def forward(self, input, mask, output, gt):
        loss_dict = {
            'hole': 0.0,
            'valid': 0.0,
            'prc': 0.0,
            'style': 0.0,
            'tv': 0.0
        }
        # create output_comp
        output_comp = mask * input + (1 - mask) * output

        # calculate loss for all channels
        for c in range(output.shape[1]):
            # only select first channel
            mask_ch = torch.unsqueeze(mask[:, c, :, :], dim=1)
            gt_ch = torch.unsqueeze(gt[:, c, :, :], dim=1)
            output_ch = torch.unsqueeze(output[:, c, :, :], dim=1)
            output_comp_ch = torch.unsqueeze(output_comp[:, c, :, :], dim=1)

            # define different loss functions from output and output_comp
            loss_dict['hole'] += self.l1((1 - mask_ch) * output_ch, (1 - mask_ch) * gt_ch)
            loss_dict['valid'] += self.l1(mask_ch * output_ch, mask_ch * gt_ch)

            # define different loss function from features from output and output_comp
            if self.extractor:
                feat_output = self.extractor(torch.cat([output_ch] * 3, 1))
                feat_output_comp = self.extractor(torch.cat([output_comp_ch] * 3, 1))
                feat_gt = self.extractor(torch.cat([gt_ch] * 3, 1))
            else:
                feat_output = torch.cat([output_ch] * 3, 1).permute(1, 0, 2, 3).unsqueeze(1)
                feat_output_comp = torch.cat([output_comp_ch] * 3, 1).permute(1, 0, 2, 3).unsqueeze(1)
                feat_gt = torch.cat([gt_ch] * 3, 1).permute(1, 0, 2, 3).unsqueeze(1)

            loss_dict['prc'] += 0.0
            loss_dict['style'] += 0.0
            for i in range(3):
                loss_dict['prc'] += self.l1(feat_output[i], feat_gt[i])
                loss_dict['prc'] += self.l1(feat_output_comp[i], feat_gt[i])
                loss_dict['style'] += self.l1(gram_matrix(feat_output[i]),
                                              gram_matrix(feat_gt[i]))
                loss_dict['style'] += self.l1(gram_matrix(feat_output_comp[i]),
                                              gram_matrix(feat_gt[i]))

            loss_dict['tv'] += total_variation_loss(output_comp_ch)

        return loss_dict


class WeightedCrossEntropyLoss(nn.Module):
    # weight should be a 1D Tensor assigning weight to each of the classes.
    def __init__(self, weight=None, LAMBDA=None):
        super().__init__()
        self._weight = weight
        self._lambda = LAMBDA

    def forward(self, input, target, mask):
        target = target.squeeze(1)
        class_index = torch.zeros_like(target).long()
        for i, threshold in enumerate([0, 5, 7, 8, 10]):
            class_index[target >= threshold] = i

        error = F.cross_entropy(input, class_index, weight=self._weight, reduction='none')
        error = error.unsqueeze(2)
        return torch.mean(error * (mask))


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, targets):
        loss = -torch.mean(targets * torch.log(outputs) +
                           (1 - targets) * torch.log(1 - outputs))
        return loss
