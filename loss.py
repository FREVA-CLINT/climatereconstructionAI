import numpy as np
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
    def __init__(self, extractor):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.extractor = extractor

    def forward(self, input, mask, output, gt, device):
        # create output_comp
        output_comp = mask * input + (1 - mask) * output

        # define different loss functions from output and output_comp
        loss_dict = {}
        loss_dict['hole'] = self.l1((1 - mask) * output, (1 - mask) * gt)
        loss_dict['valid'] = self.l1(mask * output, mask * gt)

        # define different loss function from features from output and output_comp
        feat_output = self.extractor(torch.cat([output] * 3, 1))
        feat_output_comp = self.extractor(torch.cat([output_comp] * 3, 1))
        feat_gt = self.extractor(torch.cat([gt] * 3, 1))

        loss_dict['prc'] = 0.0
        loss_dict['style'] = 0.0
        for i in range(3):
            loss_dict['prc'] += self.l1(feat_output[i], feat_gt[i])
            loss_dict['prc'] += self.l1(feat_output_comp[i], feat_gt[i])
            loss_dict['style'] += self.l1(gram_matrix(feat_output[i]),
                                          gram_matrix(feat_gt[i]))
            loss_dict['style'] += self.l1(gram_matrix(feat_output_comp[i]),
                                          gram_matrix(feat_gt[i]))

        loss_dict['tv'] = total_variation_loss(output_comp)

        return loss_dict


class PrevNextInpaintingLoss(InpaintingLoss):
    def forward(self, input, mask, output, gt, device):
        # get mid indexed element
        mid_index = torch.tensor([(input.shape[1] // 2)],dtype=torch.long).to(device)
        input = torch.index_select(input, dim=1, index=mid_index)
        gt = torch.index_select(gt, dim=1, index=mid_index)
        mask = torch.index_select(mask, dim=1, index=mid_index)

        # create output_comp
        output_comp = mask * input + (1 - mask) * output

        # define different loss functions from output and output_comp
        loss_dict = {}
        loss_dict['hole'] = self.l1((1 - mask) * output, (1 - mask) * gt)
        loss_dict['valid'] = self.l1(mask * output, mask * gt)

        # define different loss function from features from output and output_comp
        feat_output = self.extractor(torch.cat([output] * 3, 1))
        feat_output_comp = self.extractor(torch.cat([output_comp] * 3, 1))
        feat_gt = self.extractor(torch.cat([gt] * 3, 1))

        loss_dict['prc'] = 0.0
        loss_dict['style'] = 0.0
        for i in range(3):
            loss_dict['prc'] += self.l1(feat_output[i], feat_gt[i])
            loss_dict['prc'] += self.l1(feat_output_comp[i], feat_gt[i])
            loss_dict['style'] += self.l1(gram_matrix(feat_output[i]),
                                          gram_matrix(feat_gt[i]))
            loss_dict['style'] += self.l1(gram_matrix(feat_output_comp[i]),
                                          gram_matrix(feat_gt[i]))

        loss_dict['tv'] = total_variation_loss(output_comp)

        return loss_dict


class LSTMInpaintingLoss(InpaintingLoss):
    def forward(self, input, mask, output, gt):
        loss_dict = {}
        loss_dict['hole'] = 0.0
        loss_dict['valid'] = 0.0
        loss_dict['prc'] = 0.0
        loss_dict['style'] = 0.0
        loss_dict['tv'] = 0.0

        for t in range(input.shape[1]):
            input_part = input[:,t,:,:,:]
            mask_part = mask[:,t,:,:,:]
            output_part = output[:,t,:,:,:]
            gt_part = gt[:,t,:,:,:]

            # create output_comp
            output_comp = mask_part * input_part + (1 - mask_part) * output_part

            # define different loss function from features from output and output_comp
            feat_output = self.extractor(torch.cat([output_part] * 3, 1))
            feat_output_comp = self.extractor(torch.cat([output_comp] * 3, 1))
            feat_gt = self.extractor(torch.cat([gt_part] * 3, 1))

            # add loss functions from output and output_comp
            loss_dict['hole'] += self.l1((1 - mask_part) * output_part, (1 - mask_part) * gt_part)
            loss_dict['valid'] += self.l1(mask_part * output_part, mask_part * gt_part)

            for i in range(3):
                loss_dict['prc'] += self.l1(feat_output[i], feat_gt[i])
                loss_dict['prc'] += self.l1(feat_output_comp[i], feat_gt[i])
                loss_dict['style'] += self.l1(gram_matrix(feat_output[i]),
                                              gram_matrix(feat_gt[i]))
                loss_dict['style'] += self.l1(gram_matrix(feat_output_comp[i]),
                                              gram_matrix(feat_gt[i]))

            loss_dict['tv'] += total_variation_loss(output_comp)

        return loss_dict


class PrecipitationInpaintingLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, input, mask, output, gt, device):
        loss_dict = {}
        # get mid indexed element
        mid_index = torch.tensor([(input.shape[1] // 2)],dtype=torch.long).to(device)
        input = torch.index_select(input, dim=1, index=mid_index)
        gt = torch.index_select(gt, dim=1, index=mid_index)
        mask = torch.index_select(mask, dim=1, index=mid_index)
        # create output_comp
        output_comp = mask * input + (1 - mask) * output

        c = 20
        L20 = torch.sum(
            (torch.sigmoid((output - 20) * c) - torch.sigmoid((gt - 20) * c)) ** 2
        )
        L30 = torch.sum(
            (torch.sigmoid((output - 30) * c) - torch.sigmoid((gt - 30) * c)) ** 2
        )
        loss_dict['SSL-OUT'] = -(L20 + L30)

        L20 = torch.sum(
            (torch.sigmoid((output_comp - 20) * c) - torch.sigmoid((gt - 20) * c)) ** 2
        )
        L30 = torch.sum(
            (torch.sigmoid((output_comp - 30) * c) - torch.sigmoid((gt - 30) * c)) ** 2
        )
        loss_dict['SSL-OUT-COMP'] = -(L20 + L30)

        loss_dict['L1-OUT'] = self.l1(output, gt)
        loss_dict['L1-OUT-COMP'] = self.l1(output_comp, gt)
        loss_dict['HOLE'] = self.l1((1 - mask) * output, (1 - mask) * gt)
        loss_dict['STYLE-OUT'] = self.l1(gram_matrix(output), gram_matrix(gt))
        loss_dict['STYLE-OUT-COMP'] = self.l1(gram_matrix(output_comp), gram_matrix(gt))

        return loss_dict


def rainfall_to_pixel(rainfall_intensity, a=58.53, b=1.56):
    """Convert the rainfall intensity to pixel values
    Parameters
    ----------
    rainfall_intensity : np.ndarray
    a : float32, optional
    b : float32, optional
    Returns
    -------
    pixel_vals : np.ndarray
    """
    dBR = np.log10(rainfall_intensity) * 10.0
    # dBZ = 10b log(R) +10log(a)
    dBZ = dBR * b + 10.0 * np.log10(a)
    pixel_vals = (dBZ + 10.0) / 70.0
    return pixel_vals


def dBZ_to_rainfall(dBZ, a=58.53, b=1.56):
    return np.power(10, (dBZ - 10 * np.log10(a))/(10*b))


class CrossEntropyLoss(nn.Module):
    # weight should be a 1D Tensor assigning weight to each of the classes.
    def __init__(self, thresholds, weight=None, LAMBDA=None):
        super().__init__()
        self._weight = weight
        self._lambda = LAMBDA
        self._thresholds = thresholds

    # input: output prob, S*B*C*H*W
    # target: S*B*1*H*W, original data, range [0, 1]
    # mask: S*B*1*H*W
    def forward(self, input, target, mask):
        # F.cross_entropy should be B*C*S*H*W
        input = input.permute((1, 2, 0, 3, 4))
        # B*S*H*W
        target = target.permute((1, 2, 0, 3, 4)).squeeze(1)
        class_index = torch.zeros_like(target).long()
        thresholds = [0.0] + rainfall_to_pixel(self._thresholds).tolist()
        # print(thresholds)
        for i, threshold in enumerate(thresholds):
            class_index[target >= threshold] = i
        error = F.cross_entropy(input, class_index, self._weight, reduction='none')
        if self._lambda is not None:
            B, S, H, W = error.size()

            w = torch.arange(1.0, 1.0 + S * self._lambda, self._lambda)
            if torch.cuda.is_available():
                w = w.to(error.get_device())
                # B, H, W, S
            error = (w * error.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        # S*B*1*H*W
        error = error.permute(1, 0, 2, 3).unsqueeze(2)
        return torch.mean(error * mask.float())
