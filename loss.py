import torch
import torch.nn as nn


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
        self.crossentropy = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        self.extractor = extractor

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
        #loss_dict['ce'] = 0.0
        #for i in range(gt.shape[0]):
        #    print(torch.squeeze(output_comp[i], dim=0).shape)
        #    loss_dict['ce'] += self.crossentropy(torch.squeeze(output_comp[i], dim=0), torch.squeeze(gt[i], dim=0))
        #loss_dict['mse'] = self.mse(output_comp, gt)
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
