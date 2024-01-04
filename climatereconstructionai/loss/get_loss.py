import torch

from .feature_loss import FeatureLoss
from .hole_loss import HoleLoss
from .total_variation_loss import TotalVariationLoss
from .valid_loss import ValidLoss
from .. import config as cfg
from ..utils.featurizer import VGG16FeatureExtractor


def prepare_data_dict(mask, output, gt, tensor_keys):
    data_dict = dict(zip(list(tensor_keys),[None]*len(tensor_keys)))

    data_dict['mask'] = mask[:, 0]
    data_dict['output'] = output[:, cfg.recurrent_steps]
    data_dict['gt'] = gt[:, 0]

    if 'comp' in tensor_keys:
        data_dict['comp'] = data_dict['mask'] * data_dict['gt'] + (1 - data_dict['mask']) * data_dict['output']

    return data_dict


class loss_criterion(torch.nn.Module):
    def __init__(self, lambda_dict):
        super().__init__()

        self.criterions = []
        self.tensors = ['output', 'gt', 'mask']
        style_added = False

        for loss, lambda_ in lambda_dict.items():
            if lambda_ > 0:
                if (loss == 'style' or loss == 'prc') and not style_added:
                    self.criterions.append(FeatureLoss(VGG16FeatureExtractor()).to(cfg.device))
                    self.tensors.append('comp')
                    style_added = True

                elif loss == 'valid':
                    self.criterions.append(ValidLoss().to(cfg.device))
                    self.tensors.append('valid')

                elif loss == 'hole':
                    self.criterions.append(HoleLoss().to(cfg.device))
                    self.tensors.append('hole')

                elif loss == 'tv':
                    self.criterions.append(TotalVariationLoss().to(cfg.device))
                    if 'comp' not in self.tensors:
                        self.tensors.append('comp')


    def forward(self, mask, output, gt):

        data_dict = prepare_data_dict(mask, output, gt, self.tensors)

        loss_dict = {}
        for criterion in self.criterions:
            loss_dict.update(criterion(data_dict))

        loss_dict["total"] = 0
        for loss, lambda_value in cfg.lambda_dict.items():
            if lambda_value > 0 and loss in loss_dict.keys():
                loss_w_lambda = loss_dict[loss] * lambda_value
                loss_dict["total"] += loss_w_lambda
                loss_dict[loss] = loss_w_lambda.item()

        return loss_dict


class LossComputation(torch.nn.Module):
    def __init__(self, lambda_dict):
        super().__init__()
        if cfg.multi_gpus:
            self.criterion = torch.nn.DataParallel(loss_criterion(lambda_dict))
        else:
            self.criterion = loss_criterion(lambda_dict)

    def forward(self, mask, output, gt):
        loss_dict = self.criterion(mask, output ,gt)
        return loss_dict