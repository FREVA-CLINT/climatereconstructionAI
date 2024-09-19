import torch

from .feature_loss import FeatureLoss
from .hole_loss import HoleLoss
from .total_variation_loss import TotalVariationLoss
from .valid_loss import ValidLoss
from .kldiv_loss import KLDivLoss
from .extreme_loss import ExtremeLoss
from .. import config as cfg
from ..utils.featurizer import VGG16FeatureExtractor


def prepare_data_dict(mask, output, latent_dist, gt, tensor_keys):
    data_dict = dict(zip(list(tensor_keys), [None] * len(tensor_keys)))

    data_dict['mask'] = mask[:, 0]
    data_dict['output'] = output[:, cfg.recurrent_steps]
    data_dict['latent_dist'] = latent_dist
    data_dict['gt'] = gt[:, 0]

    if 'comp' in tensor_keys:
        data_dict['comp'] = data_dict['mask'] * data_dict['gt'] + (1 - data_dict['mask']) * data_dict['output']

    return data_dict


class loss_criterion(torch.nn.Module):
    def __init__(self, lambda_dict):
        super().__init__()

        self.criterions = []
        self.tensors = ['output', 'latent_dist', 'gt', 'mask']
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

                elif loss == 'kldiv':
                    self.criterions.append(KLDivLoss().to(cfg.device))

                elif loss == '-extreme' or loss == '+extreme':
                    self.criterions.append(ExtremeLoss().to(cfg.device))

    def forward(self, mask, output, latent_dist, gt):

        data_dict = prepare_data_dict(mask, output, latent_dist, gt, self.tensors)

        loss_dict = {}
        for criterion in self.criterions:
            loss_dict.update(criterion(data_dict))

        loss_dict["total"] = 0
        for loss, lambda_value in cfg.lambda_dict.items():
            if lambda_value > 0 and loss in loss_dict.keys():
                loss_w_lambda = loss_dict[loss] * lambda_value
                loss_dict["total"] += loss_w_lambda
                loss_dict[loss] = loss_w_lambda

        return loss_dict

class ModularizedFunction(torch.nn.Module):
    def __init__(self, forward_op):
        super().__init__()
        self.forward_op = forward_op

    def forward(self, *args, **kwargs):
        return self.forward_op(*args, **kwargs)

class CriterionParallel(torch.nn.Module):
    def __init__(self, criterion):
        super().__init__()
        if not isinstance(criterion, torch.nn.Module):
            criterion = ModularizedFunction(criterion)
        self.criterion = torch.nn.DataParallel(criterion)

    def forward(self, *args, **kwargs):
        multi_dict = self.criterion(*args, **kwargs)
        for key in multi_dict.keys():
            multi_dict[key] = multi_dict[key].mean()
        return multi_dict

class LossComputation(torch.nn.Module):
    def __init__(self, lambda_dict):
        super().__init__()
        if cfg.multi_gpus:
            self.criterion = CriterionParallel(loss_criterion(lambda_dict))
        else:
            self.criterion = loss_criterion(lambda_dict)

    def forward(self, mask, output, latent_dist, gt):
        loss_dict = self.criterion(mask, output, latent_dist, gt)
        return loss_dict
