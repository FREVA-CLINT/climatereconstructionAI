import torch

from .gauss_loss import GaussLoss
from .feature_loss import FeatureLoss
from .hole_loss import HoleLoss
from .total_variation_loss import TotalVariationLoss
from .valid_loss import ValidLoss
from .. import config as cfg
from ..utils.featurizer import VGG16FeatureExtractor


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


def prepare_data_dict(img_mask, loss_mask, output, gt, tensor_keys): 
    data_dict = dict(zip(list(tensor_keys),[None]*len(tensor_keys)))

    mask = img_mask
    loss_mask = img_mask
    if loss_mask is not None:
        mask += loss_mask
        mask[mask < 0] = 0
        mask[mask > 1] = 1
        assert ((mask == 0) | (mask == 1)).all(), "Not all values in mask are zeros or ones!"

    if output.shape[2]>gt.shape[2]:
        if 'gauss' in tensor_keys:
            data_dict['gauss'] = output[:, cfg.recurrent_steps, :, :, :]
        output = output[:, cfg.recurrent_steps, [0], :, :]
    else:
        output = output[:, cfg.recurrent_steps, :, :, :]

    mask = mask[:, cfg.recurrent_steps, :, :, :]
    gt = gt[:, cfg.recurrent_steps, cfg.gt_channels, :, :]

    data_dict['mask'] = mask
    data_dict['output'] = output
    data_dict['gt'] = gt

    if 'comp' in tensor_keys:
        data_dict['comp'] = mask * gt + (1 - mask) * output

    return data_dict      


class loss_criterion(torch.nn.Module):
    def __init__(self, lambda_dict):
        super().__init__()

        self.criterions = torch.nn.ModuleDict()
        self.tensors = ['output', 'gt', 'mask']
        
        for loss, lambda_ in lambda_dict.items():
            if lambda_ > 0:
                if loss == 'style' or loss == 'prc':
                    criterion = FeatureLoss(VGG16FeatureExtractor()).to(cfg.device)
                    self.tensors.append('comp')

                elif loss == 'valid':
                    criterion = ValidLoss().to(cfg.device)
                    self.tensors.append('valid')

                elif loss == 'hole':
                    criterion = HoleLoss().to(cfg.device)
                    self.tensors.append('hole')

                elif loss == 'tv':
                    criterion = TotalVariationLoss().to(cfg.device)
                    if 'comp' not in self.tensors:
                        self.tensors.append('comp')

                elif loss == 'gauss':
                    criterion = GaussLoss().to(cfg.device)
                    self.tensors.append('gauss')
                
                if not criterion in self.criterions.values():
                    self.criterions[loss] = criterion


    def forward(self, img_mask, loss_mask, output, gt):
        
        data_dict = prepare_data_dict(img_mask, loss_mask, output, gt, self.tensors)

        loss_dict = {}
        for _, criterion in self.criterions.items():
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
            self.criterion = CriterionParallel(loss_criterion(lambda_dict))
        else:
            self.criterion = loss_criterion(lambda_dict)

    def forward(self, img_mask, loss_mask, output, gt):
        loss_dict = self.criterion(img_mask, loss_mask, output ,gt)
        return loss_dict
