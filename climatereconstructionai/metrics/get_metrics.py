import warnings
import torch

from ..loss.hole_loss import HoleLoss
from ..loss.valid_loss import ValidLoss
from ..loss.feature_loss import FeatureLoss
from ..loss.var_loss import VarLoss
from ..loss.total_variation_loss import TotalVariationLoss

from ..utils.featurizer import VGG16FeatureExtractor
from .. import config as cfg

import torchmetrics.image as t_metrics


@torch.no_grad()
def get_metrics(img_mask, loss_mask, output, gt, setname):

    metric_settings = {
        'valid': {},
        'hole': {},
        'tv':{},
        'var':{},
        'feature': {
            'outputs': ['style','prc']
            },
        'StructuralSimilarityIndexMeasure': {
            'torchmetric_settings':{}
            },
        'UniversalImageQualityIndex' : {
            'torchmetric_settings':{'reset_real_features':True}
            },
        'FrechetInceptionDistance' : {
            'torchmetric_settings': {'requires_image': True, 'feature':64, 'reset_real_features':True, 'subset_size':cfg.batch_size}
            },
        'KernelInceptionDistance': {
            'torchmetric_settings': {'requires_image': True, 'feature':64, 'reset_real_features':True, 'subset_size':cfg.batch_size},
                                    'outputs': ['mu','std']
                                    }
    }

    
    mask = img_mask[:, cfg.recurrent_steps, cfg.gt_channels, :, :]

    if loss_mask is not None:
        mask += loss_mask
        assert ((mask == 0) | (mask == 1)).all(), "Not all values in mask are zeros or ones!"

    metric_dict = {}
    if setname=='train':
        metrics = cfg.train_metrics
    elif setname=='val':
        metrics = cfg.val_metrics
    elif setname=='test':
        metrics = cfg.test_metrics

    for metric in metrics:
        settings = metric_settings[metric]

        if 'valid' in metric:
            val_loss = ValidLoss().to(cfg.device)
            metric_output = val_loss(mask, output[:, cfg.recurrent_steps, :, :, :],
                            gt[:, cfg.recurrent_steps, cfg.gt_channels, :, :])
            metric_dict[f'metric/{setname}/valid']=metric_output['valid']

        elif 'hole' in metric:
            val_loss = HoleLoss().to(cfg.device)
            metric_output = val_loss(mask, output[:, cfg.recurrent_steps, :, :, :],
                            gt[:, cfg.recurrent_steps, cfg.gt_channels, :, :])
            metric_dict[f'metric/{setname}/hole']=metric_output['hole']

        elif 'tv' in metric:
            val_loss = TotalVariationLoss().to(cfg.device)
            metric_output = val_loss(mask, output[:, cfg.recurrent_steps, :, :, :],
                            gt[:, cfg.recurrent_steps, cfg.gt_channels, :, :])
            metric_dict[f'metric/{setname}/tv']=metric_output['tv']

        elif 'var' in metric:
            var_loss = VarLoss().to(cfg.device)
            metric_output = var_loss(mask, output[:, cfg.recurrent_steps, :, :, :],
                            gt[:, cfg.recurrent_steps, cfg.gt_channels, :, :])
            metric_dict[f'metric/{setname}/var']=metric_output['var']

        elif 'feature' in metric:
            feat_loss = FeatureLoss(VGG16FeatureExtractor()).to(cfg.device)
            metric_output = feat_loss(mask, output[:, cfg.recurrent_steps, :, :, :],
                            gt[:, cfg.recurrent_steps, cfg.gt_channels, :, :])
            metric_dict[f'metric/{setname}/style']=metric_output['style']
            metric_dict[f'metric/{setname}/prc']=metric_output['prc']

        else:
            metric_outputs = calculate_metric(metric,mask, output[:, cfg.recurrent_steps, :, :, :],
                        gt[:, cfg.recurrent_steps, cfg.gt_channels, :, :], torchmetrics_settings=settings['torchmetric_settings'])
            
            if len(metric_outputs)>1:
                for k, metric_name in enumerate(settings['outputs']):
                    metric_dict[f'metric/{setname}/{metric}_{metric_name}']=metric_outputs[k]
            else:
                metric_dict[f'metric/{setname}/{metric}']=metric_outputs[0]

    return metric_dict


def calculate_metric(name_expr, mask, output, gt, domain='valid', torchmetrics_settings={}, outputs=[]):

    metric_str = [m for m in t_metrics.__dict__.keys() if (name_expr == m)]

    if len(metric_str)==0:
        metric_str = [m for m in t_metrics.__dict__.keys() if (name_expr in m)]
        if len(metric_str)>1:
            warnings.warn('found multiple hits for metric name {}. Will use {}'.format(name_expr, metric_str[0]))

    assert len(metric_str)>0, 'metric {} not found in torchmetrics.image. Maybe torch-fidelity is missing.'.format(name_expr)

    
    metric = t_metrics.__dict__[metric_str[0]](**torchmetrics_settings).to(cfg.device)


    if domain=='valid':
        pred = mask * output
        target = mask * gt
    elif domain=='hole':
        pred = (1 - mask) * output
        target = (1 - mask) * gt
    elif domain=='comp_infill':
        pred = mask * gt + (1 - mask) * output
        target = gt

    results_ch = []

    for channel in range(output.shape[1]):
        pred_ch = torch.unsqueeze(pred[:, channel, :, :], dim=1)
        target_ch = torch.unsqueeze(target[:, channel, :, :], dim=1)

        if 'requires_image' in torchmetrics_settings and torchmetrics_settings['requires_image']:

            pred_ch = torch.cat([pred_ch] * 3, 1)
            target_ch = torch.cat([target_ch] * 3, 1)

            setattr(metric,'normalize',True)
            metric.update((pred_ch), real=True)
            metric.update((target_ch), real=False)
            result_ch = metric.compute()
        else:
            result_ch = metric(pred_ch, target_ch)
        
        results_ch.append(result_ch)

    if isinstance(results_ch[0], tuple):
        result_out = [0]*len(results_ch[0])
        for result_ch in results_ch:
            for k, result_ch_arg in enumerate(result_ch):
                result_out[k] += result_ch_arg
    else:
        result_out = 0
        for result_ch in results_ch:
            result_out += result_ch
        result_out = [result_out]            

    return result_out

