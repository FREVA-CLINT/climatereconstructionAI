import warnings

import torch
import torchmetrics.image as t_metrics

from .. import config as cfg
from ..loss import get_loss

@torch.no_grad()
def get_metrics(img_mask, loss_mask, output, gt, setname):
    metric_settings = {
        'valid': {},
        'hole': {},
        'tv': {},
        'var': {},
        'feature': {
            'outputs': ['style', 'prc']
        },
        'StructuralSimilarityIndexMeasure': {
            'torchmetric_settings': {}
        },
        'UniversalImageQualityIndex': {
            'torchmetric_settings': {'reset_real_features': True}
        },
        'FrechetInceptionDistance': {
            'torchmetric_settings': {'requires_image': True, 'feature': 64, 'reset_real_features': True,
                                     'subset_size': cfg.batch_size}
        },  
        'KernelInceptionDistance': {
            'torchmetric_settings': {'requires_image': True, 'feature': 64, 'reset_real_features': True,
                                     'subset_size': cfg.batch_size},
            'outputs': ['mu', 'std']
        }
    }
    

    metric_dict = {}
    metrics = cfg.val_metrics
    
    loss_metric_dict = dict(zip(metrics,[1]*len(metrics)))
    if 'feature' in metrics:
        loss_metric_dict.update(dict(zip(['style', 'prc'],[1,1])))

    loss_comp = get_loss.LossComputation(loss_metric_dict)

    loss_metrics = loss_comp(img_mask, loss_mask, output, gt)
    loss_metrics['total'] = loss_metrics['total'].item()

    for metric in metrics:
        settings = metric_settings[metric]

        if 'valid' in metric:
            metric_dict[f'metric/{setname}/valid'] = loss_metrics['valid']

        elif 'hole' in metric:
            metric_dict[f'metric/{setname}/hole'] = loss_metrics['hole']

        elif 'tv' in metric:
            metric_dict[f'metric/{setname}/tv'] = loss_metrics['tv']

        elif 'feature' in metric:
            metric_dict[f'metric/{setname}/style'] = loss_metrics['style']
            metric_dict[f'metric/{setname}/prc'] = loss_metrics['prc']

        else:
            data = get_loss.prepare_data_dict(img_mask, loss_mask, output, gt, ['mask','output','gt'])
            metric_outputs = calculate_metric(metric, data['mask'], data['output'], data['gt'],
                                              torchmetrics_settings=settings['torchmetric_settings'])

            if len(metric_outputs) > 1:
                for k, metric_name in enumerate(settings['outputs']):
                    metric_dict[f'metric/{setname}/{metric}_{metric_name}'] = metric_outputs[k]
            else:
                metric_dict[f'metric/{setname}/{metric}'] = metric_outputs[0]

    return metric_dict


def calculate_metric(name_expr, mask, output, gt, domain='valid', torchmetrics_settings={}, outputs=[]):
    metric_str = [m for m in t_metrics.__dict__.keys() if (name_expr == m)]

    if len(metric_str) == 0:
        metric_str = [m for m in t_metrics.__dict__.keys() if (name_expr in m)]
        if len(metric_str) > 1:
            warnings.warn('found multiple hits for metric name {}. Will use {}'.format(name_expr, metric_str[0]))

    assert len(metric_str) > 0, 'metric {} not found in torchmetrics.image. Maybe torch-fidelity is missing.'.format(
        name_expr)

    metric = t_metrics.__dict__[metric_str[0]](**torchmetrics_settings).to(cfg.device)

    if domain == 'valid':
        pred = mask * output
        target = mask * gt
    elif domain == 'hole':
        pred = (1 - mask) * output
        target = (1 - mask) * gt
    elif domain == 'comp_infill':
        pred = mask * gt + (1 - mask) * output
        target = gt

    results_ch = []

    for channel in range(output.shape[1]):
        pred_ch = torch.unsqueeze(pred[:, channel, :, :], dim=1)
        target_ch = torch.unsqueeze(target[:, channel, :, :], dim=1)

        if 'requires_image' in torchmetrics_settings and torchmetrics_settings['requires_image']:

            pred_ch = torch.cat([pred_ch] * 3, 1)
            target_ch = torch.cat([target_ch] * 3, 1)

            setattr(metric, 'normalize', True)
            metric.update((pred_ch), real=True)
            metric.update((target_ch), real=False)
            result_ch = metric.compute()
        else:
            result_ch = metric(pred_ch, target_ch)

        results_ch.append(result_ch)

    if isinstance(results_ch[0], tuple):
        result_out = [0] * len(results_ch[0])
        for result_ch in results_ch:
            for k, result_ch_arg in enumerate(result_ch):
                result_out[k] += result_ch_arg
    else:
        result_out = 0
        for result_ch in results_ch:
            result_out += result_ch
        result_out = [result_out]

    return result_out
