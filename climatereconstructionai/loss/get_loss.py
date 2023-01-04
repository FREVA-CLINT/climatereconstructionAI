import torch

from .. import config as cfg


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


def get_loss(criterion, lambda_dict, img_mask, loss_mask, output, gt, writer, iter_index, setname):
    if cfg.multi_gpus:
        loss_func = CriterionParallel(criterion)
    else:
        loss_func = criterion

    mask = img_mask[:, cfg.recurrent_steps, cfg.gt_channels, :, :]
    if loss_mask is not None:
        mask += loss_mask
        assert ((mask == 0) | (mask == 1)).all(), "Not all values in mask are zeros or ones!"

    loss_dict = loss_func(mask, output[:, cfg.recurrent_steps, :, :, :], gt[:, cfg.recurrent_steps, cfg.gt_channels, :, :])

    losses = {"total": 0.0}
    for key, factor in lambda_dict.items():
        value = factor * loss_dict[key]
        losses[key] = value
        losses["total"] += value

    if cfg.log_interval and iter_index % cfg.log_interval == 0:
        for key in losses.keys():
            writer.add_scalar('loss_{:s}-{:s}'.format(setname, key), losses[key], iter_index)

    return losses["total"]
