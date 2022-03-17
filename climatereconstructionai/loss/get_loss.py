
from .. import config as cfg

def get_loss(criterion, lambda_dict, mask, output, gt, writer, iter_index, setname):
    loss_dict = criterion(mask[:, cfg.lstm_steps, cfg.gt_channels, :, :],
                          output[:, cfg.lstm_steps, :, :, :],
                          gt[:, cfg.lstm_steps, cfg.gt_channels, :, :])

    losses = {"total": 0.0}
    for key, factor in lambda_dict.items():
        value = factor * loss_dict[key]
        losses[key] = value
        losses["total"] += value

    if cfg.log_interval and (iter_index + 1) % cfg.log_interval == 0:
        for key in losses.keys():
            writer.add_scalar('loss_{:s}-{:s}'.format(setname,key), losses[key], iter_index + 1)

    return losses["total"]
