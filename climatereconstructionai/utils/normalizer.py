import numpy as np
from torchvision import transforms
from .. import config as cfg


def img_normalization(img_data, train_stats):
    img_std, img_mean, img_znorm = [], [], []
    if cfg.normalize_data:
        for i in range(len(img_data)):
            if train_stats is None:
                if cfg.lazy_load:
                    img_mean.append(img_data[i].chunk('auto').mean(skipna=True).compute().item())
                    img_std.append(img_data[i].chunk('auto').std(skipna=True).compute().item())
                else:
                    img_mean.append(np.nanmean(img_data[i]))
                    img_std.append(np.nanstd(img_data[i]))
            else:
                img_mean.append(train_stats["mean"][i])
                img_std.append(train_stats["std"][i])
            img_znorm.append(transforms.Normalize(mean=[img_mean[-1]], std=[img_std[-1]]))

    return img_mean, img_std, img_znorm


def renormalize(img_data, img_mean, img_std):
    return img_std * img_data + img_mean


def bnd_normalization(img_mean, img_std):

    bounds = np.ones((cfg.out_channels, 2)) * np.inf

    if cfg.n_target_data == 0:
        mean_val, std_val = img_mean[:cfg.n_output_data], img_std[:cfg.n_output_data]
    else:
        mean_val, std_val = img_mean[-cfg.n_target_data:], img_std[-cfg.n_target_data:]

    k = 0
    for bound in (cfg.min_bounds, cfg.max_bounds):

        for i in range(cfg.n_output_data):
            idx = range(i * cfg.n_pred_steps, (i + 1) * cfg.n_pred_steps)
            bounds[idx, k] = bound[i]
            if cfg.normalize_data:
                bounds[idx, k] = (bounds[idx, k] - mean_val[i]) / std_val[i]

        k += 1

    return bounds
