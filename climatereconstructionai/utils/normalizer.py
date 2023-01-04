import numpy as np
from torchvision import transforms
from .. import config as cfg


def img_normalization(img_data):
    img_std, img_mean, img_tf = [], [], []
    for i in range(len(img_data)):
        img_mean.append(np.nanmean(img_data[i]))
        img_std.append(np.nanstd(img_data[i]))
        img_tf.append(transforms.Normalize(mean=[img_mean[-1]], std=[img_std[-1]]))

    return img_mean, img_std, img_tf


def renormalize(img_data, img_mean, img_std):
    return img_std * img_data + img_mean


def bnd_normalization(img_mean, img_std, stat_target):

    bounds = np.ones((cfg.out_channels, 2)) * np.inf
    if stat_target is None:
        if cfg.n_target_data == 0:
            mean_val, std_val = img_mean[:cfg.out_channels], img_std[:cfg.out_channels]
        else:
            mean_val = img_mean[-cfg.n_target_data:]
            std_val = img_std[-cfg.n_target_data:]
    else:
        mean_val, std_val = stat_target["mean"], stat_target["std"]
    k = 0
    for bound in (cfg.min_bounds, cfg.max_bounds):
        bounds[:, k] = bound

        if cfg.normalize_data:
            bounds[:, k] = (bounds[:, k] - mean_val) / std_val

        k += 1

    return bounds
