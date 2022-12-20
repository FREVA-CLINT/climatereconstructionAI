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


def renormalize(img_data, img_mean, img_std, cname):
    if cfg.normalize_data and cname != "mask":
        return img_std * img_data + img_mean
    else:
        return img_data

def bnd_normalization(img_mean, img_std):

    bounds = np.ones((cfg.out_channels, 2)) * np.inf
    if cfg.target_data_indices == []:
        idx = list(range(cfg.out_channels))
    else:
        idx = cfg.target_data_indices
    k = 0
    for bound in (cfg.min_bounds, cfg.max_bounds):
        bounds[:, k] = bound
        
        if cfg.normalize_data:
            for i in range(cfg.out_channels):
                bounds[i, k] = (bounds[i, k] - img_mean[idx[i]]) / img_std[idx[i]]

        k += 1

    return bounds

