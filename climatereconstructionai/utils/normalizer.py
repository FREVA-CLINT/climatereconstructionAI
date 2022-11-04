import numpy as np
from torchvision import transforms


def img_normalization(img_data):
    img_std, img_mean, img_tf = [], [], []
    for i in range(len(img_data)):
        img_mean.append(np.nanmean(img_data[i]))
        img_std.append(np.nanstd(img_data[i]))
        img_tf.append(transforms.Normalize(mean=[img_mean[-1]], std=[img_std[-1]]))

    return img_mean, img_std, img_tf


def renormalize(img_data, img_mean, img_std):
    return img_std * img_data + img_mean

def bnd_normalization(min_bounds, max_bounds, out_channels, normalize_data, img_mean, img_std):

    bounds = np.ones((out_channels, 2)) * np.inf
    k = 0
    for bound in (min_bounds, max_bounds):
        bounds[:, k] = bound
        
        if normalize_data:
            bounds[:, k] = (bounds[:, k] - img_mean) / img_std

        k += 1

    return bounds

