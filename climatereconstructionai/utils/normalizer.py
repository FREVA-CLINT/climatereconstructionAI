
import numpy as np
from torchvision import transforms
from .. import config as cfg

def img_normalization(img_data):

    img_std, img_mean, img_tf = [], [], []
    for i in range(len(img_data)):
        mean = np.nanmean(img_data[i])
        std = np.nanstd(img_data[i])
        img_std.append(std / cfg.STD)
        img_mean.append(mean - cfg.MEAN * img_std[-1])

        img_tf.append(transforms.Normalize(mean=[img_mean[-1]], std=[img_std[-1]]))

    return img_mean, img_std, img_tf

def renormalize(img_data, img_mean, img_std):
    return img_std*img_data+img_mean