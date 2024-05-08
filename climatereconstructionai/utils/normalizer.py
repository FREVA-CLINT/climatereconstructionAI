import numpy as np
import torch
from torchvision import transforms
from .. import config as cfg


def img_normalization(img_data):
    img_std, img_mean, img_znorm = [], [], []
    if cfg.normalize_data:
        for i in range(len(img_data)):
            img_mean.append(np.nanmean(np.array(img_data[i])))
            img_std.append(np.nanstd(np.array(img_data[i])))
            img_znorm.append(transforms.Normalize(mean=[img_mean[-1]], std=[img_std[-1]]))

    return img_mean, img_std, img_znorm


def renormalize(img_data, img_mean, img_std):
    return img_std * img_data + img_mean


def bnd_normalization(img_mean, img_std, train_stats):

    bounds = np.ones((cfg.out_channels, 2)) * np.inf
    if train_stats is None:
        if cfg.n_target_data == 0:
            mean_val, std_val = img_mean[:cfg.out_channels], img_std[:cfg.out_channels]
        else:
            mean_val, std_val = img_mean[-cfg.n_target_data:], img_std[-cfg.n_target_data:]
    else:
        if cfg.n_target_data == 0:
            mean_val, std_val = train_stats["mean"][:cfg.out_channels], train_stats["std"][:cfg.out_channels]
        else:
            mean_val, std_val = train_stats["mean"][-cfg.n_target_data:], train_stats["std"][-cfg.n_target_data:]
    k = 0
    for bound in (cfg.min_bounds, cfg.max_bounds):
        bounds[:, k] = bound

        if cfg.normalize_data:
            bounds[:, k] = (bounds[:, k] - mean_val) / std_val

        k += 1

    return bounds

class normalizer(torch.nn.Module):
    def __init__(self, norm_dict):
        super().__init__()

        self.norm_dict = norm_dict

        self.norm_fcn_dict = {
            'quantile':norm_min_max,
            'quantile_abs':norm_max,
            'min_max':norm_max,
            'normal':norm_mean_std,
            "None":identity
        }
       
        if 'uv' in norm_dict.keys():
            self.uniform_norm_uv = True
        else:
            self.uniform_norm_uv = False

    def __call__(self, data, denorm=False):
        for var, data_var in data.items():
            if self.uniform_norm_uv and (var=='u' or var=='v'):
                var_lookup = 'uv'
            else:
                var_lookup= var
            moments = self.norm_dict[var_lookup]['moments']
            norm_fcn = self.norm_fcn_dict[self.norm_dict[var_lookup]['type']]
            data[var] = norm_fcn(data_var, moments, denorm)
        return data

class grid_normalizer(torch.nn.Module):
    def __init__(self, norm_dict):
        super().__init__()

        self.norm_dict = norm_dict

        self.norm_fcn_dict = {
            'quantile':norm_min_max,
            'quantile_abs':norm_max,
            'min_max':norm_max,
            'normal':norm_mean_std,
            "None":identity
        }
       
        if 'uv' in norm_dict.keys():
            self.uniform_norm_uv = True
        else:
            self.uniform_norm_uv = False

    def __call__(self, data, grid_vars, denorm=False):
        for grid_type, vars in grid_vars.items():
            
            for k, var in enumerate(vars):
                data_var = data[grid_type]
                data_var = data_var[:,:,k]
                
                if self.uniform_norm_uv and (var=='u' or var=='v'):
                    var_lookup = 'uv'
                else:
                    var_lookup= var
                moments = self.norm_dict[var_lookup]['moments']
                norm_fcn = self.norm_fcn_dict[self.norm_dict[var_lookup]['type']]
                data[grid_type][:,:,k] = norm_fcn(data_var, moments, denorm)
        return data

def identity(data, moments, denorm=False):
    return data 

def norm_max(data, moments, denorm=False):
    if denorm:
        data = data * moments[1]
    else:
        data = (data) / (moments[1])
    return data 

def norm_min_max(data, moments, denorm=False):
    if denorm:
        data = data*(moments[1] - moments[0]) + moments[0]
    else:
        data = (data - moments[0])/(moments[1] - moments[0])
    return data

def norm_mean_std(data, moments, denorm=False):
    if denorm:
        data = data * moments[1] + moments[0]
    else:
        data = (data - moments[0]) / moments[1]
    return data