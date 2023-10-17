import os
import random

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset, Sampler
from torch.functional import F

from .netcdfchecker import dataset_formatter
from .normalizer import img_normalization, bnd_normalization
from .. import config as cfg


def load_steadymask(steady_mask_dir_dict, device):
    if steady_mask_dir_dict is None:
        return None
    else:
        mask_data = []
        for (mask_name, mask_type), mask_dirs in steady_mask_dir_dict.items():
            data, _, _ = load_netcdf(mask_dirs, len(mask_dirs) * [mask_type])
            mask_data.append(np.concatenate(data))
        return torch.stack([torch.from_numpy(np.array(mask)).to(device) for mask in mask_data]).squeeze()


class InfiniteSampler(Sampler):
    def __init__(self, num_samples, data_source=None):
        super().__init__(data_source)
        self.num_samples = num_samples

    def __iter__(self):
        return iter(self.loop())

    def __len__(self):
        return 2 ** 31

    def loop(self):
        i = 0
        n_samples = self.num_samples - sum(cfg.time_steps)
        order = np.random.permutation(n_samples) + cfg.time_steps[0]
        while True:
            yield order[i]
            i += 1
            if i >= n_samples:
                order = np.random.permutation(n_samples) + cfg.time_steps[0]
                i = 0


class FiniteSampler(Sampler):
    def __init__(self, num_samples, data_source=None):
        super().__init__(data_source)
        self.num_samples = num_samples

    def __iter__(self):
        return iter(range(cfg.time_steps[0], self.num_samples - cfg.time_steps[1]))

    def __len__(self):
        return self.num_samples


def nc_loadchecker(filename, data_type):
    basename = filename.split("/")[-1]

    if not os.path.isfile(filename):
        print('File {} not found.'.format(filename))

    try:
        ds = xr.open_dataset(filename)
    except Exception:
        try:
            ds = xr.open_dataset(filename, decode_times=False)
        except Exception:
            raise ValueError('Impossible to read {}.'
                             '\nPlease, check that it is a netCDF file and it is not corrupted.'.format(basename))

    ds1 = dataset_formatter(ds, data_type, basename)
    ds = ds.drop_vars(data_type)

    if cfg.lazy_load:
        data = ds1[data_type]
    else:
        data = ds1[data_type].values

    dims = ds1[data_type].dims
    coords = {key: ds1[data_type].coords[key] for key in ds1[data_type].coords if key != "time"}
    ds1 = ds1.drop_vars(ds1.keys())
    ds1 = ds1.drop_dims("time")

    return [ds, ds1, dims, coords], data, data.shape[0], data.shape[1:]


def load_netcdf(data_paths, data_types, keep_dss=False):
    if data_paths is None:
        return None, None
    else:
        ndata = len(data_paths)
        assert ndata == len(data_types)
        dss, data, lengths, sizes = zip(*[nc_loadchecker(data_paths[i], data_types[i]) for i in range(ndata)])

        if keep_dss:
            return dss[0], data, lengths[0], (sizes[0],)
        else:
            return data, lengths[0], (sizes[0],)


class NetCDFLoader(Dataset):
    def __init__(self, data_dir_dict, mask_dir_dict, time_steps, train_stats=None, remap_data=None):
        super(NetCDFLoader, self).__init__()

        self.random = random.Random(cfg.loop_random_seed)

        self.time_steps = time_steps

        self.n_time_steps = sum(time_steps) + 1

        self.img_data, self.data_types, self.mask_data = [], [], []
        self.xr_dss = None

        self.remap_data = remap_data

        for (data_name, data_type), data_dirs in data_dir_dict.items():
            if self.xr_dss is None:
                self.xr_dss, dir_data, self.img_length, self.img_sizes = load_netcdf(data_dirs,
                                                                                     len(data_dirs)*[data_type],
                                                                                     keep_dss=True)
            else:
                dir_data, _, _ = load_netcdf(data_dirs, len(data_dirs)*[data_type])
            self.img_data.append(np.concatenate(dir_data))
            self.data_types.append(data_type)

        if mask_dir_dict:
            for (mask_name, mask_type), mask_dirs in mask_dir_dict.items():
                mask_data, self.mask_length, _ = load_netcdf(mask_dirs, len(mask_dirs)*[mask_type])
                self.mask_data.append(np.concatenate(mask_data))

        if not self.mask_data:
            self.mask_length = self.img_length
        else:
            if not cfg.shuffle_masks:
                assert self.img_length == self.mask_length

        self.img_mean, self.img_std, self.img_znorm = img_normalization(self.img_data)

        self.bounds = bnd_normalization(self.img_mean, self.img_std, train_stats)

        if self.remap_data:
            _, x, y = self.remap_data.split("_")
            self.img_sizes = ((int(x), int(y)),)

    def load_data(self, ind_data, img_indices, mask_indices):

        if self.mask_data is None:
            # Get masks from images
            image = np.array(self.img_data[ind_data][mask_indices])
            mask = torch.from_numpy((1 - (np.isnan(image))).astype(image.dtype))
        else:
            mask = torch.from_numpy(np.array(self.mask_data[ind_data][mask_indices]))
        image = np.array(self.img_data[ind_data][img_indices])
        image = torch.from_numpy(np.nan_to_num(image))

        if cfg.normalize_data:
            image = self.img_znorm[ind_data](image)

        return image, mask

    def get_single_item(self, ind_data, index, shuffle_masks):
        # define range of lstm or prev-next steps -> adjust, if out of boundaries
        img_indices = np.array(list(range(index - self.time_steps[0], index + self.time_steps[1] + 1)))
        img_indices[img_indices < 0] = 0
        img_indices[img_indices > self.img_length - 1] = self.img_length - 1
        if shuffle_masks:
            mask_indices = []
            for j in range(self.n_time_steps):
                mask_indices.append(self.random.randint(0, self.mask_length - 1))
            mask_indices = sorted(mask_indices)
        else:
            mask_indices = img_indices
        # load data from ranges
        images, masks = self.load_data(ind_data, img_indices, mask_indices)

        # stack to correct dimensions
        images = torch.stack([images], dim=1)
        masks = torch.stack([masks], dim=1)

        # interpolate
        if self.remap_data:
            mode, x, y = self.remap_data.split("_")
            x, y = int(x), int(y)
            if masks.shape[-1] != y or masks.shape[-2] != x:
                masks = F.interpolate(masks, (x, y), mode=mode)
            if images.shape[-1] != y or images.shape[-2] != x:
                images = F.interpolate(images, (x, y), mode=mode)

        return images, masks

    def __getitem__(self, index):

        images = []
        masks = []
        masked = []
        ndata = len(self.data_types)

        for i in range(ndata):

            image, mask = self.get_single_item(i, index, cfg.shuffle_masks)

            if i >= ndata - cfg.n_target_data:
                images.append(image)
            else:
                if cfg.n_target_data == 0:
                    images.append(image)
                masks.append(mask)
                masked.append(image * mask)

        if cfg.channel_steps:
            masked, masks, images = torch.cat(masked, dim=0).transpose(0, 1), torch.cat(masks, dim=0) \
                .transpose(0, 1), torch.cat(images, dim=0).transpose(0, 1)
        else:
            masked, masks, images = torch.cat(masked, dim=1), torch.cat(masks, dim=1), torch.cat(images, dim=1)

        if cfg.predict_diff:
            images -= masked

        return masked, masks, images, index

    def __len__(self):
        return self.img_length
