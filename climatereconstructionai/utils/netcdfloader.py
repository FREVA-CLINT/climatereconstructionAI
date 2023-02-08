import os
import random

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset, Sampler

from .netcdfchecker import dataset_formatter
from .normalizer import img_normalization, bnd_normalization
from .. import config as cfg


def load_steadymask(path, mask_names, data_types, device):
    if mask_names is None:
        return None
    else:
        assert len(mask_names) == cfg.out_channels
        if cfg.n_target_data == 0:
            steady_mask = load_netcdf(path, mask_names, data_types[:cfg.out_channels])[0]
        else:
            steady_mask = load_netcdf(path, mask_names, data_types[-cfg.n_target_data:])[0]
        # stack + squeeze ensures that it works with steady masks with one timestep or no timestep
        return torch.stack([torch.from_numpy(np.array(mask)).to(device) for mask in steady_mask]).squeeze()


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


def load_netcdf(path, data_names, data_types, keep_dss=False):
    if data_names is None:
        return None, None, None
    else:
        ndata = len(data_names)
        assert ndata == len(data_types)

        dss, data, lengths, sizes = zip(*[nc_loadchecker('{}{}'.format(path, data_names[i]),
                                                         data_types[i]) for i in range(ndata)])

        assert len(set(lengths)) == 1

        if keep_dss:
            return dss, data, lengths[0], sizes
        else:
            return data, lengths[0], sizes


class NetCDFLoader(Dataset):
    def __init__(self, data_root, img_names, mask_root, mask_names, split, data_types, time_steps, train_stats=None):
        super(NetCDFLoader, self).__init__()

        self.random = random.Random(cfg.loop_random_seed)

        self.data_types = data_types
        self.time_steps = time_steps
        self.n_time_steps = sum(time_steps) + 1

        mask_path = mask_root
        if split == 'infill':
            data_path = '{:s}/test/'.format(data_root)
            self.xr_dss, self.img_data, self.img_length, self.img_sizes = load_netcdf(data_path, img_names, data_types,
                                                                                      keep_dss=True)
        else:
            if split == 'train':
                data_path = '{:s}/train/'.format(data_root)
            else:
                data_path = '{:s}/val/'.format(data_root)
                if not cfg.shuffle_masks:
                    mask_path = '{:s}/val/'.format(mask_root)
            self.img_data, self.img_length, self.img_sizes = load_netcdf(data_path, img_names, data_types)

        self.mask_data, self.mask_length, _ = load_netcdf(mask_path, mask_names, data_types)

        if self.mask_data is None:
            self.mask_length = self.img_length
        else:
            if not cfg.shuffle_masks:
                assert self.img_length == self.mask_length

        self.img_mean, self.img_std, self.img_tf = img_normalization(self.img_data)

        self.bounds = bnd_normalization(self.img_mean, self.img_std, train_stats)

    def load_data(self, ind_data, img_indices, mask_indices):

        if self.mask_data is None:
            # Get masks from images
            image = np.array(self.img_data[ind_data][mask_indices])
            mask = torch.from_numpy((1 - (np.isnan(image))).astype(image.dtype))
        else:
            mask = torch.from_numpy(np.array(self.mask_data[ind_data][mask_indices]))
        image = np.array(self.img_data[ind_data][img_indices])
        image = torch.from_numpy(image)

        if cfg.normalize_data:
            image = self.img_tf[ind_data](image)

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
                masked.append(torch.nan_to_num(image) * mask)

        if cfg.channel_steps:
            return torch.cat(masked, dim=0).transpose(0, 1), torch.cat(masks, dim=0)\
                .transpose(0, 1), torch.cat(images, dim=0).transpose(0, 1), index
        else:
            return torch.cat(masked, dim=1), torch.cat(masks, dim=1), torch.cat(images, dim=1), index

    def __len__(self):
        return self.img_length
