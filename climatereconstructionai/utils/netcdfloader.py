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
        if cfg.target_data_indices == []:
            steady_mask, _ = load_netcdf(path, mask_names, data_types[:cfg.out_channels])
        else:
            steady_mask, _ = load_netcdf(path, mask_names, [data_types[i] for i in cfg.target_data_indices])
        # stack + squeeze ensures that it works with steady masks with one timestep or no timestep
        return torch.stack([torch.from_numpy(mask).to(device) for mask in steady_mask]).squeeze()


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
        order = np.random.permutation(self.num_samples)
        while True:
            yield order[i]
            i += 1
            if i >= self.num_samples:
                order = np.random.permutation(self.num_samples)
                i = 0


def nc_loadchecker(filename, data_type, image_size, keep_dss=False):
    basename = filename.split("/")[-1]

    if not os.path.isfile(filename):
        print('File {} not found.'.format(filename))

    try:
        # We use load_dataset instead of open_dataset because of lazy transpose
        ds = xr.load_dataset(filename, decode_times=False)
    except Exception:
        raise ValueError('Impossible to read {}.'
                         '\nPlease, check that it is a netCDF file and it is not corrupted.'.format(basename))

    ds1 = dataset_formatter(ds, data_type, image_size, basename)

    if keep_dss:
        dtype = ds[data_type].dtype
        ds = ds.drop_vars(data_type)
        ds[data_type] = np.empty(0, dtype=dtype)
        dss = [ds, ds1]
    else:
        dss = None

    return dss, ds1[data_type].values, ds1[data_type].shape[0]


def load_netcdf(path, data_names, data_types, keep_dss=False):
    if data_names is None:
        return None, None
    else:
        ndata = len(data_names)
        assert ndata == len(data_types)

        dss, data, lengths = zip(*[nc_loadchecker('{}{}'.format(path, data_names[i]), data_types[i], cfg.image_sizes[i],
                                   keep_dss=keep_dss) for i in range(ndata)])

        # if cfg.input_data_index is None:
        assert len(set(lengths)) == 1

        if keep_dss:
            return dss, data, lengths[0]
        else:
            return data, lengths[0]


class NetCDFLoader(Dataset):
    def __init__(self, data_root, img_names, mask_root, mask_names, split, data_types, time_steps, stat_target=None):
        super(NetCDFLoader, self).__init__()

        self.random = random.Random(cfg.loop_random_seed)

        self.data_types = data_types
        self.time_steps = time_steps

        mask_path = mask_root
        if split == 'infill':
            data_path = '{:s}/test/'.format(data_root)
            self.xr_dss, self.img_data, self.img_length = load_netcdf(data_path, img_names, data_types, keep_dss=True)
        else:
            if split == 'train':
                data_path = '{:s}/train/'.format(data_root)
            else:
                data_path = '{:s}/val/'.format(data_root)
                if not cfg.shuffle_masks:
                    mask_path = '{:s}/val/'.format(mask_root)
            self.img_data, self.img_length = load_netcdf(data_path, img_names, data_types)

        self.mask_data, self.mask_length = load_netcdf(mask_path, mask_names, data_types)

        if self.mask_data is None:
            self.mask_length = self.img_length
        else:
            if not cfg.shuffle_masks:
                assert self.img_length == self.mask_length

        self.img_mean, self.img_std, self.img_tf = img_normalization(self.img_data)

        self.bounds = bnd_normalization(self.img_mean, self.img_std, stat_target)

    def load_data(self, ind_data, img_indices, mask_indices):

        if self.mask_data is None:
            # Get masks from images
            image = self.img_data[ind_data][mask_indices]
            mask = torch.from_numpy((1 - (np.isnan(image))).astype(image.dtype))
        else:
            mask = torch.from_numpy(self.mask_data[ind_data][mask_indices])
        image = self.img_data[ind_data][img_indices]
        image = torch.from_numpy(np.nan_to_num(image))

        if cfg.normalize_data:
            image = self.img_tf[ind_data](image)

        return image, mask

    def get_single_item(self, ind_data, index, shuffle_masks):
        # define range of lstm or prev-next steps -> adjust, if out of boundaries
        img_indices = np.array(list(range(index - self.time_steps, index + self.time_steps + 1)))
        img_indices[img_indices < 0] = 0
        img_indices[img_indices > self.img_length - 1] = self.img_length - 1
        if shuffle_masks:
            mask_indices = []
            for j in range(2 * self.time_steps + 1):
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

            if i in cfg.target_data_indices:
                images.append(image)
            else:
                if cfg.target_data_indices == []:
                    images.append(image)
                masks.append(mask)
                masked.append(image * mask)

        if len(images) == 1:
            if cfg.channel_steps:
                return masked[0].transpose(0, 1), masks[0].transpose(0, 1), images[0].transpose(0, 1)
            else:
                return masked[0], masks[0], images[0]
        else:
            if cfg.channel_steps:
                return torch.cat(masked, dim=0).transpose(0, 1), torch.cat(masks, dim=0)\
                    .transpose(0, 1), torch.cat(images, dim=0).transpose(0, 1)
            else:
                return torch.cat(masked, dim=1), torch.cat(masks, dim=1), torch.cat(images, dim=1)

    def __len__(self):
        return self.img_length
