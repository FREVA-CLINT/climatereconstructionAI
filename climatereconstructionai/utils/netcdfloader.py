import random
import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
import os
import sys
import logging
from .netcdfchecker import check_ncformat

from .. import config as cfg

def SteadyMaskLoader(path, mask_name, data_type, device):

    if mask_name is None:
        return None
    else:
        steady_mask, _ = load_netcdf(path, [mask_name], [data_type])
        return torch.from_numpy(steady_mask[0][data_type].values).to(device)


def img_normalization(img_data,data_types):

    img_mean, img_std, img_tf = [], [], []
    for i in range(len(data_types)):
        img_mean.append(img_data[i][data_types[i]].mean().item())
        img_std.append(img_data[i][data_types[i]].std().item())
        img_tf.append(transforms.Normalize(mean=[img_mean[-1]], std=[img_std[-1]]))

    return img_mean, img_std, img_tf

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
        np.random.seed(cfg.random_seed)
        order = np.random.permutation(self.num_samples)
        while True:
            yield order[i]
            i += 1
            if i >= self.num_samples:
                np.random.seed(cfg.random_seed)
                order = np.random.permutation(self.num_samples)
                i = 0

def nc_loadchecker(filename,data_type,image_size):

    basename = filename.split("/")[-1]

    if not os.path.isfile(filename):
        print('File {} not found.'.format(filename))

    try:
        # We use load_dataset instead of open_dataset because of lazy transpose
        ds = xr.load_dataset(filename,decode_times=False)
    except:
        logging.error('Impossible to read the input file {}.\nPlease, check that the input file is a netCDF file and is not corrupted.'.format(basename))
        sys.exit()

    ds = check_ncformat(ds,data_type,image_size,basename)

    return ds

def load_netcdf(path,data_names,data_types):

    if data_names is None:
        return None, None
    else:
        ndata = len(data_names)
        assert ndata == len(data_types)

        data, lengths = [], []
        for i in range(ndata):
            data.append(nc_loadchecker('{}{}'.format(path,data_names[i]),data_types[i],cfg.image_sizes[0]))
            lengths.append(len(data[-1][data_types[i]]))

        if cfg.img_index is None:
            assert len(set(lengths)) == 1
        return data, lengths[0]


class NetCDFLoader(Dataset):
    def __init__(self, data_root, img_names, mask_root, mask_names, split, data_types, lstm_steps, prev_next_steps):
        super(NetCDFLoader, self).__init__()
        assert lstm_steps == 0 or prev_next_steps == 0

        self.data_types = data_types
        self.lstm_steps = lstm_steps
        self.prev_next_steps = prev_next_steps

        if split == 'train':
            data_path = '{:s}/data_large/'.format(data_root)
        elif split == 'infill':
            data_path = '{:s}/test_large/'.format(data_root)
        elif split == 'val':
            data_path = '{:s}/val_large/'.format(data_root)

        self.img_data, self.img_length = load_netcdf(data_path,img_names,data_types)
        self.mask_data, self.mask_length = load_netcdf(mask_root,mask_names,data_types)

        if self.mask_data is None:
            self.mask_length = self.img_length
        else:
            if not cfg.shuffle_masks:
                assert self.img_length == self.mask_length

        self.img_mean, self.img_std, self.img_tf = img_normalization(self.img_data,data_types)


    def load_data(self, ind_data, img_indices, mask_indices):

        data_type = self.data_types[ind_data]
        if self.mask_data is None:
            # Get masks from images
            image = self.img_data[ind_data][data_type].values[mask_indices]
            mask = torch.from_numpy((1-(np.isnan(image))).astype(image.dtype))
        else:
            mask = torch.from_numpy(self.mask_data[ind_data][data_type].values[mask_indices])
        image = self.img_data[ind_data][data_type].values[img_indices]
        image = torch.from_numpy(np.nan_to_num(image))

        if cfg.normalize_images:
            image = self.img_tf[ind_data](image)
        # print(img_indices,image.mean(),image.std())

        # # open netcdf file
        # try:
        #     total_data = torch.from_numpy(h5_data[indices, :, :])
        # except TypeError:
        #     # get indices that occur more than once
        #     unique, counts = np.unique(indices, return_counts=True)
        #     copy_indices = [(index, counts[index] - 1) for index in range(len(counts)) if counts[index] > 1]
        #     if h5_data.ndim == 4:
        #         total_data = torch.from_numpy(h5_data[unique, 0, :, :])
        #     else:
        #         total_data = torch.from_numpy(h5_data[unique, :, :])
        #     if unique[copy_indices[0][0]] == 0:
        #         total_data = torch.cat([torch.stack(copy_indices[0][1] * [total_data[copy_indices[0][0]]]), total_data])
        #     else:
        #         total_data = torch.cat([total_data, torch.stack(copy_indices[0][1] * [total_data[copy_indices[0][0]]])])
        return image, mask

    def get_single_item(self, ind_data, index, shuffle_masks):
        if self.lstm_steps == 0:
            prev_steps = next_steps = self.prev_next_steps
        else:
            prev_steps = next_steps = self.lstm_steps

        # define range of lstm or prev-next steps -> adjust, if out of boundaries
        img_indices = np.array(list(range(index - prev_steps, index + next_steps + 1)))
        img_indices[img_indices < 0] = 0
        img_indices[img_indices > self.img_length - 1] = self.img_length - 1
        if shuffle_masks:
            mask_indices = []
            for j in range(prev_steps + next_steps + 1):
                mask_indices.append(random.randint(0, self.mask_length - 1))
            mask_indices = sorted(mask_indices)
        else:
            mask_indices = img_indices

        # load data from ranges
        images, masks = self.load_data(ind_data, img_indices, mask_indices)

        # stack to correct dimensions
        if self.lstm_steps == 0:
            images = torch.cat([images], dim=0).unsqueeze(0)
            masks = torch.cat([masks], dim=0).unsqueeze(0)
        else:
            images = torch.stack([images], dim=1)
            masks = torch.stack([masks], dim=1)

        return images, masks

    def __getitem__(self, index):

        images = []
        masks = []
        masked = []
        for i in range(len(self.data_types)):

            if i == cfg.img_index:
                image, mask = self.get_single_item(i,index,False)
                masks[0] = masks[0]*mask
                masked[0] = image*masks[0]
            else:
                image, mask = self.get_single_item(i,index,cfg.shuffle_masks)
                images.append(image)
                masks.append(mask)
                masked.append(image*mask)

        if len(images) == 1:
            return masked[0], masks[0], images[0], torch.tensor([]), torch.tensor([]), torch.tensor([])
        else:
            return masked[0], masks[0], images[0], torch.cat(masked[1:], dim=1), torch.cat(masks[1:], dim=1), torch.cat(images[1:], dim=1)


    def __len__(self):
        return self.img_length
