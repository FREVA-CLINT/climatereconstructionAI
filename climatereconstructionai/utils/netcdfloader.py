import random
import numpy as np
import torch
import xarray as xr
import xesmf as xe
from torch.utils.data import Dataset, Sampler
import os
import sys
import logging

from .. import config as cfg


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

def nc_checker(filename,data_type,image_size):

    basename = filename.split("/")[-1]

    if not os.path.isfile(filename):
        print('File {} not found.'.format(filename))

    try:
        # We use load_dataset instead of open_dataset because of lazy transpose
        ds = xr.load_dataset(filename,decode_times=False)
    except:
        logging.error('Impossible to read the input file {}.\nPlease, check that the input file is a netCDF file and is not corrupted.'.format(basename))
        sys.exit()

    if not data_type in list(ds.keys()):
        logging.error('Variable name \'{}\' not found in file {}.'.format(data_type,basename))
        sys.exit()

    if not cfg.dataset_name is None:

        ds_dims = list(ds[data_type].dims)
        ndims = len(cfg.dataset_format["dimensions"])

        if ndims != len(ds_dims):
            logging.error('Inconsistent number of dimensions in file {}.\nThe number of dimensions should be: {}'.format(basename,ndims))
            sys.exit()

        for i in range(ndims):
            if cfg.dataset_format["dimensions"][i] != ds_dims[i]:
                logging.error('Inconsistent dimensions in file {}.\nThe list of dimensions should be: {}'.format(basename,cfg.dataset_format["dimensions"]))
                sys.exit()

        ds[data_type] = ds[data_type].transpose(*cfg.dataset_format["axes"])

        shape = ds[data_type].shape

        step = []
        regrid = False
        for i in range(2):
            coordinate = cfg.dataset_format["axes"][i+1]

            step.append(np.unique(np.gradient(ds[data_type][coordinate].values)))
            if len(step[i]) != 1:
                logging.error('The {} grid in file {} is not uniform.'.format(coordinate,basename))
                sys.exit()

            extent = cfg.dataset_format["grid"][i][1]-cfg.dataset_format["grid"][i][0]
            if abs( ds[data_type][coordinate].values[-1] - ds[data_type][coordinate].values[0] + step[i] - extent ) > 1e-2:
                logging.error('Incorrect {} extent in file {}.\nThe extent should be: {}'.format(coordinate,basename,extent))
                sys.exit()

            if shape[i+1] != image_size:
                step[i] *= shape[i+1]/image_size
                logging.warning('The length of {} does not correspond to the image size for file {}.'.format(coordinate,basename))
                regrid = True

        if regrid:
            logging.warning('The spatial coordinates have been interpolated using nearest_s2d in file {}.'.format(basename))
            grid = xr.Dataset({cfg.dataset_format["axes"][1]: ([cfg.dataset_format["axes"][1]], xe.util._grid_1d(*cfg.dataset_format["grid"][0][:2],step[0])[0]),
                              cfg.dataset_format["axes"][2]: ([cfg.dataset_format["axes"][2]], xe.util._grid_1d(*cfg.dataset_format["grid"][1][:2],step[1])[0])})
            ds = xe.Regridder(ds, grid, "nearest_s2d")(ds,keep_attrs=True)

        if ds[data_type].dtype != "float32":
            logging.warning('Incorrect data type for file {}.\nData type has been converted to float32.'.format(basename))
            ds[data_type] = ds[data_type].astype(dtype=np.float32)
    return ds

def get_data(path,data_names,data_types):

    if data_names is None:
        return None, None
    else:
        ndata = len(data_names)
        assert ndata == len(data_types)

        data, shape = [], []
        for i in range(ndata):
            data.append(nc_checker('{}{}'.format(path,data_names[i]),data_types[i],cfg.image_sizes[0]))
            shape.append(data[-1][data_types[i]].shape)

        if cfg.img_index is None:
            assert len(set(shape)) == 1
        return data, shape[0][0]

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

        self.img_data, self.img_length = get_data(data_path,img_names,data_types)
        self.mask_data, self.mask_length = get_data(mask_root,mask_names,data_types)

        if self.mask_data is None:
            self.mask_length = self.img_length
        else:
            if not cfg.shuffle_masks:
                assert self.img_length == self.mask_length


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
