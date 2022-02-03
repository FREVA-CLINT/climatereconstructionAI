import random
import numpy as np
import torch
import h5py
import xarray as xr
from torch.utils.data import Dataset, Sampler
import sys

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
        order = np.random.permutation(self.num_samples)
        while True:
            yield order[i]
            i += 1
            if i >= self.num_samples:
                np.random.seed()
                order = np.random.permutation(self.num_samples)
                i = 0

class NetCDFLoader(Dataset):
    def __init__(self, data_root, img_names, mask_root, mask_names, split, data_types, lstm_steps, prev_next_steps):
        super(NetCDFLoader, self).__init__()
        assert lstm_steps == 0 or prev_next_steps == 0

        self.split = split
        self.data_types = data_types
        self.img_names = img_names

        self.lstm_steps = lstm_steps
        self.prev_next_steps = prev_next_steps

        if split == 'train':
            self.data_path = '{:s}/data_large/'.format(data_root)
        elif split == 'test' or split == 'infill':
            self.data_path = '{:s}/test_large/'.format(data_root)
        elif split == 'val':
            self.data_path = '{:s}/val_large/'.format(data_root)
        self.mask_path = mask_root

        if mask_names is None:
            # Convert img to masks
            self.mask_names = []
            for i in range(len(img_names)):
                self.convert_to_mask(img_names[i], data_types[i])

        else:
            self.mask_names = mask_names

        # define image and mask lenghts
        self.img_lengths = {}
        self.mask_lengths = {}
        assert len(img_names) == len(self.mask_names) == len(data_types)
        for i in range(len(img_names)):
            self.init_dataset(img_names[i], self.mask_names[i], data_types[i])

    def convert_to_mask(self, img_name, data_type):
        ds_src = xr.open_dataset('{}{}'.format(self.data_path, img_name))
        data = 1-(np.isnan(ds_src[data_type].values))
        ds_dest = ds_src.copy(data={data_type: data})
        self.mask_names.append("msk_"+img_name)
        ds_dest.to_netcdf('{}{}'.format(self.mask_path, self.mask_names[-1]))

    def init_dataset(self, img_name, mask_name, data_type):
        # set img and mask length
        img_file = h5py.File('{}{}'.format(self.data_path, img_name), 'r')
        img_data = img_file.get(data_type)
        mask_file = h5py.File('{}{}'.format(self.mask_path, mask_name), 'r')
        mask_data = mask_file.get(data_type)
        self.img_lengths[img_name] = img_data.shape[0]
        self.mask_lengths[mask_name] = mask_data.shape[0]

        # if infill, check if img length matches mask length
        if self.split == 'infill':
            assert img_data.shape[0] == mask_data.shape[0]

    def load_data(self, file, data_type, indices):
        # open netcdf file
        h5_data = file.get(data_type)
        try:
            if h5_data.ndim == 4:
                total_data = torch.from_numpy(h5_data[indices, 0, :, :])
            else:
                total_data = torch.from_numpy(h5_data[indices, :, :])
        except TypeError:
            # get indices that occur more than once
            unique, counts = np.unique(indices, return_counts=True)
            copy_indices = [(index, counts[index] - 1) for index in range(len(counts)) if counts[index] > 1]
            if h5_data.ndim == 4:
                total_data = torch.from_numpy(h5_data[unique, 0, :, :])
            else:
                total_data = torch.from_numpy(h5_data[unique, :, :])
            if unique[copy_indices[0][0]] == 0:
                total_data = torch.cat([torch.stack(copy_indices[0][1] * [total_data[copy_indices[0][0]]]), total_data])
            else:
                total_data = torch.cat([total_data, torch.stack(copy_indices[0][1] * [total_data[copy_indices[0][0]]])])
        return total_data

    def get_single_item(self, index, img_name, mask_name, data_type):
        if self.lstm_steps == 0:
            prev_steps = next_steps = self.prev_next_steps
        else:
            prev_steps = next_steps = self.lstm_steps

        # define range of lstm or prev-next steps -> adjust, if out of boundaries
        img_indices = np.array(list(range(index - prev_steps, index + next_steps + 1)))
        img_indices[img_indices < 0] = 0
        img_indices[img_indices > self.img_lengths[img_name] - 1] = self.img_lengths[img_name] - 1
        if self.split == 'infill':
            mask_indices = img_indices
        else:
            mask_indices = []
            for j in range(prev_steps + next_steps + 1):
                mask_indices.append(random.randint(0, self.mask_lengths[mask_name] - 1))
            mask_indices = sorted(mask_indices)

        # load data from ranges
        img_file = h5py.File('{}{}'.format(self.data_path, img_name), 'r')
        mask_file = h5py.File('{}{}'.format(self.mask_path, mask_name), 'r')
        images = self.load_data(img_file, data_type, img_indices)
        masks = self.load_data(mask_file, data_type, mask_indices)

        # stack to correct dimensions
        if self.lstm_steps == 0:
            images = torch.cat([images], dim=0).unsqueeze(0)
            masks = torch.cat([masks], dim=0).unsqueeze(0)
        else:
            images = torch.stack([images], dim=1)
            masks = torch.stack([masks], dim=1)
        return images, masks

    def __getitem__(self, index):
        image, mask = self.get_single_item(index, self.img_names[0], self.mask_names[0], self.data_types[0])
        images = []
        masks = []
        for i in range(1, len(self.data_types)):
            img, m = self.get_single_item(index, self.img_names[i], self.mask_names[i], self.data_types[i])
            images.append(img)
            masks.append(m)
        if images and masks:
            images = torch.cat(images, dim=1)
            masks = torch.cat(masks, dim=1)
            return mask*image, mask, image, masks*images, masks, images
        else:
            return mask*image, mask, image, torch.tensor([]), torch.tensor([]), torch.tensor([])

    def __len__(self):
        return self.img_lengths[self.img_names[0]]
