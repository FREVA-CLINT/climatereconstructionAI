import random

import numpy as np
import torch
import h5py
from glob import glob
import torch.utils.data as data


class InfiniteSampler(data.sampler.Sampler):
    def __init__(self, num_samples):
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


class NetCDFLoader(torch.utils.data.Dataset):
    def __init__(self, data_root, img_names, mask_root, mask_names, split, data_types, lstm_steps):
        super(NetCDFLoader, self).__init__()
        assert len(img_names) == len(mask_names) == len(data_types)

        self.split = split
        self.data_types = data_types
        self.img_names = img_names
        self.mask_names = mask_names
        self.lstm_steps = lstm_steps

        if split == 'train':
            self.data_path = '{:s}/data_large/'.format(data_root)
        elif split == 'test' or split == 'infill':
            self.data_path = '{:s}/test_large/'.format(data_root)
        elif split == 'val':
            self.data_path = '{:s}/val_large/'.format(data_root)
        self.mask_path = mask_root

        img_lengths = []
        self.mask_lengths = []
        # define lengths of image and mask
        for i in range(len(self.data_types)):
            img_file = h5py.File('{}{}'.format(self.data_path, self.img_names[i]), 'r')
            img_data = img_file.get(self.data_types[i])
            print(self.img_names[i])
            print(self.data_types[i])
            img_lengths.append(len(img_data[:, 1, 1]))
            mask_file = h5py.File('{}{}'.format(self.mask_path, self.mask_names[i]), 'r')
            mask_data = mask_file.get(self.data_types[i])
            self.mask_lengths.append(len((mask_data[:, 1, 1])))

        # check if images all have same length
        assert img_lengths[:-1] == img_lengths[1:]
        self.img_length = img_lengths[0]

    def load_data(self):
        # open netcdf file for img and mask
        img_data = []
        mask_data = []
        for i in range(len(self.data_types)):
            img_file = h5py.File('{}{}'.format(self.data_path, self.img_names[i]), 'r')
            img_data.append(img_file.get(self.data_types[i]))
            mask_file = h5py.File('{}{}'.format(self.mask_path, self.mask_names[i]))
            mask_data.append(mask_file.get(self.data_types[i]))
        return img_data, mask_data

    def __getitem__(self, index):
        img_data, mask_data = self.load_data()

        img_total = []
        mask_total = []

        for i in range(len(img_data)):
            img_inner = []
            mask_inner = []
            for j in range(self.lstm_steps + 1):
                # img
                new_index = index - j
                if new_index < 0:
                    new_index = 0
                # get next img
                img_prev = img_data[i][new_index, :, :]
                img_prev = torch.from_numpy(img_prev[:, :])
                img_prev = img_prev.unsqueeze(0)
                img_inner.insert(0, img_prev)

                # mask
                new_index = index - j
                if new_index < 0:
                    new_index = 0
                # get next mask
                if self.split == 'infill':
                    mask_prev = mask_data[i][new_index, :, :]
                else:
                    mask_prev = mask_data[i][random.randint(0, self.mask_lengths[i] - 1), :, :]
                mask_prev = torch.from_numpy(mask_prev[:, :])
                mask_prev = mask_prev.unsqueeze(0)
                mask_inner.insert(0, mask_prev)
            img_inner = torch.cat(img_inner)
            mask_inner = torch.cat(mask_inner)
            img_total.append(img_inner)
            mask_total.append(mask_inner)

        img_total = torch.stack(img_total, dim=1)
        mask_total = torch.stack(mask_total, dim=1)

        return img_total * mask_total, mask_total, img_total

    def __len__(self):
        return self.img_length
