import random
import numpy as np
import torch
import h5py
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
            img_lengths.append(img_data.shape[0])
            mask_file = h5py.File('{}{}'.format(self.mask_path, self.mask_names[i]), 'r')
            mask_data = mask_file.get(self.data_types[i])
            self.mask_lengths.append(mask_data.shape[0])

        # check if images all have same length
        assert img_lengths[:-1] == img_lengths[1:]
        self.img_length = img_lengths[0]

        # if infill, check if img length matches mask length
        if split == 'infill':
            assert self.mask_lengths[:-1] == self.mask_lengths[1:]
            assert self.img_length == self.mask_lengths[0]

    def load_data(self, img_ranges, mask_ranges):
        images = []
        masks = []
        # iterate each defined netcdf file
        for i in range(len(self.data_types)):
            # open netcdf file for img and mask
            img_file = h5py.File('{}{}'.format(self.data_path, self.img_names[i]), 'r')
            img_data = img_file.get(self.data_types[i])
            if img_data.ndim == 4:
                images.append(torch.from_numpy(img_data[img_ranges[i][0]:img_ranges[i][1], 0, :, :]))
            else:
                images.append(torch.from_numpy(img_data[img_ranges[i][0]:img_ranges[i][1], :, :]))
            mask_file = h5py.File('{}{}'.format(self.mask_path, self.mask_names[i]))
            mask_data = mask_file.get(self.data_types[i])
            if mask_data.ndim == 4:
                masks.append(torch.from_numpy(mask_data[mask_ranges[i][0]:mask_ranges[i][1], 0, :, :]))
            else:
                masks.append(torch.from_numpy(mask_data[mask_ranges[i][0]:mask_ranges[i][1], :, :]))
        return images, masks

    def __getitem__(self, index):
        copy_first_img = 0
        copy_first_mask = 0

        # define range of lstm steps -> adjust, if out of boundaries
        if index < self.lstm_steps:
            img_ranges = len(self.data_types) * [(0, index + 1)]
            copy_first_img = self.lstm_steps - index
        else:
            img_ranges = len(self.data_types) * [(index - self.lstm_steps, index + 1)]
        if self.split == 'infill':
            mask_ranges = img_ranges
        else:
            mask_ranges = []
            for i in range(len(self.data_types)):
                mask_index = random.randint(0, self.mask_lengths[i] - 1)
                mask_ranges.append((mask_index, mask_index + 1))
            copy_first_mask = self.lstm_steps

        # load data from ranges
        images, masks = self.load_data(img_ranges, mask_ranges)

        # copy first image if necessary
        images = torch.stack(images, dim=1)
        if copy_first_img != 0:
            image = images[0, :, :, :]
            image = torch.stack(copy_first_img * [image], dim=0)
            images = torch.cat([image, images], dim=0)
        # copy first mask if necessary
        masks = torch.stack(masks, dim=1)
        if copy_first_mask != 0:
            mask = masks[0, :, :, :]
            mask = torch.stack(copy_first_mask * [mask], dim=0)
            masks = torch.cat([mask, masks], dim=0)

        return images * masks, masks, images

    def __len__(self):
        return self.img_length
