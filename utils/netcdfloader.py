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
    def __init__(self, data_root, img_names, mask_root, mask_names, split, data_types, lstm_steps, prev_next_steps):
        super(NetCDFLoader, self).__init__()
        assert lstm_steps == 0 or prev_next_steps == 0
        assert len(img_names) == len(mask_names) == len(data_types)

        self.split = split
        self.data_types = data_types
        self.img_names = img_names
        self.mask_names = mask_names
        self.lstm_steps = lstm_steps
        self.prev_next_steps = prev_next_steps

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

    def load_data(self, img_indices, mask_indices):
        images = []
        masks = []
        # iterate each defined netcdf file
        for i in range(len(self.data_types)):
            # open netcdf file for img and mask
            img_file = h5py.File('{}{}'.format(self.data_path, self.img_names[i]), 'r')
            img_data = img_file.get(self.data_types[i])
            try:
                if img_data.ndim == 4:
                    images.append(torch.from_numpy(img_data[img_indices[i], 0, :, :]))
                else:
                    images.append(torch.from_numpy(img_data[img_indices[i], :, :]))
            except (ValueError, TypeError):
                # get indices that occur more than once
                unique, counts = np.unique(img_indices[i], return_counts=True)
                copy_indices = [(index, counts[index] - 1) for index in range(len(counts)) if counts[index] > 1]
                if img_data.ndim == 4:
                    data = torch.from_numpy(img_data[unique, 0, :, :])
                else:
                    data = torch.from_numpy(img_data[unique, :, :])
                if unique[copy_indices[0][0]] == 0:
                    total_data = torch.cat([torch.stack(copy_indices[0][1] * [data[copy_indices[0][0]]]), data])
                else:
                    total_data = torch.cat([data, torch.stack(copy_indices[0][1] * [data[copy_indices[0][0]]])])
                images.append(total_data)
            mask_file = h5py.File('{}{}'.format(self.mask_path, self.mask_names[i]))
            mask_data = mask_file.get(self.data_types[i])
            print(i)
            try:
                print(mask_indices)
                if mask_data.ndim == 4:
                    masks.append(torch.from_numpy(mask_data[mask_indices[i], 0, :, :]))
                else:
                    masks.append(torch.from_numpy(mask_data[mask_indices[i], :, :]))
            except (ValueError, TypeError):
                # get indices that occur more than once
                unique, counts = np.unique(mask_indices[i], return_counts=True)
                copy_indices = [(index, counts[index] - 1) for index in range(len(counts)) if counts[index] > 1]
                if img_data.ndim == 4:
                    data = torch.from_numpy(mask_data[unique, 0, :, :])
                else:
                    data = torch.from_numpy(mask_data[unique, :, :])
                if unique[copy_indices[0][0]] == 0:
                    total_data = torch.cat([torch.stack(copy_indices[0][1] * [data[copy_indices[0][0]]]), data])
                else:
                    total_data = torch.cat([data, torch.stack(copy_indices[0][1] * [data[copy_indices[0][0]]])])
                masks.append(total_data)

        return images, masks

    def __getitem__(self, index):
        if self.lstm_steps == 0:
            prev_steps = next_steps = self.prev_next_steps
        else:
            prev_steps = self.lstm_steps
            next_steps = 0

        # define range of lstm steps -> adjust, if out of boundaries
        img_indices = np.array(list(range(index - prev_steps, index + next_steps + 1)))
        img_indices[img_indices < 0] = 0
        img_indices[img_indices > self.img_length - 1] = self.img_length - 1
        img_indices = len(self.data_types) * [img_indices]
        if self.split == 'infill':
            mask_indices = img_indices
        else:
            mask_indices = []
            for i in range(len(self.data_types)):
                indices = []
                for j in range(prev_steps + next_steps + 1):
                    indices.append(random.randint(0, self.mask_lengths[i] - 1))
                mask_indices.append(sorted(indices))

        # load data from ranges
        images, masks = self.load_data(img_indices, mask_indices)

        # stack to correct dimensions
        if self.lstm_steps == 0:
            images = torch.cat(images, dim=0).unsqueeze(0)
            masks = torch.cat(masks, dim=0).unsqueeze(0)
        else:
            images = torch.stack(images, dim=1)
            masks = torch.stack(masks, dim=1)
        return images * masks, masks, images

    def __len__(self):
        return self.img_length
