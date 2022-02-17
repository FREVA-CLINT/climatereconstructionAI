import random
import numpy as np
import torch
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

def get_data(path,data_names,data_types):

    if data_names is None:
        return None, None
    else:
        ndata = len(data_names)
        assert ndata == len(data_types)

        data, shape = [], []
        for i in range(ndata):
            data.append(xr.open_dataset('{}{}'.format(path, data_names[i])))
            shape.append(data[-1][data_types[i]].shape)

        assert len(set(shape)) == 1

        return data, shape[0][0]

class NetCDFLoader(Dataset):
    def __init__(self, data_root, img_names, mask_root, mask_names, split, data_types, lstm_steps, prev_next_steps):
        super(NetCDFLoader, self).__init__()
        assert lstm_steps == 0 or prev_next_steps == 0

        self.data_types = data_types
        self.split = "infill"
        self.lstm_steps = lstm_steps
        self.prev_next_steps = prev_next_steps

        if split == 'train':
            data_path = '{:s}/data_large/'.format(data_root)
        elif split == 'test' or split == 'infill':
            data_path = '{:s}/test_large/'.format(data_root)
        elif split == 'val':
            data_path = '{:s}/val_large/'.format(data_root)

        self.img_data, self.img_length = get_data(data_path,img_names,data_types)
        self.mask_data, self.mask_length = get_data(mask_root,mask_names,data_types)

        if self.split == "infill" and not self.mask_data is None:
            assert self.img_length == self.mask_length


    def load_data(self, ind_data, indices):

        data_type = self.data_types[ind_data]
        image = self.img_data[ind_data][data_type].values[indices]
        if self.mask_data is None:
            # Get masks from images
            mask = torch.from_numpy((1-(np.isnan(image))).astype(image.dtype))
            image = np.nan_to_num(image)
        else:
            mask = torch.from_numpy(self.mask_data[ind_data][data_type].values[indices])

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
        return torch.from_numpy(image), mask

    def get_single_item(self, ind_data, index):
        if self.lstm_steps == 0:
            prev_steps = next_steps = self.prev_next_steps
        else:
            prev_steps = next_steps = self.lstm_steps

        # define range of lstm or prev-next steps -> adjust, if out of boundaries
        img_indices = np.array(list(range(index - prev_steps, index + next_steps + 1)))
        img_indices[img_indices < 0] = 0
        img_indices[img_indices > self.img_length - 1] = self.img_length - 1
        if self.split == 'infill':
            mask_indices = img_indices
        else:
            mask_indices = []
            for j in range(prev_steps + next_steps + 1):
                mask_indices.append(random.randint(0, self.mask_length - 1))
            mask_indices = sorted(mask_indices)

        # load data from ranges
        images, masks = self.load_data(ind_data, img_indices)

        # stack to correct dimensions
        if self.lstm_steps == 0:
            images = torch.cat([images], dim=0).unsqueeze(0)
            masks = torch.cat([masks], dim=0).unsqueeze(0)
        else:
            images = torch.stack([images], dim=1)
            masks = torch.stack([masks], dim=1)

        return images, masks

    def __getitem__(self, index):

        image, mask = self.get_single_item(0,index)
        images = []
        masks = []
        for i in range(1, len(self.data_types)):
            img, m = self.get_single_item(i,index)
            images.append(img)
            masks.append(m)
        if images and masks:
            images = torch.cat(images, dim=1)
            masks = torch.cat(masks, dim=1)
            return mask*image, mask, image, masks*images, masks, images
        else:
            return mask*image, mask, image, torch.tensor([]), torch.tensor([]), torch.tensor([])

    def __len__(self):
        return self.img_length
