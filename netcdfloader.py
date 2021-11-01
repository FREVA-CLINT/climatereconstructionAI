import random
import torch
import h5py
from glob import glob
import torch.utils.data as data


class NetCDFLoader(torch.utils.data.Dataset):
    def __init__(self, img_root, mask_root, img_transform, mask_transform, split, data_type):
        super(NetCDFLoader, self).__init__()
        self.split = split
        self.data_type = data_type
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        # use about 8M images in the challenge dataset
        if split == 'train':
            self.paths = glob('{:s}/data_large/*.h5'.format(img_root),
                              recursive=True)
        elif split == 'test' or split == 'infill':
            self.paths = glob('{:s}/test_large/*.h5'.format(img_root))
        elif split == 'val':
            self.paths = glob('{:s}/val_large/*.h5'.format(img_root))
        self.maskpath = mask_root

        # define length
        h5_file = h5py.File('{:s}'.format(self.paths[0]), 'r')
        hdata = h5_file.get(self.data_type)
        self.length = len(hdata[:, 1, 1])

    def __len__(self):
        return self.length


class SingleImageNetCDFLoader(NetCDFLoader):
    def __getitem__(self, index):
        h5_file = h5py.File('{:s}'.format(self.paths[0]), 'r')
        hdata = h5_file.get(self.data_type)

        gt_img = hdata[index, :, :]
        gt_img = torch.from_numpy(gt_img[:, :])
        gt_img = gt_img.unsqueeze(0)

        mask_file = h5py.File(self.maskpath)
        maskdata = mask_file.get(self.data_type)
        N_mask = len((maskdata[:, 1, 1]))
        if self.split == 'infill':
            mask = torch.from_numpy(maskdata[index, :, :])
        else:
            mask = torch.from_numpy(maskdata[random.randint(0, N_mask - 1), :, :])
        mask = mask.unsqueeze(0)

        return gt_img * mask, mask, gt_img


class PrevNextImageNetCDFLoader(NetCDFLoader):
    def __getitem__(self, index):
        h5_file = h5py.File('{:s}'.format(self.paths[0]), 'r')
        hdata = h5_file.get(self.data_type)

        gt_img = hdata[index,:,:]
        gt_img = torch.from_numpy(gt_img[:,:])
        gt_img = gt_img.unsqueeze(0)

        if index == 0:
            gt_img_prev = hdata[index, :, :]
        else:
            gt_img_prev = hdata[index - 1, :, :]
        gt_img_prev = torch.from_numpy(gt_img_prev[:, :])
        gt_img_prev = gt_img_prev.unsqueeze(0)
        if index == self.__len__() - 1:
            gt_img_next = hdata[index, :, :]
        else:
            gt_img_next = hdata[index + 1, :, :]
        gt_img_next = torch.from_numpy(gt_img_next[:, :])
        gt_img_next = gt_img_next.unsqueeze(0)
        gt_img_time = []
        gt_img_time.append(gt_img_prev)
        gt_img_time.append(gt_img)
        gt_img_time.append(gt_img_next)
        gt_img = torch.cat(gt_img_time)

        mask_file = h5py.File(self.maskpath)
        maskdata = mask_file.get(self.data_type)
        N_mask = len((maskdata[:, 1, 1]))
        if self.split == 'infill':
            mask = torch.from_numpy(maskdata[index, :, :])
        else:
            mask = torch.from_numpy(maskdata[random.randint(0, N_mask - 1), :, :])
        mask = mask.unsqueeze(0)[0, :, :].repeat(3, 1, 1)
        return gt_img * mask, mask, gt_img
