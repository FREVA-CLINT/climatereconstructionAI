import random
import torch
#import torch.multiprocessing as mp
#mp.set_start_method('spawn')
import h5py
import numpy as np
import local_settings

from PIL import Image
from glob import glob
import torch.utils.data as data

class Places2(torch.utils.data.Dataset):
    def __init__(self, img_root, mask_root, img_transform, mask_transform,
                 split='train'):
        super(Places2, self).__init__()
        self.split = split
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        # use about 8M images in the challenge dataset
        print(img_root)
        if split == 'train':
            self.paths = glob('{:s}/data_large/*.h5'.format(img_root),
                              recursive=True)
        elif split == 'test':
            self.paths = glob('{:s}/test_large/*.h5'.format(img_root))
        elif split == 'val':
            self.paths = glob('{:s}/val_large/*.h5'.format(img_root))
        self.maskpath = mask_root
            
    def __getitem__(self, index):
        h5_file = h5py.File('{:s}'.format(self.paths[0]), 'r')
        hdata = h5_file.get(local_settings.data_type)

        gt_img = hdata[index,:,:]
        gt_img = torch.from_numpy(gt_img[:,:])
        gt_img = gt_img.unsqueeze(0)

        if local_settings.prev_next_train:
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
            #time_data = h5_file.get('time')
            #time_stamp = time_data[index]
            gt_img_time = []
            #time_stamp = torch.empty(gt_img.shape[0], gt_img.shape[1], gt_img.shape[2]).fill_(time_stamp)
            gt_img_time.append(gt_img_prev)
            gt_img_time.append(gt_img)
            gt_img_time.append(gt_img_next)
            gt_img = torch.cat(gt_img_time)
        else:
            b = gt_img[0, :, :]
            gt_img = b.repeat(3, 1, 1)

        mask_file = h5py.File(self.maskpath)
        maskdata = mask_file.get(local_settings.data_type)
        N_mask = len((maskdata[:, 1, 1]))
        if self.split == 'infill':
            mask = torch.from_numpy(maskdata[index, :, :])
        else:
            mask = torch.from_numpy(maskdata[random.randint(0, N_mask - 1), :, :])
        mask = mask.unsqueeze(0)

        b = mask[0, :, :]
        mask = b.repeat(3, 1, 1)

        return gt_img * mask, mask, gt_img

    def __len__(self):
        h5_file = h5py.File('{:s}'.format(self.paths[0]), 'r')
        hdata = h5_file.get(local_settings.data_type)
        leng = len(hdata[:,1,1])
        return leng
