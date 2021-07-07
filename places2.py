import random
import torch
#import torch.multiprocessing as mp
#mp.set_start_method('spawn')
import h5py
import numpy as np

from PIL import Image
from glob import glob
import torch.utils.data as data

class Places2(torch.utils.data.Dataset):
    def __init__(self, img_root, mask_root, img_transform, mask_transform,
                 split='train'):
        super(Places2, self).__init__()
        
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        # use about 8M images in the challenge dataset
        if split == 'train':
            self.paths = glob('{:s}/data_large/**/*.h5'.format(img_root),
                              recursive=True)
        else:
            self.paths = glob('{:s}/{:s}_large/*.h5'.format(img_root, split))

        #self.h5_file = h5py.File('{:s}'.format(self.paths[0]), 'r')
        #self.hdata = self.h5_file.get('tas')
        #self.leng = len((self.hdata[:,1,1]))
        #print(self.leng)
        #print(self.hdata)
        self.maskpath = mask_root
        #self.mask_file = h5py.File(mask_root)
        #self.maskdata = self.mask_file.get('tas')
        #self.N_mask = len((self.maskdata[:,1,1]))
            
    def __getitem__(self, index):
        h5_file = h5py.File('{:s}'.format(self.paths[0]), 'r')
        hdata = h5_file.get('tas')
        gt_img = hdata[index,:,:]
        #gt_img -= np.mean(gt_img) # the -= means can be read as x = x- np.mean(x)
        #gt_img /= np.std(gt_img) # the /= means can be read as x = x/np.std(x)
        gt_img = torch.from_numpy(gt_img[:,:])#.float()
        gt_img = gt_img.unsqueeze(0)
        a = gt_img[0,:,:]
        gt_img = a.repeat(3, 1, 1)
        #gt_img = self.img_transform(gt_img)
        
        mask_file = h5py.File(self.maskpath)
        maskdata = mask_file.get('tas')
        N_mask = len((maskdata[:,1,1]))
        #mask = torch.from_numpy(maskdata[index,:,:])
        mask = torch.from_numpy(maskdata[random.randint(0, N_mask - 1),:,:])#.float()
        mask = mask.unsqueeze(0)
        b = mask[0,:,:]
        mask = b.repeat(3, 1, 1)

        #print(gt_img)
        #print(mask)
        
        return gt_img * mask, mask, gt_img

    def __len__(self):
        h5_file = h5py.File('{:s}'.format(self.paths[0]), 'r')
        hdata = h5_file.get('tas')
        leng = len(hdata[:,1,1])
        return leng
        
