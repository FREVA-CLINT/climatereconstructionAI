import os
import random

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset, Sampler

from .netcdfchecker import dataset_formatter
from .. import config as cfg
from .grid_utils import random_region_generator, PositionCalculator


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
        n_samples = self.num_samples - sum(cfg.time_steps)
        order = np.random.permutation(n_samples) + cfg.time_steps[0]
        while True:
            yield order[i]
            i += 1
            if i >= n_samples:
                order = np.random.permutation(n_samples) + cfg.time_steps[0]
                i = 0


class img_norm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.moments = tuple()
        
    def __call__(self, img):
        img_norm, moments = norm_img_mm(img, moments=self.moments)
        self.moments = moments
        return img_norm
    
def norm_img_mm(image, min_max_output=(0,1), moments=tuple(), epsilon=1e-15):
    if len(moments)==0:
        img_norm = (image-image.min())/(epsilon + image.max()-image.min())
        moments = (image.min(), image.max())
    else:
        img_norm = (image-moments[0])/(epsilon + moments[1]-moments[0])

    img_norm = img_norm * (min_max_output[1] - min_max_output[0]) + min_max_output[0]
    return img_norm, moments


class FiniteSampler(Sampler):
    def __init__(self, num_samples, data_source=None):
        super().__init__(data_source)
        self.num_samples = num_samples

    def __iter__(self):
        return iter(range(cfg.time_steps[0], self.num_samples - cfg.time_steps[1]))

    def __len__(self):
        return self.num_samples


def nc_loadchecker(filename, data_type):
    basename = filename.split("/")[-1]

    if not os.path.isfile(filename):
        print('File {} not found.'.format(filename))

    try:
        ds = xr.open_dataset(filename)
    except Exception:
        try:
            ds = xr.open_dataset(filename, decode_times=False)
        except Exception:
            raise ValueError('Impossible to read {}.'
                             '\nPlease, check that it is a netCDF file and it is not corrupted.'.format(basename))

    ds1 = dataset_formatter(ds, data_type, basename)
    ds = ds.drop_vars(data_type)

    if cfg.lazy_load:
        data = ds1[data_type]
    else:
        data = ds1[data_type].values

    dims = ds1[data_type].dims
    coords = {key: ds1[data_type][key] for key in dims if key != "time"}

    return [ds, ds1, dims, coords], data, data.shape[0], data.shape[1:]


def load_netcdf(path, data_names, data_types, keep_dss=False):
    if data_names is None:
        return None, None, None
    else:
        ndata = len(data_names)
        assert ndata == len(data_types)

        dss, data, lengths, sizes = zip(*[nc_loadchecker('{}{}'.format(path, data_names[i]),
                                                         data_types[i]) for i in range(ndata)])

        assert len(set(lengths)) == 1

        if keep_dss:
            return dss, data, lengths[0], sizes
        else:
            return data, lengths[0], sizes


class NetCDFLoader(Dataset):
    def __init__(self, data_root, img_names_source, img_names_target, split, data_types, coord_names, random_region=None):
        super(NetCDFLoader, self).__init__()
        
        self.PosCalc = PositionCalculator()
        self.random = random.Random(cfg.loop_random_seed)
        self.data_types = data_types
        self.coord_names = coord_names

        if 'lon' in self.coord_names[0]:
            self.flatten=True
        else:
            self.flatten=False

        if split == 'train':
            data_path = '{:s}/train/'.format(data_root)
        else:
            data_path = '{:s}/val/'.format(data_root)

        ds_targets = []
        for img_names_target in img_names_target:
            ds_targets.append(xr.load_dataset(os.path.join(data_path, img_names_target)))

        self.ds_target = xr.concat(ds_targets, dim='time')


        ds_sources = []
        for img_name_source in img_names_source:
            ds_sources.append(xr.load_dataset(os.path.join(data_path, img_name_source)))

        self.ds_source = xr.concat(ds_sources, dim='time')

        self.num_tp = self.ds_target[data_types[0]].shape[0]

        if random_region is not None:
            self.region_generator = random_region_generator(torch.tensor(random_region['lon_range']), 
                                                            torch.tensor(random_region['lat_range']),
                                                            torch.tensor(self.ds_source[coord_names[0][0]].values),
                                                            torch.tensor(self.ds_source[coord_names[0][1]].values),
                                                            torch.tensor(self.ds_target[coord_names[1][0]].values),
                                                            torch.tensor(self.ds_target[coord_names[1][1]].values),
                                                            torch.tensor(random_region['radius_target']),
                                                            torch.tensor(random_region['radius_factor']),
                                                            torch.tensor(random_region['batch_size']))
            self.coord_dict = {}
            self.generate_region()

        else:
            self.region_generator=None

            lons_s = torch.tensor(self.ds_source[self.coord_names[0][0]].values)
            lats_s = torch.tensor(self.ds_source[self.coord_names[0][1]].values)
            lons_t = torch.tensor(self.ds_target[self.coord_names[1][0]].values)
            lats_t = torch.tensor(self.ds_target[self.coord_names[1][1]].values)

            if self.flatten:
                lons_s = lons_s.flatten().repeat(len(lons_s))
                lats_s = lats_s.flatten().repeat(len(lats_s))
                lons_t = lons_t.flatten().repeat(len(lons_t))
                lats_t = lats_t.flatten().repeat(len(lats_t))

            coord_dict_lr  = {'lons': lons_s,
                   'lats': lats_s}
            
            coord_dict_hr  = {'lons': lons_t,
                   'lats': lats_t}
            
            coord_dict = {'lr':coord_dict_lr, 'hr':coord_dict_hr, 'seeds':[torch.tensor([coord_dict_lr['lons'].median()]),torch.tensor([coord_dict_lr['lats'].median()])]}

            
            self.coord_dict = self.get_coords(coord_dict)

    def generate_region(self):
        self.region_dict = self.region_generator.generate()
        self.coord_dict = self.get_coords(self.region_dict)
        if self.coord_dict['abs']['target'][0].abs().mean()>1:
            pass


    def get_coords(self, coord_dict):

        _, _, d_lon_lr_hr, d_lat_lr_hr = self.PosCalc(coord_dict['lr']['lons'], coord_dict['lr']['lats'], coord_dict['hr']['lons'], coord_dict['hr']['lats'])

        _, _, d_lon_lr_lr, d_lat_lr_lr  = self.PosCalc(coord_dict['lr']['lons'], coord_dict['lr']['lats'], coord_dict['lr']['lons'], coord_dict['lr']['lats'])

        _, _, d_lon_hr_hr, d_lat_hr_hr  = self.PosCalc(coord_dict['hr']['lons'], coord_dict['hr']['lats'], coord_dict['hr']['lons'], coord_dict['hr']['lats'])

        _, _, d_lons_s, d_lats_s = self.PosCalc(coord_dict['lr']['lons'], coord_dict['lr']['lats'], (coord_dict['seeds'][0]), (coord_dict['seeds'][1]))

        _, _, d_lons_t, d_lats_t = self.PosCalc(coord_dict['hr']['lons'], coord_dict['hr']['lats'], (coord_dict['seeds'][0]), (coord_dict['seeds'][1]))

         
        rel_coords = {'source': [d_lon_lr_lr.float().to(cfg.device), d_lat_lr_lr.float().to(cfg.device)],
                    'target': [d_lon_hr_hr.float().to(cfg.device), d_lat_hr_hr.float().to(cfg.device)],
                    'target-source': [d_lon_lr_hr.float().to(cfg.device), d_lat_lr_hr.float().to(cfg.device)]}
        
        abs_coords = {'source': [d_lons_s.float().to(cfg.device), d_lats_s.float().to(cfg.device)],
                    'target': [d_lons_t.float().to(cfg.device), d_lats_t.float().to(cfg.device)]}
        
        return {'rel': rel_coords, 'abs': abs_coords}
    

    def __getitem__(self, index):

        data_source = torch.tensor(self.ds_source[self.data_types[0]][index].values)
        data_target = torch.tensor(self.ds_target[self.data_types[0]][index].values)

        if self.flatten:
            data_source = data_source.flatten().unsqueeze(dim=1)
            data_target = data_target.flatten().unsqueeze(dim=1)
        else:
            data_source = data_source.view(-1,1)
            data_target = data_target.view(-1,1)


        if self.region_generator is not None:
            data_source = data_source.view(-1,1)
            data_target = data_target.view(-1,1)
            data_source = data_source[self.region_dict['lr']['indices'][:,0]]
            data_target = data_target[self.region_dict['hr']['indices'][:,0]]


        if cfg.apply_img_norm:
            norm = img_norm()
            data_source = norm(data_source)
            data_target = norm(data_target)

        return data_source, data_target

    def __len__(self):
        return self.num_tp
