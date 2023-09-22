import os
import random
import copy

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset, Sampler

from .netcdfchecker import dataset_formatter
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
        n_samples = self.num_samples 
        order = np.random.permutation(n_samples) 
        while True:
            yield order[i]
            i += 1
            if i >= n_samples:
                order = np.random.permutation(n_samples) 
                i = 0

class ds_norm(torch.nn.Module):
    def __init__(self, moments=tuple()):
        super().__init__()
        self.moments = moments
        
    def __call__(self, data):
        data_norm, self.moments = norm_mm(data, moments=self.moments)
        return data_norm


def norm_mm(data, min_max_output=(0,1), moments=tuple(), epsilon=1e-15):
    if len(moments)==0:
        #moments = (data.min(), data.max())
        moments = (np.quantile(data, 0.05), np.quantile(data, 0.95))

    data_norm = (data-moments[0])/(epsilon + moments[1]-moments[0])

    data_norm = data_norm * (min_max_output[1] - min_max_output[0]) + min_max_output[0]
    return data_norm, moments


class FiniteSampler(Sampler):
    def __init__(self, num_samples, data_source=None):
        super().__init__(data_source)
        self.num_samples = num_samples

    def __iter__(self):
        return iter(range(self.num_samples))

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
    def __init__(self, data_root, img_names_source, img_names_target, split, data_types, coord_names, apply_img_norm=False, normalize_data=True, random_region=None, norm_stats=tuple(), p_input_dropout=0):
        super(NetCDFLoader, self).__init__()
        
        self.PosCalc = PositionCalculator()
        self.random = random.Random()
        self.data_types = data_types
        self.coord_names = coord_names
        self.apply_img_norm = apply_img_norm
        self.normalize_data = normalize_data
        self.p_input_dropout = p_input_dropout
        
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

        lons_s = torch.tensor(self.ds_source[self.coord_names[0][0]].values)
        lats_s = torch.tensor(self.ds_source[self.coord_names[0][1]].values)
        lons_t = torch.tensor(self.ds_target[self.coord_names[1][0]].values)
        lats_t = torch.tensor(self.ds_target[self.coord_names[1][1]].values)

        if lons_s.max()>torch.pi:
            lons_s = lons_s.deg2rad()
            lats_s = lats_s.deg2rad()
            lons_t = lons_t.deg2rad()
            lats_t = lats_t.deg2rad()
        
        lo_s, la_s, lo_t, la_t = len(lons_s), len(lats_s), len(lons_t), len(lats_t)

        if self.flatten:
            lons_s = lons_s.flatten().repeat(la_s)
            lats_s = lats_s.view(-1,1).repeat(1,lo_s).flatten()
            lons_t = lons_t.flatten().repeat(la_t)
            lats_t = lats_t.view(-1,1).repeat(1,lo_t).flatten()

        if random_region is not None:

            self.generate_region_prob = 1/random_region['generate_interval']

            if "n_points_hr" not in random_region.keys():
                self.region_generator = random_region_generator(torch.tensor(random_region['lon_range']), 
                                                                torch.tensor(random_region['lat_range']),
                                                                lons_s,
                                                                lats_s,
                                                                lons_t,
                                                                lats_t,
                                                                torch.tensor(random_region['radius_factor']),
                                                                batch_size=torch.tensor(random_region['batch_size']),
                                                                radius_target=torch.tensor(random_region['radius_target']))
            else:
                self.region_generator = random_region_generator(torch.tensor(random_region['lon_range']), 
                                                                torch.tensor(random_region['lat_range']),
                                                                lons_s,
                                                                lats_s,
                                                                lons_t,
                                                                lats_t,
                                                                torch.tensor(random_region['radius_factor']),
                                                                n_points_hr=torch.tensor(random_region['n_points_hr']))
            self.coord_dict = {}
            self.generate_region()

        else:
            self.region_generator=None

            coord_dict_lr  = {'lons': lons_s,
                   'lats': lats_s}
            
            coord_dict_hr  = {'lons': lons_t,
                   'lats': lats_t}
            
            coord_dict = {'lr':coord_dict_lr, 'hr':coord_dict_hr, 'seeds':[torch.tensor([coord_dict_lr['lons'].median()]), torch.tensor([coord_dict_lr['lats'].median()])]}

            
            self.coord_dict = self.get_coords(coord_dict)
        
        self.n_source = int((1-p_input_dropout) * self.coord_dict['rel']['source'][0].shape[0])

        value_data = np.concatenate([self.ds_source[data_types[0]].values.flatten(), self.ds_target[data_types[0]].values.flatten()])
        self.normalizer = ds_norm(norm_stats)  
        self.normalizer(value_data)


    def input_dropout(self, x, coord_dict):

        coords = copy.deepcopy(coord_dict)

        indices = torch.randperm(x.shape[0]-1)[:self.n_source]

        x = x[indices,:]

        coords['rel']['source'] = coords['rel']['source'][:,indices,:]

        return x, coords


    def generate_region(self):
        self.region_dict = self.region_generator.generate()
        self.coord_dict = self.get_coords(self.region_dict)


    def get_coords(self, coord_dict):

     #   d_mat_lr_hr, d_phi_lr_hr, d_lon_lr_hr, d_lat_lr_hr = self.PosCalc(coord_dict['lr']['lons'], coord_dict['lr']['lats'], coord_dict['hr']['lons'], coord_dict['hr']['lats'])

     #   d_mat_lr_hr, d_phi_lr_lr, d_lon_lr_lr, d_lat_lr_lr  = self.PosCalc(coord_dict['lr']['lons'], coord_dict['lr']['lats'], coord_dict['lr']['lons'], coord_dict['lr']['lats'])

     #   d_mat_hr_hr, d_phi_hr_hr, d_lon_hr_hr, d_lat_hr_hr  = self.PosCalc(coord_dict['hr']['lons'], coord_dict['hr']['lats'], coord_dict['hr']['lons'], coord_dict['hr']['lats'])

        _, _, d_lons_s, d_lats_s = self.PosCalc(coord_dict['lr']['lons'], coord_dict['lr']['lats'], (coord_dict['seeds'][0]), (coord_dict['seeds'][1]))

        _, _, d_lons_t, d_lats_t = self.PosCalc(coord_dict['hr']['lons'], coord_dict['hr']['lats'], (coord_dict['seeds'][0]), (coord_dict['seeds'][1]))

         
        #rel_coords = {'source': torch.stack([d_lon_lr_lr.float(), d_lat_lr_lr.float()],dim=0),
        #            'target': torch.stack([d_lon_hr_hr.float(), d_lat_hr_hr.float()],dim=0),
        #            'target-source': torch.stack([d_lon_lr_hr.float(), d_lat_lr_hr.float()],dim=0)}
        
        rel_coords =  {'source': torch.stack([d_lons_s.float().T, d_lats_s.float().T],dim=0),
                    'target': torch.stack([d_lons_t.float().T, d_lats_t.float().T],dim=0)}
        
        coord_dict = {'rel': rel_coords}

        return coord_dict
    

    def __getitem__(self, index):

        if self.region_generator is not None:
            if torch.rand(1) < self.generate_region_prob:
                self.generate_region()

        data_source = torch.tensor(self.ds_source[self.data_types[0]][index].values)
        data_target = torch.tensor(self.ds_target[self.data_types[0]][index].values)

        if self.normalize_data:
            data_source = self.normalizer(data_source)
            data_target = self.normalizer(data_target)
            
        if self.flatten:
            data_source = data_source.flatten().unsqueeze(dim=1)
            data_target = data_target.flatten().unsqueeze(dim=1)
        else:
            data_source = data_source.view(-1,1)
            data_target = data_target.view(-1,1)


        if self.region_generator is not None:
            data_source = data_source[self.region_dict['lr']['indices'][:,0]]
            data_target = data_target[self.region_dict['hr']['indices'][:,0]]


        if self.apply_img_norm:
            norm = ds_norm()
            data_source = norm(data_source)
            data_target = norm(data_target)

        if self.p_input_dropout > 0:
            data_source, coord_dict = self.input_dropout(data_source, self.coord_dict)
        else:
            coord_dict = self.coord_dict

        return data_source, data_target, coord_dict

    def __len__(self):
        return self.num_tp
