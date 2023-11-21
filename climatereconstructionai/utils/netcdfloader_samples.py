import os
import random
import copy
import json
from typing import Any
import datetime

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset, Sampler
from .grid_utils import generate_region, PositionCalculator, get_coord_dict_from_var, get_coords_as_tensor, invert_dict, rotate_coord_system

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

def calc_stats(data):
    stat_dict = {
        "min":data.min(),
        "max":data.max(),
        "q_05":np.quantile(data,0.05),
        "q_95":np.quantile(data,0.95),
        "mean":data.mean(),
        "std":data.std()
    }
    for key, value in stat_dict.items():
        stat_dict[key] = float(value)
    return stat_dict


class normalizer(torch.nn.Module):
    def __init__(self, stat_dict, type="quantile"):
        super().__init__()

        self.state_dict = stat_dict

        if type == 'quantile':
            self.norm_fcn = norm_min_max
            self.moments = ("q_05", "q_95")

        elif type == 'min_max':
            self.norm_fcn = norm_min_max
            self.moments = ("min", "max")

        else:
            self.norm_fcn = norm_mean_std
            self.moments = ("mean", "std")


    def __call__(self, data):
        for var, data_var in data.items():
             stats = (self.state_dict[var][self.moments[0]], self.state_dict[var][self.moments[1]])
             data[var] = self.norm_fcn(data_var, stats)[0]
        return data
        

def norm_min_max(data, moments):
    data = (data-moments[0])/(moments[1] - moments[0])
    return data, moments

def norm_mean_std(data, moments):
    data = (data-moments[0])/(moments[1])
    return data, moments


class FiniteSampler(Sampler):
    def __init__(self, num_samples, data_source=None):
        super().__init__(data_source)
        self.num_samples = num_samples

    def __iter__(self):
        return iter(range(self.num_samples))

    def __len__(self):
        return self.num_samples


def get_dims_coordinates(ds, variables, flatten=False):

    coord_dict_variables = {}
    
    spatial_dims = {}
    coord_dicts = {}
    for variable in variables:
        
        coord_dict = get_coord_dict_from_var(ds, variable)

        spatial_dims[variable] = coord_dict['spatial_dim']
        coord_dicts[coord_dict['spatial_dim']] = coord_dict

    coord_dict_variables['var_spatial_dims'] = spatial_dims
    coord_dict_variables['coord_dicts'] = coord_dicts
    coord_dict_variables['spatial_dims_var'] = invert_dict(coord_dict_variables['var_spatial_dims'])


    return coord_dict_variables

def save_sample(ds, time_index, spatial_dims_dict, save_path):

    ds_save = copy.deepcopy(ds['ds'])
    
    k = 0
    for spatial_dim, vars in spatial_dims_dict.items():

        dims = tuple(ds_save[vars[0]].dims)
        coord_names = get_coord_dict_from_var(ds_save, vars[0])
        indices = np.array(ds['spatial_dims'][spatial_dim]['region_indices']).squeeze()

        if k>0:
            time_index = 0

        if len(dims)==2:
            sel_indices = ([time_index], indices)
        else:
            sel_indices = ([time_index],[0],indices)
        
        sel_dict = dict(zip(dims, sel_indices))

        ds_save = ds_save.isel(sel_dict)

        coord_dict = {coord_names['lon']: ds['spatial_dims'][spatial_dim]['rel_coords'][0,:,0].numpy(),
                      coord_names['lat']: ds['spatial_dims'][spatial_dim]['rel_coords'][1,:,0].numpy()}

        ds_save = ds_save.assign_coords(coord_dict)

        k+=1

    ds_save.to_netcdf(save_path)

def get_stats(files, variable, n_sample=None):
    if (n_sample is not None) and (n_sample < len(files)):
        file_indices = np.random.randint(0,len(files), (n_sample,))
    else:
        file_indices = np.arange(len(files))
    
    data = np.concatenate([xr.load_dataset(file_index)[variable].values.flatten() for file_index in np.array(files)[file_indices]])

    return calc_stats(data)


class NetCDFLoader_lazy(Dataset):
    def __init__(self, files_source, 
                 files_target, 
                 variables_source, 
                 variables_target,
                 apply_img_norm=False, 
                 normalize_data=True, 
                 random_region=None, 
                 stat_dict=None, 
                 p_dropout_source=0,
                 p_dropout_target=0,
                 sampling_mode='mixed', 
                 n_points_s=None, 
                 n_points_t=None, 
                 coordinate_pert=0,
                 save_sample_path='',
                 index_range_source=None,
                 index_offset_target=0,
                 rel_coords=False,
                 sample_for_norm=-1,
                 norm_stats_save_path='',
                 lazy_load=False,
                 rotate_cs=False):
        
        super(NetCDFLoader_lazy, self).__init__()
        
        self.PosCalc = PositionCalculator()
        self.random = random.Random()
        self.variables_source = variables_source
        self.variables_target = variables_target
        self.apply_img_norm = apply_img_norm
        self.normalize_data = normalize_data
        self.p_dropout_source = p_dropout_source
        self.p_dropout_target = p_dropout_target
        self.sampling_mode = sampling_mode
        self.random_region=random_region
        self.n_points_s = n_points_s
        self.n_points_t = n_points_t
        self.coordinate_pert = coordinate_pert
        self.save_sample_path = save_sample_path
        self.index_range_source=index_range_source
        self.index_offset_target = index_offset_target
        self.rel_coords=rel_coords
        self.sample_for_norm = sample_for_norm
        self.norm_stats_save_path = norm_stats_save_path
        self.lazy_load=lazy_load
        self.rotate_cs = rotate_cs

        self.flatten=False

        if random_region is not None:
            self.generate_region_prob = 1/random_region['generate_interval']

        self.ds_dict = {}
 
        self.files_source = files_source
        self.files_target = files_target

        ds_source = xr.open_dataset(files_source[0])
        ds_target = xr.open_dataset(files_target[0])

        self.dims_variables_source = get_dims_coordinates(ds_source, self.variables_source)    
        self.dims_variables_target = get_dims_coordinates(ds_target, self.variables_target)         

        self.num_datapoints_time = ds_source[self.variables_source[0]].shape[0]

        start_seeds = [] if self.random_region is None or 'start_seeds' not in self.random_region.keys() else self.random_region['start_seeds']

        _, _, seeds, self.n_dict_source = self.get_coordinates(ds_source, self.dims_variables_source, p_drop=p_dropout_source, seeds=start_seeds)
        _, _, _, self.n_dict_target = self.get_coordinates(ds_target, self.dims_variables_target, seeds=seeds, p_drop=p_dropout_target)

        if stat_dict is None:
            self.stat_dict = {}

            unique_variables = np.unique(np.array(self.variables_source + self.variables_target))
          
            for var in unique_variables:
                files = []
                if var in self.variables_source:
                    files+=list(files_source)
                if var in self.variables_target:
                    files+=list(files_target)

                self.stat_dict[var] = get_stats(files, var, self.sample_for_norm)
            
            with open(os.path.join(self.norm_stats_save_path,"norm_stats.json"),"w+") as f:
                json.dump(self.stat_dict,f, indent=4)

        else:
            self.stat_dict=stat_dict

        self.normalizer = normalizer(self.stat_dict)



    def get_coordinates(self, ds, dims_variables_dict, seeds=[], n_drop_dict={}, p_drop=0):

        spatial_dim_indices = {}
        rel_coords_dict = {} 
        n_drop_dict = copy.deepcopy(n_drop_dict)
        seeds = copy.deepcopy(seeds)

        for spatial_dim, coord_dict in dims_variables_dict['coord_dicts'].items():
            
            coords = get_coords_as_tensor(ds, lon=coord_dict['lon'], lat=coord_dict['lat'])

            if self.random_region is not None:

                if len(seeds)==0:
                    radius = self.random_region["radius_source"] if "radius_source" in self.random_region.keys() else None
                    n_points = self.random_region["n_points_source"] if "n_points_source" in self.random_region.keys() else None
                    rect = self.random_region["rect_source"] if "rect_source" in self.random_region.keys() else False
                else:
                    seeds = [torch.tensor(seed).view(1,1) for seed in seeds] if not torch.is_tensor(seeds[0]) else seeds
                    radius = self.random_region["radius_target"] if "radius_target" in self.random_region.keys() else None
                    n_points = self.random_region["n_points_target"] if "n_points_target" in self.random_region.keys() else None
                    rect = self.random_region["rect_target"] if "rect_target" in self.random_region.keys() else False

                region_dict = generate_region({'lon': coords[0], 'lat': coords[1]}, 
                                              self.random_region['lon_range'], 
                                              self.random_region['lat_range'], 
                                              n_points=n_points, 
                                              radius=radius, batch_size=self.random_region['batch_size'], 
                                              locations=seeds, 
                                              rect=rect)

                seeds = region_dict['locations']

                indices = region_dict['indices'].squeeze()
            
            else:
                indices = torch.arange(coords.shape[1])
            
            if len(n_drop_dict) > 0 and spatial_dim in n_drop_dict.keys():
                n_drop = n_drop_dict[spatial_dim]

            elif p_drop > 0:
                n_drop = int((1-p_drop)*len(indices))
                n_drop_dict[spatial_dim] = n_drop

            if n_drop > len(indices):
                pad_indices = torch.randint(len(indices), size=(n_drop - len(indices),1)).view(-1)
                indices = torch.concat((indices, indices[pad_indices]))
            else:    
                indices = indices[torch.randperm(len(indices-1))[:n_drop]]

            coords = coords[:,indices]

            if not self.rel_coords:
                coords = {'lon': coords[0], 'lat': coords[1]}

                if len(seeds)==[]:
                    rel_coords, _  = self.get_rel_coords(coords, [coords[0].median(), coords[1].median()], rotate_cs=self.rotate_cs)
                else:
                    rel_coords, _  = self.get_rel_coords(coords, [seeds[0].deg2rad(), seeds[1].deg2rad()], rotate_cs=self.rotate_cs)  

            else:
                rel_coords = coords

            spatial_dim_indices[spatial_dim] = indices
            rel_coords_dict[spatial_dim] = rel_coords.squeeze()

        return spatial_dim_indices, rel_coords_dict, seeds, n_drop_dict


    def apply_spatial_dim_indices(self, ds, dims_variables_dict, spatial_dim_indices, rel_coords_dict={}):
        
        ds_return = copy.deepcopy(ds)

        for spatial_dim, vars in dims_variables_dict['spatial_dims_var'].items():
            
            coord_dict = dims_variables_dict['coord_dicts'][spatial_dim]

            spatial_indices = spatial_dim_indices[spatial_dim]

            dims = tuple(ds[vars[0]].dims)
           
            time_indices = np.arange(self.num_datapoints_time)

            if len(dims)==2:
                sel_indices = (time_indices, spatial_indices)
            else:
                sel_indices = (time_indices,[0],spatial_indices)
            
            sel_dict = dict(zip(dims, sel_indices))

            ds_return = ds_return.isel(sel_dict)
            
            if len(rel_coords_dict)==0:
                coords = coords[:,spatial_indices] 
                coords = get_coords_as_tensor(ds, lon=coord_dict['lon'], lat=coord_dict['lat'])
            else:
                coords = rel_coords_dict[spatial_dim]

            coord_dict = {coord_dict['lon']: coords[0].numpy(),
                        coord_dict['lat']: coords[1].numpy()}

            ds_return = ds_return.assign_coords(coords=coord_dict)

        return ds_return
    

    def get_rel_coords(self, coords, seeds, rotate_cs=False):
        
        if rotate_cs:
            distances, _, d_lons_s, d_lats_s = self.PosCalc(coords['lon'], coords['lat'], (seeds[0]), (seeds[1]), rotation_center=(seeds[0],seeds[1]))
        else:
            distances, _, d_lons_s, d_lats_s = self.PosCalc(coords['lon'], coords['lat'], (seeds[0]), (seeds[1]))
        
        return torch.stack([d_lons_s.float().T, d_lats_s.float().T],dim=0), distances
    

    def get_files(self, file_path_source, file_path_target=None):
        
        if self.lazy_load:
            ds_source = xr.open_dataset(file_path_source)
        else:
            ds_source = xr.load_dataset(file_path_source)

        if file_path_target is None:
            ds_target = copy.deepcopy(ds_source)
        else:
            if self.lazy_load:
                ds_target = xr.open_dataset(file_path_target)
            else:
                ds_target = xr.load_dataset(file_path_target)

        spatial_dim_indices_source, rel_coords_dict_source, seeds, _ = self.get_coordinates(ds_source, self.dims_variables_source, n_drop_dict=self.n_dict_source)
        spatial_dim_indices_target, rel_coords_dict_target, _, _ = self.get_coordinates(ds_target, self.dims_variables_target, seeds=seeds, n_drop_dict=self.n_dict_target)

        ds_source = self.apply_spatial_dim_indices(ds_source, self.dims_variables_source, spatial_dim_indices_source, rel_coords_dict=rel_coords_dict_source)
        ds_target = self.apply_spatial_dim_indices(ds_target, self.dims_variables_target, spatial_dim_indices_target, rel_coords_dict=rel_coords_dict_target)

        if len(self.save_sample_path)>0:
            
            save_path_source = os.path.join(self.save_sample_path, os.path.basename(file_path_source).replace('.nc', f'_{float(seeds[0]):.3f}_{float(seeds[1]):.3f}_source.nc'))
            ds_source.to_netcdf(save_path_source)

            save_path_target = os.path.join(self.save_sample_path, os.path.basename(file_path_target).replace('.nc', f'_{float(seeds[0]):.3f}_{float(seeds[1]):.3f}_target.nc'))
            ds_target.to_netcdf(save_path_target)

        return ds_source, ds_target, seeds


    def get_data(self, ds, index, dims_variables_dict, seeds):

        data = {}
        coords = {}
        for spatial_dim, vars in dims_variables_dict['spatial_dims_var'].items():
            coord_dict = dims_variables_dict['coord_dicts'][spatial_dim]

            coords[spatial_dim] = get_coords_as_tensor(ds, lon=coord_dict['lon'], lat=coord_dict['lat']).float()

            for var in vars:
                data_var = torch.tensor(ds[var][index].values).squeeze()
                data[var] = data_var.unsqueeze(dim=-1)

     #       if len(seeds)>0 and self.rotate_cs and 'u' in vars:
     #           u_rad, v_rad = data['u']/(6371000), data['v']/6371000
     #           mag = (u_rad**2+v_rad**2).sqrt()
     #          u_rad, v_rad = u_rad/mag,v_rad/mag
     #           u_rad, v_rad = rotate_coord_system(u_rad, v_rad, seeds[0].deg2rad(),seeds[1].deg2rad())
     #           data['u'], data['v'] = mag*u_rad*6371000, mag*v_rad*6371000

        return data, coords


    def __getitem__(self, index):
        
        if self.index_range_source is not None:
            if (index < self.index_range_source[0]) or (index > self.index_range_source[1]):
                index = int(torch.randint(self.index_range_source[0], self.index_range_source[1]+1, (1,1)))

        index_target = index + self.index_offset_target

        if len(self.files_source)>0:
            source_index = torch.randint(0, len(self.files_source), (1,1))
            source_file = self.files_source[source_index]

        if self.sampling_mode=='mixed':
            
            if len(self.files_target)>0:
                target_file = self.files_target[torch.randint(0, len(self.files_target), (1,1))]
            else:
                if len(self.files_source)>0:
                    target_file = self.files_source[torch.randint(0, len(self.files_source)-1, (1,1))]

        elif self.sampling_mode=='self':
            target_file = None
        
        elif self.sampling_mode=='paired':
            target_file = self.files_target[source_index]

        ds_source, ds_target, seeds = self.get_files(source_file, file_path_target=target_file)

        data_source, rel_coords_source = self.get_data(ds_source, index, self.dims_variables_source, seeds)
        data_target, rel_coords_target = self.get_data(ds_target, index_target, self.dims_variables_target, seeds)

        if self.normalize_data:
            data_source = self.normalizer(data_source)
            data_target = self.normalizer(data_target)

        if self.apply_img_norm:  
            stats = [calc_stats(data) for data in data_source.values()]
            stat_dict = dict(zip(self.variables_source,stats))
            norm = normalizer(stat_dict)
            data_source = norm(data_source)
            data_target = norm(data_target)

        return data_source, data_target, rel_coords_source, rel_coords_target

    def __len__(self):
        return self.num_datapoints_time
