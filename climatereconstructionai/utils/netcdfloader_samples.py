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
from .grid_utils import generate_region, PositionCalculator, get_coord_dict_from_var, get_coords_as_tensor, invert_dict
from climatereconstructionai.model.transformer_helpers import unstructured_to_reg_interpolator
from .normalizer import normalizer

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

def get_moments(data, type, level=0.9):

    if type == 'quantile':
        moments = (np.quantile((data), ((1-level), level)).astype(float))

    elif type == 'quantile_abs':
        q = np.quantile(np.abs(data), level).astype(float)
        moments = (q,q)

    elif type == 'min_max':
        moments = (data.min().astype(float), data.max().astype(float))
    else:
        moments = (data.mean().astype(float), data.std().astype(float))
    
    return tuple(moments)



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

def get_stats(files, variable, norm_dict, n_sample=None):
    if (n_sample is not None) and (n_sample < len(files)):
        file_indices = np.random.randint(0,len(files), (n_sample,))
    else:
        file_indices = np.arange(len(files))
    
    if variable == "uv":
        variables = ["u", "v"]
    else:
        variables = [variable]
    
    data = []
    for file_index in np.array(files)[file_indices]:
        ds = xr.load_dataset(file_index)
        data += [ds[variable].values.flatten() for variable in variables]

    return get_moments(np.concatenate(data), norm_dict["type"], level=norm_dict["level"])


class NetCDFLoader_lazy(Dataset):
    def __init__(self, files_source, 
                 files_target, 
                 variables_source, 
                 variables_target,
                 normalization,
                 random_region=None, 
                 p_dropout_source=0,
                 p_dropout_target=0,
                 sampling_mode='mixed', 
                 n_points_s=None, 
                 n_points_t=None, 
                 coordinate_pert=0,
                 save_nc_sample_path='',
                 save_tensor_sample_path='',
                 index_range_source=None,
                 index_offset_target=0,
                 rel_coords=False,
                 sample_for_norm=-1,
                 lazy_load=False,
                 rotate_cs=False,
                 interpolation_size_s=None,
                 interpolation_size_t=None,
                 range_target=None,
                 interpolation_method='nearest'):
        
        super(NetCDFLoader_lazy, self).__init__()
        
        self.PosCalc = PositionCalculator()
        self.variables_source = variables_source
        self.variables_target = variables_target
        self.normalization = normalization
        self.p_dropout_source = p_dropout_source
        self.p_dropout_target = p_dropout_target
        self.sampling_mode = sampling_mode
        self.random_region=random_region
        self.n_points_s = n_points_s
        self.n_points_t = n_points_t
        self.coordinate_pert = coordinate_pert
        self.save_nc_sample_path = save_nc_sample_path
        self.save_tensor_sample_path = save_tensor_sample_path
        self.index_range_source=index_range_source
        self.index_offset_target = index_offset_target
        self.rel_coords=rel_coords
        self.sample_for_norm = sample_for_norm
        self.lazy_load=lazy_load
        self.rotate_cs = rotate_cs
        self.interpolation_size_s = interpolation_size_s
        self.interpolation_size_t = interpolation_size_t
        self.range_target = range_target
        self.interpolation_method = interpolation_method

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

    
        for var, norm_dict in normalization.items():
            if len(norm_dict["moments"])==0:
                var_check = "u" if var == "uv" else var
                files = []
                if var_check in self.variables_source:
                    files+=list(files_source)
                if var_check in self.variables_target:
                    files+=list(files_target)

                norm_dict["moments"] = get_stats(files, var, norm_dict, self.sample_for_norm)
        
        self.norm_dict = normalization
        self.normalizer = normalizer(self.norm_dict)

        if self.interpolation_size_s is not None:
            self.input_mapper_s = unstructured_to_reg_interpolator(
                self.interpolation_size_s,
                range_target,
                method=interpolation_method
                )
        
        if self.interpolation_size_t is not None:
            self.input_mapper_t = unstructured_to_reg_interpolator(
                self.interpolation_size_t,
                range_target,
                method=interpolation_method
                )


    def get_coordinates(self, ds, dims_variables_dict, seeds=[], n_drop_dict={}, p_drop=0, rotate_cs=False):

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
                                              rect=rect,
                                              return_rotated_coords=rotate_cs)
                
                coords = torch.stack([region_dict['lon'],region_dict['lat']], dim=0)
                seeds = region_dict['locations']
                global_indices = region_dict['indices'].squeeze()
                
            else:
                global_indices = torch.arange(coords.shape[1])

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
            global_indices = global_indices[indices]

            #if not self.rel_coords:
            #    coords = {'lon': coords[0], 'lat': coords[1]}

            #    if len(seeds)==[]:
            #        rel_coords, _  = self.get_rel_coords(coords, [coords[0].median(), coords[1].median()], rotated_cs=rotate_cs)
            #    else:
            #        rel_coords, _  = self.get_rel_coords(coords, [seeds[0].deg2rad(), seeds[1].deg2rad()], rotated_cs=rotate_cs)  

            #else:
            #    rel_coords = coords

            spatial_dim_indices[spatial_dim] = global_indices
            rel_coords_dict[spatial_dim] = coords.squeeze()

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
    

    def get_rel_coords(self, coords, seeds, rotated_cs=False):
        
        if rotated_cs:
            seeds = (torch.tensor([0.]).view(-1,1),torch.tensor([0.]).view(-1,1))
        
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

        spatial_dim_indices_source, rel_coords_dict_source, seeds, _ = self.get_coordinates(ds_source, self.dims_variables_source, n_drop_dict=self.n_dict_source, rotate_cs=self.rotate_cs)
        spatial_dim_indices_target, rel_coords_dict_target, _, _ = self.get_coordinates(ds_target, self.dims_variables_target, seeds=seeds, n_drop_dict=self.n_dict_target, rotate_cs=self.rotate_cs)

        ds_source = self.apply_spatial_dim_indices(ds_source, self.dims_variables_source, spatial_dim_indices_source, rel_coords_dict=rel_coords_dict_source)
        ds_target = self.apply_spatial_dim_indices(ds_target, self.dims_variables_target, spatial_dim_indices_target, rel_coords_dict=rel_coords_dict_target)

        if len(self.save_nc_sample_path)>0:
            
            save_path_source = os.path.join(self.save_nc_sample_path, os.path.basename(file_path_source).replace('.nc', f'_{float(seeds[0]):.3f}_{float(seeds[1]):.3f}_source.nc'))
            ds_source.to_netcdf(save_path_source)

            save_path_target = os.path.join(self.save_nc_sample_path, os.path.basename(file_path_target).replace('.nc', f'_{float(seeds[0]):.3f}_{float(seeds[1]):.3f}_target.nc'))
            ds_target.to_netcdf(save_path_target)

        return ds_source, ds_target, seeds, spatial_dim_indices_target


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

        ds_source, ds_target, seeds, spatial_dim_indices = self.get_files(source_file, file_path_target=target_file)

        data_source, rel_coords_source = self.get_data(ds_source, index, self.dims_variables_source, seeds)
        data_target, rel_coords_target = self.get_data(ds_target, index_target, self.dims_variables_target, seeds)

        if self.normalization is not None:
            data_source = self.normalizer(data_source)
            data_target = self.normalizer(data_target)

        if self.interpolation_size_s is not None:
            data_source = self.input_mapper_s(data_source, rel_coords_source, self.dims_variables_source['spatial_dims_var'])
            rel_coords_source = dict(zip(rel_coords_source.keys(),[torch.empty(0) for _ in rel_coords_source.values()]))

        if self.interpolation_size_t is not None:
            data_target = self.input_mapper_t(data_target, rel_coords_target, self.dims_variables_target['spatial_dims_var'])
            rel_coords_target = dict(zip(rel_coords_target.keys(),[torch.empty(0) for _ in rel_coords_target.values()]))

        if len(self.save_tensor_sample_path)>0:
            save_path = os.path.join(self.save_tensor_sample_path, os.path.basename(source_file).replace('.nc', f'_{float(seeds[0]):.3f}_{float(seeds[1]):.3f}.pt'))
            torch.save([data_source, data_target, rel_coords_source, rel_coords_target, spatial_dim_indices], save_path)

            dict_file = os.path.join(self.save_tensor_sample_path,'dims_var_source.json')
            
            if not os.path.isfile(dict_file):
                with open(os.path.join(self.save_tensor_sample_path,'dims_var_source.json'), 'w') as f:
                    json.dump(self.dims_variables_source, f, indent=4)

                with open(os.path.join(self.save_tensor_sample_path,'dims_var_target.json'), 'w') as f:
                    json.dump(self.dims_variables_target, f, indent=4)

        return data_source, data_target, rel_coords_source, rel_coords_target, spatial_dim_indices

    def __len__(self):
        return self.num_datapoints_time


class SampleLoader(Dataset):
    def __init__(self, root_dir, dims_variables_source, dims_variables_target):
        self.root_dir = root_dir
        self.file_list = os.listdir(root_dir)
        sample_path = os.path.join(self.root_dir, self.file_list[0])
        _, _, coords_source, coords_target, target_indices = torch.load(sample_path, map_location='cpu')

        self.n_dict_source = dict(zip(coords_source.keys(),[val.shape[-1] for val in coords_source.values()]))
        self.n_dict_target = dict(zip(coords_target.keys(),[val.shape[-1] for val in coords_target.values()]))
        
        self.dims_variables_source = dims_variables_source
        self.dims_variables_target = dims_variables_target
   

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        valid_file = False

        while not valid_file:
            path = os.path.join(self.root_dir, self.file_list[idx])
            if os.path.isfile(path):
                try:
                    data = torch.load(path, map_location='cpu')
                    valid_file=True
                except:
                    idx = torch.randint(0,len(self.file_list), size=(1,))

        source, target, coords_source, coords_target, target_indices = data

        n_dict_source_sample = dict(zip(coords_source.keys(),[val.shape[-1] for val in coords_source.values()]))
        n_dict_target_sample = dict(zip(coords_target.keys(),[val.shape[-1] for val in coords_target.values()]))

        for spatial_dim, n_pts_sample in n_dict_target_sample.items():
            n_pts = self.n_dict_target[spatial_dim]

            if n_pts_sample > n_pts:
                indices = torch.randperm(n_pts)

            elif n_pts_sample < n_pts:
                indices = torch.randperm(n_pts_sample-1)[:(n_pts - n_pts_sample)]
                indices = torch.concat((torch.arange(n_pts_sample), indices))
            
            else:
                indices = torch.arange(n_pts_sample)

            for var in self.dims_variables_target['spatial_dims_var'][spatial_dim]:
                target[var] = target[var][indices]

            target_indices[spatial_dim] = target_indices[spatial_dim][indices]
            coords_target[spatial_dim] = coords_target[spatial_dim][:,indices]

        return data