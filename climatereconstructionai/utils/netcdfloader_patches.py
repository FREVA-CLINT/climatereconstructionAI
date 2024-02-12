import os
import random
import copy
import json
from typing import Any
import datetime
import math
import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset, Sampler
from .grid_utils import generate_region, get_coord_dict_from_var, get_coords_as_tensor, invert_dict, get_ids_in_patches, get_patches
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

    elif type == 'None':
        moments = (1., 1.)

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


def get_dims_coordinates(ds, variables):

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
                 grid_spacing_equator_km,
                 pix_size_patch,
                 patches_overlap_source,
                 p_dropout_source=0,
                 p_dropout_target=0,
                 n_pts_min = True,
                 save_nc_sample_path='',
                 save_tensor_sample_path='',
                 index_range_source=None,
                 index_offset_target=0,
                 rel_coords=False,
                 sample_for_norm=-1,
                 lazy_load=False,
                 rotate_cs=False,
                 interpolation_dict=None,
                 sample_patch_range_lat=[-math.pi/2,math.pi/2]):
        
        super(NetCDFLoader_lazy, self).__init__()
        
        self.variables_source = variables_source
        self.variables_target = variables_target
        self.normalization = normalization
        self.p_dropout_source = p_dropout_source
        self.p_dropout_target = p_dropout_target
        self.save_nc_sample_path = save_nc_sample_path
        self.save_tensor_sample_path = save_tensor_sample_path
        self.index_range_source=index_range_source
        self.index_offset_target = index_offset_target
        self.rel_coords=rel_coords
        self.sample_for_norm = sample_for_norm
        self.lazy_load=lazy_load
        self.rotate_cs = rotate_cs
        self.sample_patch_range_lat = sample_patch_range_lat

        self.patches_source = get_patches(grid_spacing_equator_km, pix_size_patch, patches_overlap_source)
        self.patches_target = get_patches(grid_spacing_equator_km, pix_size_patch, 0)
        
        if interpolation_dict is not None:
                 
            if interpolation_dict is not None:

                self.interpolate_source = interpolation_dict['interpolate_source']
                self.interpolate_target = interpolation_dict['interpolate_target']

                if interpolation_dict['interpolate_target']:
                    self.input_mapper_t = unstructured_to_reg_interpolator(
                        interpolation_dict['interpolation_size_t'],
                        interpolation_dict['range_target_x'],
                        interpolation_dict['range_target_y'],
                        method=interpolation_dict['interpolation_method']
                        )
                    
                if interpolation_dict['interpolate_source']:
                    self.input_mapper_s = unstructured_to_reg_interpolator(
                        interpolation_dict['interpolation_size_s'],
                        [0,1],
                        [0,1],
                        method=interpolation_dict['interpolation_method']
                        )
 
        self.flatten=False

        self.ds_dict = {}
 
        self.files_source = files_source
        self.files_target = files_target

        ds_source = xr.open_dataset(files_source[0])
        ds_target = xr.open_dataset(files_target[0])

        self.dims_variables_source = get_dims_coordinates(ds_source, self.variables_source)    
        self.dims_variables_target = get_dims_coordinates(ds_target, self.variables_target)  

        self.num_datapoints_time = ds_source[self.variables_source[0]].shape[0]

        self.spatial_dims_patches_ids_source, spatial_dims_n_pts_source, _ = self.get_ids_patches(ds_source, self.dims_variables_source, self.patches_source)
        self.spatial_dims_patches_ids_target, spatial_dims_n_pts_target, self.patch_ids = self.get_ids_patches(ds_target, self.dims_variables_target, self.patches_target)

        if n_pts_min:
            self.n_dict_source = dict(zip(spatial_dims_n_pts_source.keys(), [int(n_pts.min()*(1-p_dropout_source)) for n_pts in spatial_dims_n_pts_source.values()]))
            self.n_dict_target = dict(zip(spatial_dims_n_pts_target.keys(), [int(n_pts.min()*(1-p_dropout_target)) for n_pts in spatial_dims_n_pts_target.values()]))
        else:
            self.n_dict_source = dict(zip(spatial_dims_n_pts_source.keys(), [int(n_pts.median()*(1-p_dropout_source)) for n_pts in spatial_dims_n_pts_source.values()]))
            self.n_dict_target = dict(zip(spatial_dims_n_pts_target.keys(), [int(n_pts.median()*(1-p_dropout_target)) for n_pts in spatial_dims_n_pts_target.values()]))
    
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


    def get_ids_patches(self, ds, dims_variables_dict, patches):

        spatial_dims_patches = {}
        spatial_dims_n_pts = {}
        for spatial_dim, coord_dict in dims_variables_dict['coord_dicts'].items():
            
            coords = get_coords_as_tensor(ds, lon=coord_dict['lon'], lat=coord_dict['lat'])

            ids_in_patches, patch_ids = get_ids_in_patches(patches, coords.numpy())
            spatial_dims_patches[spatial_dim] = ids_in_patches
            spatial_dims_n_pts[spatial_dim] = torch.tensor([len(ids_in_patch) for ids_in_patch in ids_in_patches])

        return spatial_dims_patches, spatial_dims_n_pts, patch_ids


    def get_coordinates(self, ds, dims_variables_dict, spatial_dims_patches_ids, n_dict, patches, patch_id=None):

        spatial_dim_indices = {}
        rel_coords_dict = {} 

        if patch_id is None:
            in_lat_range = False
            while not in_lat_range:
                patch_id = torch.randint(0, len(patches['centers_lon'])* len(patches['centers_lat']) -1, (1,))
                center_lat = patches["centers_lat"][self.patch_ids["lat"][int(patch_id)]]

                if self.sample_patch_range_lat[0]<center_lat and self.sample_patch_range_lat[1]>center_lat:
                    in_lat_range = True

        for spatial_dim, coord_dict in dims_variables_dict['coord_dicts'].items():
            
            coords = get_coords_as_tensor(ds, lon=coord_dict['lon'], lat=coord_dict['lat'])

            indices = spatial_dims_patches_ids[spatial_dim][patch_id]
            
            n_pts = n_dict[spatial_dim]

            if n_pts > len(indices):
                pad_indices = torch.randint(len(indices), size=(n_pts - len(indices),1)).view(-1)
                indices = torch.concat((indices, indices[pad_indices]))
            else:    
                indices = indices[torch.randperm(len(indices-1))[:n_pts]]

            coords = coords[:,indices]

            patch_borders_lon = patches["borders_lon"][self.patch_ids["lon"][int(patch_id)]]
            patch_borders_lat = patches["borders_lat"][self.patch_ids["lat"][int(patch_id)]]

            spatial_dim_indices[spatial_dim] = indices

            if patch_borders_lon[0] < -math.pi:
                shift_indices = coords[0,:] > 2*math.pi + patch_borders_lon[0]
                coords[0,shift_indices] =  coords[0, shift_indices] - 2*math.pi

            elif patch_borders_lon[1] > math.pi:
                shift_indices = coords[0,:] < patch_borders_lon[1] - 2*math.pi
                coords[0,shift_indices] =  coords[0,shift_indices] + 2*math.pi

            rel_coords_lon = (coords[0,:] - patch_borders_lon[0])/(patch_borders_lon[1]-patch_borders_lon[0])
            rel_coords_lat = (coords[1,:] - patch_borders_lat[0])/(patch_borders_lat[1]-patch_borders_lat[0])
            rel_coords_dict[spatial_dim] = torch.stack((rel_coords_lon,rel_coords_lat),dim=0)
            

        return spatial_dim_indices, rel_coords_dict, patch_id


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


        spatial_dim_indices_source, rel_coords_dict_source, patch_id = self.get_coordinates(ds_source, self.dims_variables_source, self.spatial_dims_patches_ids_source, self.n_dict_source, self.patches_source)
        spatial_dim_indices_target, rel_coords_dict_target, _ = self.get_coordinates(ds_target, self.dims_variables_target, self.spatial_dims_patches_ids_target, self.n_dict_target, self.patches_target, patch_id=patch_id)

        ds_source = self.apply_spatial_dim_indices(ds_source, self.dims_variables_source, spatial_dim_indices_source, rel_coords_dict=rel_coords_dict_source)
        ds_target = self.apply_spatial_dim_indices(ds_target, self.dims_variables_target, spatial_dim_indices_target, rel_coords_dict=rel_coords_dict_target)

        if len(self.save_nc_sample_path)>0:
            
            save_path_source = os.path.join(self.save_nc_sample_path, os.path.basename(file_path_source).replace('.nc', f'_{float(patch_id):.3f}_source.nc'))
            ds_source.to_netcdf(save_path_source)

            save_path_target = os.path.join(self.save_nc_sample_path, os.path.basename(file_path_target).replace('.nc', f'_{float(patch_id):.3f}_target.nc'))
            ds_target.to_netcdf(save_path_target)

        return ds_source, ds_target, patch_id, spatial_dim_indices_source, spatial_dim_indices_target


    def get_data(self, ds, index, dims_variables_dict):

        data = {}
        coords = {}
        for spatial_dim, vars in dims_variables_dict['spatial_dims_var'].items():
            coord_dict = dims_variables_dict['coord_dicts'][spatial_dim]

            coords[spatial_dim] = get_coords_as_tensor(ds, lon=coord_dict['lon'], lat=coord_dict['lat']).float()

            for var in vars:
                data_var = torch.tensor(ds[var][index].values).squeeze().float()
                data[var] = data_var.unsqueeze(dim=-1)

        return data, coords


    def __getitem__(self, index):
        
        if self.index_range_source is not None:
            if (index < self.index_range_source[0]) or (index > self.index_range_source[1]):
                index = int(torch.randint(self.index_range_source[0], self.index_range_source[1]+1, (1,1)))

        index_target = index + self.index_offset_target

        if len(self.files_source)>0:
            source_index = torch.randint(0, len(self.files_source), (1,1))
            source_file = self.files_source[source_index]

        target_file = self.files_target[source_index]

        ds_source, ds_target, patch_id, spatial_dim_indices_source, spatial_dim_indices_target = self.get_files(source_file, file_path_target=target_file)

        data_source, rel_coords_source = self.get_data(ds_source, index, self.dims_variables_source)
        data_target, rel_coords_target = self.get_data(ds_target, index_target, self.dims_variables_target)

        if self.normalization is not None:
            data_source = self.normalizer(data_source)
            data_target = self.normalizer(data_target)

        if self.interpolate_source:
            data_source = self.input_mapper_s(data_source, rel_coords_source, self.dims_variables_source['spatial_dims_var'])
            rel_coords_source = dict(zip(rel_coords_source.keys(),[torch.empty(0) for _ in rel_coords_source.values()]))

        if self.interpolate_target:
            data_target = self.input_mapper_t(data_target, rel_coords_target, self.dims_variables_target['spatial_dims_var'])
            rel_coords_target = dict(zip(rel_coords_target.keys(),[torch.empty(0) for _ in rel_coords_target.values()]))

        if len(self.save_tensor_sample_path)>0:
            save_path = os.path.join(self.save_tensor_sample_path, os.path.basename(source_file).replace('.nc', f'_{float(patch_id):.3f}.pt'))
            if not os.path.isfile(save_path):
                torch.save([data_source, data_target, rel_coords_source, rel_coords_target, spatial_dim_indices_source, spatial_dim_indices_target], save_path)

            dict_file = os.path.join(self.save_tensor_sample_path,'dims_var_source.json')
            
            if not os.path.isfile(dict_file):
                with open(os.path.join(self.save_tensor_sample_path,'dims_var_source.json'), 'w') as f:
                    json.dump(self.dims_variables_source, f, indent=4)

                with open(os.path.join(self.save_tensor_sample_path,'dims_var_target.json'), 'w') as f:
                    json.dump(self.dims_variables_target, f, indent=4)

        return data_source, data_target, rel_coords_source, rel_coords_target, spatial_dim_indices_source, spatial_dim_indices_target

    def __len__(self):
        return self.num_datapoints_time


class SampleLoader(Dataset):
    def __init__(self, root_dir, dims_variables_source, dims_variables_target, variables_source, variables_target):
        self.root_dir = root_dir
        self.file_list = os.listdir(root_dir)
        sample_path = os.path.join(self.root_dir, self.file_list[0])
        coords_source, coords_target = torch.load(sample_path, map_location='cpu')[2:4]
        
        self.n_dict_source = dict(zip(coords_source.keys(),[val.shape[-1] for val in coords_source.values()]))
        self.n_dict_target = dict(zip(coords_target.keys(),[val.shape[-1] for val in coords_target.values()]))
        
        sample_variables_source = list(dims_variables_source['var_spatial_dims'].keys())
        self.indices_source = [k for k,var in enumerate(sample_variables_source) if var in variables_source]

        self.dims_variables_source = dims_variables_source
        self.dims_variables_target = dims_variables_target
   

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        valid_file = False

        while not valid_file:
            idx = torch.randint(0,len(self.file_list), size=(1,))
            path = os.path.join(self.root_dir, self.file_list[idx])
            if os.path.isfile(path):
                try:
                    data = torch.load(path, map_location='cpu')
                    valid_file=True
                   # if len(data)<6:
                   #     idx = torch.randint(0,len(self.file_list), size=(1,))
                   #     valid_file=False
                except:
                    idx = torch.randint(0,len(self.file_list), size=(1,))

        source = data[0]
        target = data[1]
        coords_source = data[2]
        coords_target = data[3]
        target_indices = data[-1]

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

        data = [source[self.indices_source], target, torch.zeros((10,)), coords_target, target_indices]
        
        return data