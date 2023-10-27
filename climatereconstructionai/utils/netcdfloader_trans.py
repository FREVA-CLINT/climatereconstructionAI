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

from .netcdfchecker import dataset_formatter
from .grid_utils import generate_region, PositionCalculator, get_coord_dict_from_var

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


def prepare_coordinates(ds_dict, tags, variables, flatten=False, random_region=None):

    for tag in tags:
        ds = ds_dict[tag]['ds']

        ds_dict[tag]['spatial_dims'] = {}
        ds_dict[tag]['var_spatial_dims'] = {}

        spatial_dims = {}
        for variable in variables:
            
            coord_dict = get_coord_dict_from_var(ds, variable)

            lon = torch.tensor(ds[coord_dict['lon']].values) 
            lat = torch.tensor(ds[coord_dict['lat']].values) 
            
            lon = lon.deg2rad() if lon.max()>2*torch.pi else lon 
            lat = lat.deg2rad() if lat.max()>2*torch.pi else lat
            
            len_lon = len(lon) 
            len_lat = len(lat) 

            if flatten:
                lon = lon.flatten().repeat(len_lat) 
                lat = lat.view(-1,1).repeat(1,len_lon).flatten()
            
            spatial_coord_dict = {
                'coords':
                    {'lon': lon,
                    'lat': lat}
                        }
            
            ds_dict[tag]['spatial_dims'][coord_dict['spatial_dim']] = spatial_coord_dict
            spatial_dims[variable] = coord_dict['spatial_dim']

        ds_dict[tag]['var_spatial_dims'] = spatial_dims

    return ds_dict


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

def invert_dict(dict):
    dict_out = {}
    unique_values = np.unique(np.array(list(dict.values())))

    for uni_value in unique_values:
        dict_out[uni_value] = [key for key,value in dict.items() if value==uni_value]
    return dict_out

def get_stats(ds_dict, tags, variable, n_sample=None):
    if (n_sample is not None) and (n_sample < len(tags)):
        tag_indices = np.random.randint(0,len(tags), (n_sample,))
    else:
        tag_indices = np.arange(len(tags))
    
    data = np.concatenate([ds_dict[tag]['ds'][variable].values.flatten() for tag in np.array(tags)[tag_indices]])

    return calc_stats(data)


class NetCDFLoader(Dataset):
    def __init__(self, img_names_source, 
                 img_names_target, 
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
                 index_range=None,
                 rel_coords=False,
                 lazy_load=False,
                 sample_for_norm=-1,
                 norm_stats_save_path=''):
        
        super(NetCDFLoader, self).__init__()
        
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
        self.index_range=index_range
        self.rel_coords=rel_coords
        self.lazy_load=lazy_load
        self.sample_for_norm = sample_for_norm
        self.norm_stats_save_path = norm_stats_save_path

        self.flatten=False

        if random_region is not None:
            self.generate_region_prob = 1/random_region['generate_interval']

        self.ds_dict = {}
        file_tags_source = []
        for img_name_source in img_names_source:
            if img_name_source not in self.ds_dict.keys():
                file_tag = os.path.basename(img_name_source)
                file_tags_source.append(file_tag)
                if self.lazy_load:
                    ds = xr.open_dataset(img_name_source)
                else:
                    ds = xr.load_dataset(img_name_source)
                self.ds_dict[file_tag] = {'ds': ds}

                
        file_tags_target = []
        if len(img_names_target) > 0:
            for img_name_target in img_names_target:
                file_tag = os.path.basename(img_name_target)
                file_tags_target.append(file_tag)
                if file_tag not in self.ds_dict.keys():
                    if self.lazy_load:
                        ds = xr.open_dataset(img_name_target)
                    else:
                        ds = xr.load_dataset(img_name_target)
                    self.ds_dict[file_tag] = {'ds': ds}

        self.num_files_source = len(img_names_source)
        self.num_files_target = len(img_names_target)

        self.target_names = file_tags_target
        self.source_names = file_tags_source

        self.ds_dict = prepare_coordinates(self.ds_dict, file_tags_source, self.variables_source)    
        self.ds_dict = prepare_coordinates(self.ds_dict, file_tags_target, self.variables_target)         

        self.var_spatial_dims_source = self.ds_dict[file_tags_source[0]]['var_spatial_dims']
        self.var_spatial_dims_target = self.ds_dict[file_tags_target[0]]['var_spatial_dims']

        self.spatial_dims_var_source = invert_dict(self.var_spatial_dims_source)
        self.spatial_dims_var_target = invert_dict(self.var_spatial_dims_target)

        self.num_datapoints = self.ds_dict[file_tags_source[0]]['ds'][self.variables_source[0]].shape[0]        
           
        self.update_coordinates(file_tags_source + file_tags_target)

        if n_points_s is None:
            n_points_s = [coords['rel_coords'].shape[1] for coords in self.ds_dict[file_tags_source[0]]['spatial_dims'].values()]
            self.n_points_s = dict(zip(list(self.ds_dict[file_tags_source[0]]['spatial_dims'].keys()),n_points_s))

        if n_points_t is None:
            n_points_s = [coords['rel_coords'].shape[1] for coords in self.ds_dict[file_tags_target[0]]['spatial_dims'].values()]
            self.n_points_t = dict(zip(list(self.ds_dict[file_tags_target[0]]['spatial_dims'].keys()),n_points_s))
         
        n_dropout_source = [int((1-p_dropout_source) * n_points_s) for n_points_s in self.n_points_s.values()]
        self.n_dropout_source = dict(zip(list(self.n_points_s.keys()),n_dropout_source))
        
        n_dropout_target = [int((1-p_dropout_target) * n_points_t) for n_points_t in self.n_points_t.values()]
        self.n_dropout_target = dict(zip(list(self.n_points_t.keys()),n_dropout_target))


        if stat_dict is None:
            self.stat_dict = {}

            unique_variables = np.unique(np.array(self.variables_source + self.variables_target))
          
            for var in unique_variables:
                tags = []
                if var in self.variables_source:
                    tags+=file_tags_source
                if var in self.variables_target:
                    tags+=file_tags_target

                self.stat_dict[var] = get_stats(self.ds_dict, tags, var, self.sample_for_norm)
            
            with open(os.path.join(self.norm_stats_save_path,"norm_stats.json"),"w+") as f:
                json.dump(self.stat_dict,f, indent=4)

        else:
            self.stat_dict=stat_dict

        self.normalizer = normalizer(self.stat_dict)


    def input_dropout(self, x, coords, n_drop, spatial_dims_var_dict):
        
        coords_drop_indices = {}
        coords_drop = copy.deepcopy(coords)

        for spatial_dim, spatial_coords in coords_drop.items():
            
            indices = torch.randperm(spatial_coords.shape[1]-1)[:n_drop[spatial_dim]]
            coords_drop_indices[spatial_dim] = indices
            coords_drop[spatial_dim] = spatial_coords[:,indices,:]

        for spatial_dim, vars in spatial_dims_var_dict.items():
            for var in vars:
                x[var] = x[var][coords_drop_indices[spatial_dim]]
 
        return x, coords_drop


    def update_coordinates(self, tags):
        
        iteration=0
        for tag in tags:

            if self.random_region is not None:
                

                radius = self.random_region["radius"] if "radius" in self.random_region.keys() else None
                n_points = self.random_region["n_points"] if "n_points" in self.random_region.keys() else None
                rect = self.random_region["rect"] if "rect" in self.random_region.keys() else False

                for spatial_dim, spatial_dim_dict in self.ds_dict[tag]['spatial_dims'].items():

                    if iteration==0:
                        seeds = []

                    coord_dict = spatial_dim_dict['coords']
                    
                    region_dict = generate_region(coord_dict, self.random_region['lon_range'], self.random_region['lat_range'], n_points=n_points, radius=radius, batch_size=self.random_region['batch_size'],locations=seeds, rect=rect)

                    seeds = region_dict['locations']
                    seeds = [seeds[0].rad2deg(), seeds[1].rad2deg()]

                    self.ds_dict[tag]['spatial_dims'][spatial_dim]['region_indices'] = region_dict['indices']
                    self.ds_dict[tag]['seeds'] = region_dict['locations']

                    rel_coords, distances = self.get_rel_coords(region_dict, region_dict['locations'])

                    self.ds_dict[tag]['spatial_dims'][spatial_dim]['rel_coords'] = rel_coords
                    self.ds_dict[tag]['spatial_dims'][spatial_dim]['distances'] = distances

                    iteration+=1

            elif self.rel_coords:
                if 'rel_coords' not in self.ds_dict[tag].keys():
                    for spatial_dim, spatial_dim_dict in self.ds_dict[tag]['spatial_dims'].items():
                        coord_dict = spatial_dim_dict['coords']

                        rel_coords = torch.stack([coord_dict['lon'], coord_dict['lat']],dim=0).unsqueeze(dim=-1)

                        self.ds_dict[tag]['spatial_dims'][spatial_dim]['rel_coords'] = rel_coords
                        self.ds_dict[tag]['spatial_dims'][spatial_dim]['distances'] = (rel_coords[0]**2+rel_coords[1]**2).sqrt()

            else:
                if 'rel_coords' not in self.ds_dict[tag].keys():
                    if iteration==0:
                        seeds = [torch.tensor([self.ds_dict[tag]['lon'].median()]), torch.tensor([self.ds_dict[tag]['lat'].median()])] 

                    for spatial_dim, spatial_dim_dict in self.ds_dict[tag]['spatial_dims'].items():  
                        self.ds_dict[tag]['seeds'] = seeds

                        rel_coords, distances  = self.get_rel_coords(spatial_dim_dict, seeds)

                        self.ds_dict[tag]['spatial_dims'][spatial_dim]['rel_coords'] = rel_coords
                        self.ds_dict[tag]['spatial_dims'][spatial_dim]['distances'] = distances
                        iteration += 1
             

    def get_rel_coords(self, coords, seeds):
    
        distances, _, d_lons_s, d_lats_s = self.PosCalc(coords['lon'], coords['lat'], (seeds[0]), (seeds[1]))
         
        return torch.stack([d_lons_s.float().T, d_lats_s.float().T],dim=0), distances
    

    def get_data(self, key, index, n_points=None, origin='source'):
        if origin=='source':
            variables = self.variables_source
            spatial_dim_var_dict = self.spatial_dims_var_source
        else:
            variables = self.variables_target
            spatial_dim_var_dict = self.spatial_dims_var_target

        data = {}
        for var in variables:
            data_var = torch.tensor(self.ds_dict[key]['ds'][var][index].values).squeeze()

            if self.random_region is not None:
                spatial_dim = self.ds_dict[key]['var_spatial_dims'][var]
                indices = self.ds_dict[key]['spatial_dims'][spatial_dim]['region_indices']
                data_var = data_var[indices]
            else:
                data_var = data_var.unsqueeze(dim=-1)
            data[var] = data_var

        rel_coords = {}

        for spatial_dim, spatial_dim_dict in self.ds_dict[key]['spatial_dims'].items():
            rel_coords[spatial_dim] = spatial_dim_dict['rel_coords']

        if len(self.save_sample_path)>0:
            save_path = os.path.join(self.save_sample_path, key.replace('.nc', f'_{index}_{float(self.ds_dict[key]["seeds"][0]):.3f}_{float(self.ds_dict[key]["seeds"][1]):.3f}_{origin}.nc'))
            save_sample(self.ds_dict[key], index, spatial_dim_var_dict, save_path)

        if n_points is not None:
            for spatial_dim, n_pts in n_points.items():
                n_actual = rel_coords[spatial_dim].shape[1]

                if n_actual > n_pts:
                    rel_coords[spatial_dim] = rel_coords[spatial_dim][:,:n_pts]

                elif n_actual < n_pts:
                    diff = n_pts-n_actual
                    pad_indices = torch.randint(n_actual, size=(diff,1)).view(-1)

                    pad_rel_coords = rel_coords[spatial_dim][:,pad_indices,:]
                    rel_coords[spatial_dim] = torch.concat((rel_coords[spatial_dim], pad_rel_coords),dim=1)

                for var in spatial_dim_var_dict[spatial_dim]: 
                    data_var = data[var]       

                    if n_actual > n_pts:
                        data_var = data_var[:n_pts]

                    elif n_actual < n_pts:
                        pad_data = data_var[pad_indices]
                        data_var = torch.concat((data_var, pad_data),dim=0)

                    data[var] = data_var   

        if self.coordinate_pert>0:
            avg_dist = rel_coords.max()/torch.tensor(rel_coords.shape[1]).sqrt()

            pertubation = torch.randn_like(rel_coords)*self.coordinate_pert*avg_dist
            rel_coords = rel_coords+pertubation

        return data, rel_coords


    def __getitem__(self, index):
        
        if self.index_range is not None:
            if (index < self.index_range[0]) or (index > self.index_range[1]):
                index = int(torch.randint(self.index_range[0], self.index_range[1]+1, (1,1)))

        if len(self.source_names)>0:
            source_index = torch.randint(0, len(self.source_names), (1,1))
            source_key = self.source_names[source_index]

        if self.sampling_mode=='mixed':
            
            if len(self.target_names)>0:
                target_key = self.target_names[torch.randint(0, len(self.target_names), (1,1))]
            else:
                if len(self.source_names)>0:
                    target_key = self.source_names[torch.randint(0, len(self.source_names)-1, (1,1))]

        elif self.sampling_mode=='self':
            target_key = source_key
        
        elif self.sampling_mode=='paired':
            target_key = self.target_names[source_index]

        if self.random_region is not None:
            if torch.rand(1) < self.generate_region_prob:
                self.update_coordinates([source_key, target_key])

    
        data_source, rel_coords_source = self.get_data(source_key, index, self.n_points_s, 'source')

        if self.sampling_mode=='self':
            data_target = copy.deepcopy(data_source)
            rel_coords_target = copy.deepcopy(rel_coords_source)
            spatial_dims_var_target = self.spatial_dims_var_source
            n_dropout_target = self.n_dropout_source
        else:
            spatial_dims_var_target = self.spatial_dims_var_target
            data_target, rel_coords_target = self.get_data(target_key, index, self.n_points_t, 'target')
            n_dropout_target = self.n_dropout_target

        if self.normalize_data:
            data_source = self.normalizer(data_source)
            data_target = self.normalizer(data_target)
            
        #if self.apply_img_norm:
        #    norm = ds_norm()
        #    data_source = norm(data_source)
        #    data_target = norm(data_target)

        if self.p_dropout_source > 0:
            data_source, rel_coords_source = self.input_dropout(data_source, rel_coords_source, self.n_dropout_source, self.spatial_dims_var_source)
        
        if self.p_dropout_target > 0:
            data_target, rel_coords_target = self.input_dropout(data_target, rel_coords_target, n_dropout_target, spatial_dims_var_target)
   
        return data_source, data_target, rel_coords_source, rel_coords_target

    def __len__(self):
        return self.num_datapoints
