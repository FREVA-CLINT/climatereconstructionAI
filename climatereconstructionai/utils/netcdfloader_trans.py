import os
import random
import copy
import json
from typing import Any

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset, Sampler

from .netcdfchecker import dataset_formatter
from .grid_utils import generate_region, PositionCalculator

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

def get_moment(stat_dict, variables, stat_var):
    return torch.tensor([stat_dict[var][stat_var] for var in variables]).view(1,len(variables))

class normalizer(torch.nn.Module):
    def __init__(self, stat_dict, variables, type="quantile"):
        super().__init__()
        
        if type == 'quantile':
            self.norm_fcn = norm_min_max
            self.moments = (get_moment(stat_dict, variables, "q_05"),
                            get_moment(stat_dict, variables, "q_95"))
        elif type == 'min_max':
            self.norm_fcn = norm_min_max
            self.moments = (get_moment(stat_dict, variables, "min"),
                            get_moment(stat_dict, variables, "max"))
        else:
            self.norm_fcn = norm_mean_std
            self.moments = (get_moment(stat_dict, variables, "mean"),
                            get_moment(stat_dict, variables, "std"))
    
    def __call__(self, data):
        return self.norm_fcn(data, self.moments)
        

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


def prepare_coordinates(ds_dict, coord_names, flatten=False, random_region=None):

    for tag, entry in ds_dict.items():
        ds = entry['ds']

        lon = torch.tensor(ds[coord_names[0]].values) 
        lat = torch.tensor(ds[coord_names[1]].values) 
        
        lon = lon.deg2rad() if lon.max()>2*torch.pi else lon 
        lat = lat.deg2rad() if lat.max()>2*torch.pi else lat
        
        len_lon = len(lon) 
        len_lat = len(lat) 

        if flatten:
            lon = lon.flatten().repeat(len_lat) 
            lat = lat.view(-1,1).repeat(1,len_lon).flatten()
        
        ds_dict[tag]['lon'] = lon
        ds_dict[tag]['lat'] = lat

    return ds_dict



class NetCDFLoader(Dataset):
    def __init__(self, img_names_source, img_names_target, data_types, coord_names, apply_img_norm=False, normalize_data=True, random_region=None, stat_dict=None, p_input_dropout=0, sampling_mode='mixed',n_points=None,coordinate_pert=0):
        super(NetCDFLoader, self).__init__()
        
        self.PosCalc = PositionCalculator()
        self.random = random.Random()
        self.data_types = data_types
        self.coord_names = coord_names
        self.apply_img_norm = apply_img_norm
        self.normalize_data = normalize_data
        self.p_input_dropout = p_input_dropout
        self.sampling_mode = sampling_mode
        self.random_region=random_region
        self.n_points = n_points
        self.coordinate_pert = coordinate_pert

        if 'lon' in self.coord_names:
            self.flatten=True
        else:
            self.flatten=False
        
        if random_region is not None:
            self.generate_region_prob = 1/random_region['generate_interval']

        self.ds_dict = {}
        file_tags_train = []
        for img_name_source in img_names_source:
            if img_name_source not in self.ds_dict.keys():
                file_tag = os.path.basename(img_name_source)
                file_tags_train.append(file_tag)
                self.ds_dict[file_tag] = {'ds': xr.load_dataset(img_name_source)}

                
        file_tags_target = []
        if len(img_names_target) > 0:
            for img_name_target in img_names_target:
                file_tag = os.path.basename(img_name_target)
                file_tags_target.append(file_tag)
                if file_tag not in self.ds_dict.keys():
                    self.ds_dict[file_tag] = {'ds': xr.load_dataset(img_name_target)}

        self.num_files_source = len(img_names_source)
        self.num_files_target = len(img_names_target)

        self.ds_dict = prepare_coordinates(self.ds_dict, coord_names=coord_names, flatten=self.flatten)            

        self.target_names = file_tags_target
        self.source_names = file_tags_train

        self.num_datapoints = self.ds_dict[file_tags_train[0]]['ds'][data_types[0]].shape[0]        
           
        self.update_coordinates(file_tags_train+file_tags_target)

        self.n_source_dropout = int((1-p_input_dropout) * self.ds_dict[file_tags_train[0]]['rel_coords'].shape[1])

        if stat_dict is None:
            self.stat_dict = {}
            for var in data_types:
                data = np.concatenate([ds_d['ds'][var].values.flatten() for ds_d in self.ds_dict.values()])
                self.stat_dict[var] = calc_stats(data)
            
            with open(os.path.join(os.path.dirname(img_names_source[0]),"norm_stats.json"),"w+") as f:
                json.dump(self.stat_dict,f)

        else:
            self.stat_dict=stat_dict

        self.normalizer = normalizer(self.stat_dict, data_types)


    def input_dropout(self, x, coords):

        coords = copy.deepcopy(coords)

        indices = torch.randperm(x.shape[0]-1)[:self.n_source_dropout]

        x = x[indices,:]

        coords = coords[:,indices,:]
 
        return x, coords



    def update_coordinates(self, tags):
        
        for k, tag in enumerate(tags):

            if self.random_region is not None:
                if k==0:
                    seeds = []

                radius = self.random_region["radius"] if "radius" in self.random_region.keys() else None
                n_points = self.random_region["n_points"] if "n_points" in self.random_region.keys() else None
                region_dict = generate_region(self.ds_dict[tag], self.random_region['lon_range'], self.random_region['lat_range'], n_points=n_points, radius=radius, batch_size=self.random_region['batch_size'],locations=seeds)

                seeds = region_dict['locations']
                seeds = [seeds[0].rad2deg(), seeds[1].rad2deg()]
                self.ds_dict[tag]['region_indices'] = region_dict['indices']
                self.ds_dict[tag]['seeds'] = region_dict['locations']

                rel_coords, distances = self.get_rel_coords(region_dict, region_dict['locations'])

                self.ds_dict[tag]['rel_coords'] = rel_coords
                self.ds_dict[tag]['distances'] = distances

            else:
                if 'rel_coords' not in self.ds_dict[tag].keys():
                    if k== 0:
                        seeds = [torch.tensor([self.ds_dict[tag]['lon'].median()]), torch.tensor([self.ds_dict[tag]['lat'].median()])]   
                    self.ds_dict[tag]['seeds'] = seeds

                    rel_coords, distances  = self.get_rel_coords(self.ds_dict[tag], seeds)

                    self.ds_dict[tag]['rel_coords'] = rel_coords
                    self.ds_dict[tag]['distances'] = distances

             

    def get_rel_coords(self, coords, seeds):
    
        distances, _, d_lons_s, d_lats_s = self.PosCalc(coords['lon'], coords['lat'], (seeds[0]), (seeds[1]))
         
        return torch.stack([d_lons_s.float().T, d_lats_s.float().T],dim=0), distances
    

    def get_data(self, key, index):
        data = [torch.tensor(self.ds_dict[key]['ds'][data_type][index].values).squeeze() for data_type in self.data_types]
        data = torch.stack(data, dim=-1)

        if self.flatten:
            data = data.flatten().unsqueeze(dim=1)
            data = data.flatten().unsqueeze(dim=1)
        else:
            data = data.view(-1,len(self.data_types))

        if self.random_region is not None:
            indices = self.ds_dict[key]['region_indices']
            data = data[indices[:,0]]

        rel_coords = self.ds_dict[key]['rel_coords']

        if self.n_points is not None:

            if data.shape[0] > self.n_points:
                data = data[:self.n_points]
                rel_coords = rel_coords[:,:self.n_points]

            elif data.shape[0] < self.n_points:
                diff = self.n_points-data.shape[0]
                pad_data = torch.zeros_like(data)[:diff] + torch.mean(data)
                data = torch.concat((data,pad_data),dim=0)

                pad_rel_coords = torch.zeros_like(rel_coords)[:,:diff] + 1000
                rel_coords = torch.concat((rel_coords, pad_rel_coords),dim=1)

        if self.coordinate_pert>0:
            avg_dist = rel_coords.max()/torch.tensor(rel_coords.shape[1]).sqrt()

            pertubation = torch.randn_like(rel_coords)*self.coordinate_pert*avg_dist
            rel_coords = rel_coords+pertubation
        return data, rel_coords


    def __getitem__(self, index):
        
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

        data_source, rel_coords_source = self.get_data(source_key, index)

        if self.sampling_mode=='self':
            data_target = data_source
            rel_coords_target = rel_coords_source
        else:
            data_target, rel_coords_target = self.get_data(target_key, index)

        if self.normalize_data:
            data_source = self.normalizer(data_source)[0]
            data_target = self.normalizer(data_target)[0]
            
        #if self.apply_img_norm:
        #    norm = ds_norm()
        #    data_source = norm(data_source)
        #    data_target = norm(data_target)

        if self.p_input_dropout > 0:
            data_source, rel_coords_source = self.input_dropout(data_source, rel_coords_source)
   
        coord_dict ={'rel': {'source': rel_coords_source,
                      'target': rel_coords_target}}

        return data_source, data_target, coord_dict

    def __len__(self):
        return self.num_datapoints
