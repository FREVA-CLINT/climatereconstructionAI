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
from .grid_utils import generate_region, get_coord_dict_from_var, get_coords_as_tensor, invert_dict, get_ids_in_patches, get_patches, rotate_ds, get_mapping_to_icon_grid, get_nh_variable_mapping_icon
from climatereconstructionai.model.transformer_helpers import coarsen_global_cells
from .normalizer import grid_normalizer

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
                 coarsen_sample_level,
                 files_target_past=None,
                 index_range_source=None,
                 index_offset_target=0,
                 sample_for_norm=-1,
                 lazy_load=False,
                 sample_condition_dict={},
                 model_settings={}):
        
        super(NetCDFLoader_lazy, self).__init__()
        
        self.coarsen_sample_level = coarsen_sample_level
        self.files_target_past = files_target_past
        self.variables_source = variables_source
        self.variables_target = variables_target
        self.normalization = normalization
        self.index_range_source=index_range_source
        self.index_offset_target = index_offset_target
        self.sample_for_norm = sample_for_norm
        self.lazy_load=lazy_load
        self.sample_condition_dict = sample_condition_dict

        self.files_source = files_source
        self.files_target = files_target
        self.model_settings = model_settings
               
        grid_processing = xr.open_dataset(model_settings['processing_grid'])

        self.coords_processing = get_coords_as_tensor(grid_processing, lon='clon', lat='clat')

        self.input_mapping = get_nh_variable_mapping_icon(model_settings['processing_grid'], ['cell'], 
                                                     model_settings['input_grid'], self.variables_source.keys(), 
                                                     search_raadius=model_settings['search_raadius'], max_nh=model_settings['nh_input'], level_start=model_settings['level_start_input'])

        self.output_mapping = get_nh_variable_mapping_icon(model_settings['processing_grid'], ['cell'], 
                                                     model_settings['processing_grid'], self.variables_target.keys(), 
                                                     search_raadius=model_settings['search_raadius'], max_nh=model_settings['nh_input'], level_start=model_settings['level_start_input'])

        ds_source = xr.open_dataset(files_source[0])

        self.global_cells = coarsen_global_cells(torch.arange(len(grid_processing.clon)), torch.tensor(grid_processing.edge_of_cell.values-1), torch.tensor(grid_processing.adjacent_cell_of_edge.values-1), global_level=coarsen_sample_level)[0]

        self.num_datapoints_time = ds_source[list(self.variables_source.values())[0][0]].shape[0]

        self.ds_dict = {}
        for var, norm_dict in normalization.items():
            if len(norm_dict["moments"])==0:
                var_check = "u" if var == "uv" else var
                files = []
                variables = []
                for grid_vars in self.variables_source.values():
                    for variable in grid_vars:
                        variables.append(variable)

                if var_check in variables:
                    files+=list(files_source)

                variables = []
                for grid_vars in self.variables_target.values():
                    for variable in grid_vars:
                        variables.append(variable)

                if var_check in self.variables_target:
                    files+=list(files_target)

                norm_dict["moments"] = get_stats(files, var, norm_dict, self.sample_for_norm)
        
        self.norm_dict = normalization
        self.normalizer = grid_normalizer(self.norm_dict)

    

    def get_files(self, file_path_source, file_path_target=None, file_path_target_past=None):
      
        include_past = True if file_path_target_past is not None else False

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

        if include_past:
            if self.lazy_load:
                ds_target_past = xr.open_dataset(file_path_target_past)
            else:
                ds_target_past = xr.load_dataset(file_path_target_past)
        

        return ds_source, ds_target


    def get_data(self, ds, ts, global_indices, variables_dict, index_mapping_dict=None):
        
        
        sampled_data = {}
        for key, variables in variables_dict.items():
            data_g = []
            for variable in variables:
                data = torch.tensor(ds[variable][ts].values)
                data = data[0] if data.dim() > 1  else data
                data_g.append(data)

            data_g = torch.stack(data_g, dim=-1)

            if index_mapping_dict is not None:
                indices = index_mapping_dict[key][global_indices]
            else:
                indices = global_indices.reshape(-1,1)

            data_g = data_g[indices]
            data_g = data_g.view(data_g.shape[0], data_g.shape[1], -1)

            sampled_data[key] = data_g

        return sampled_data

    def __getitem__(self, index):
        
        if self.index_range_source is not None:
            if (index < self.index_range_source[0]) or (index > self.index_range_source[1]):
                index = int(torch.randint(self.index_range_source[0], self.index_range_source[1]+1, (1,1)))

        if len(self.files_source)>0:
            source_index = torch.randint(0, len(self.files_source), (1,1))
            source_file = self.files_source[source_index]

        target_file = self.files_target[source_index]
        target_file_past = self.files_target_past[source_index] if self.files_target_past is not None else None

        ds_source, ds_target = self.get_files(source_file, file_path_target=target_file, file_path_target_past=target_file_past)

        condition_not_met = True
        while condition_not_met:
            
            sample_index = torch.randint(self.global_cells.shape[0],(1,))[0]
            global_cells_sample = self.global_cells[sample_index]

            data_source = self.get_data(ds_source, index, global_cells_sample, self.variables_source, self.input_mapping['cell'])
            data_target = self.get_data(ds_target, index, global_cells_sample, self.variables_target, self.output_mapping['cell'])

            for key, val in self.sample_condition_dict.items():
                if data_source['cell'][:,:,0].abs().mean() >= val:
                    condition_not_met = False


        if self.normalization is not None:
            data_source = self.normalizer(data_source, self.variables_source)
            data_target = self.normalizer(data_target, self.variables_target)

        indices = {'global_cell': global_cells_sample,
             'sample': sample_index,
             'sample_level': self.coarsen_sample_level}

        return data_source, data_target, indices

    def __len__(self):
        return self.num_datapoints_time