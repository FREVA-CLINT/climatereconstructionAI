import json
import os
import copy
import xarray as xr
import math
import numpy as np
import torch
import torch.nn as nn

from .. import transformer_training as trainer
from torchvision.transforms import GaussianBlur

import climatereconstructionai.model.transformer_helpers as helpers

from ..utils.io import load_ckpt, load_model
from ..utils import grid_utils as gu
from ..utils.normalizer import normalizer


class output_net(nn.Module):
    def __init__(self, model_settings, s, nh, n, use_gnlll=False, use_poly=False):
        super().__init__()

        self.use_gnlll = use_gnlll
        self.use_poly = use_poly

        if 'train_s' not in model_settings.keys():
            train_s=False
        else:
            train_s = model_settings['train_s']

        if nh < 3 or s<0.01:
            self.grid_to_target_sampler = helpers.nu_grid_sampler_simple()
            
        else:
            self.grid_to_target_sampler = helpers.nu_grid_sampler(n_res=n,
                                        s=s,
                                        nh=nh,
                                        train_s=train_s)
        
     #   self.grid_to_target_sampler = helpers.trans_nu_grid_sampler(n_res=n,
     #                                 s=s,
     #                                 nh=nh,
     #                                 model_dim=model_settings['output_dim_core'],
     #                                 output_dim=int(np.array(model_settings['output_dims']).sum()))
        
        self.n_output_groups = model_settings['n_output_groups']
        self.output_dims = model_settings['output_dims']
        self.spatial_dim_var_dict = model_settings['spatial_dims_var_target']

        if use_gnlll or use_poly:
            self.output_dims = [out_dim*2 for out_dim in self.output_dims]

        if use_poly:
            inital_k = 1. if 'poly_k' not in model_settings.keys() else model_settings['poly_k']
            self.k = torch.nn.Parameter(torch.tensor([float(inital_k)]), requires_grad=True)

        self.activation_mu = nn.Identity()

        total_dim_output = int(torch.sum(torch.tensor(self.output_dims)).sum())
        
        if model_settings['gauss']:
            total_dim_output *=2


    def forward(self, x, coords_target, non_valid_mask=None):
        
        data_out = {}
        non_valid_mask_var = {}

        x = torch.split(x, self.output_dims, dim=1)

        idx = 0
        for spatial_dim, vars in self.spatial_dim_var_dict.items():
  
            data = x[idx]

            data = self.grid_to_target_sampler(data, coords_target[spatial_dim].permute(0,2,1))

            if self.use_gnlll:
                data = torch.split(data, len(vars), dim=1)
                data = torch.stack((self.activation_mu(data[0]), nn.functional.softplus(data[1])), dim=2)

            elif self.use_poly:
                data = torch.split(data, len(vars), dim=1)
                data = (self.k*data[0] + (1-self.k)*data[1]**3).unsqueeze(dim=2)

            else:
                data = self.activation_mu(data).unsqueeze(dim=2)
            
            data_out.update(dict(zip(vars, torch.split(data, 1, dim=1))))

            if non_valid_mask is not None:
                non_valid_mask_var.update(dict(zip(vars, [non_valid_mask[spatial_dim]]*len(vars))))
            idx += 1

        return data_out, non_valid_mask_var

class pyramid_step_model(nn.Module):
    def __init__(self, model_settings, model_dir=None, eval=False):
        super().__init__()
        self.eval_mode = eval

        self.model_settings = load_settings(model_settings, 'model')
        if 'domain' in self.model_settings.keys():
            self.model_settings.update(load_settings(self.model_settings['domain'], 'model'))

        if model_dir is not None:
            self.model_settings['model_dir'] = model_dir

        self.fusion_modules = None

        if 'calc_vort' not in self.model_settings.keys():
            self.model_settings['calc_vort'] = True

        self.model_settings['n_input_groups'] = len(self.model_settings['spatial_dims_var_source'])
        self.model_settings['input_dims'] = [len(values) for key, values in self.model_settings['spatial_dims_var_source'].items()]

        self.model_settings['n_output_groups'] = len(self.model_settings['spatial_dims_var_target'])
        self.model_settings['output_dims'] = [len(values) - int(self.model_settings['calc_vort'])*int('vort' in values) for key, values in self.model_settings['spatial_dims_var_target'].items()]   

        self.output_res_indices = {}
        for var_target in self.model_settings['variables_target']:
            var_in_source = [k for k, var_source in enumerate(self.model_settings['variables_source']) if var_source==var_target]
            if len(var_in_source)==1:
                self.output_res_indices[var_target] = var_in_source[0]

        self.res_mode = self.model_settings['res_mode']
        self.use_gnlll = self.model_settings['gauss']
        self.use_poly = self.model_settings['poly']

        self.core_model = nn.Identity()
        
        self.create_grids()
        
        model_settings_pre = self.model_settings

    #    self.output_net_pre = output_net(model_settings_pre,
    #                                     model_settings_pre["interpolation_std_s"],
    #                                      model_settings_pre["interpolation_nh_s"],
    #                                      model_settings_pre["interpolation_sample_pts"],
    #                                     use_gnlll=False)
        
        self.output_net_post = output_net(self.model_settings,
                                          model_settings_pre["interpolation_std"],
                                          model_settings_pre["interpolation_nh"],
                                          model_settings_pre["interpolation_sample_pts"],
                                          use_gnlll=self.use_gnlll,
                                          use_poly=self.use_poly)
        
        if self.eval_mode:
            self.eval()
            self.set_normalizer()

        self.check_model_dir()

        if 'input_type' not in self.model_settings.keys():
            self.set_input_mapper(mode='interpolation')
        else:
            self.set_input_mapper(mode=self.model_settings['input_type'])
     
        
    def forward(self, x, coords_target, coords_source=None, norm=False):
    
        if norm:
            x = self.normalize(x)

        if self.input_mapper is not None:
            x = self.input_mapper(x, coords_source, self.model_settings['spatial_dims_var_source'])

        x_reg_lr = x

        if not isinstance(self.core_model, nn.Identity):

            x = self.core_model(x)
            
            x_reg_hr = x
            coords_target_hr, non_valid = helpers.scale_coords(coords_target, self.range_region_target_radx, rngy=self.range_region_target_rady)
            x, non_valid_var = self.output_net_post(x, coords_target_hr, non_valid)

        else:
            x_reg_hr = x
            coords_target_hr, non_valid = helpers.scale_coords(coords_target, self.range_region_target_radx, rngy=self.range_region_target_rady)
            x, non_valid_var = self.output_net_post(x[:,list(self.output_res_indices.values()),:,:], coords_target_hr, non_valid)

        
        if self.res_mode=='sample' and not isinstance(self.core_model, nn.Identity):
            x_pre = self.output_net_pre(x_reg_lr[:,list(self.output_res_indices.values()),:,:], coords_target_hr, non_valid)[0]

            for var in self.output_res_indices.keys():
                if self.use_gnlll:
                    mu, std = torch.split(x[var], 1, dim=2)
                    mu = mu + x_pre[var]
                    x[var] = torch.concat((mu,std), dim=2)
                else:
                    x[var] = x[var] + x_pre[var]
        
        if norm:
            x = self.normalize(x, denorm=True)

        return x, x_reg_lr, x_reg_hr, non_valid_var
    
    def apply_global(self, ds, ts=-1, device='cpu', ds_target=None):
            
            data_source = {}
            coords_source = {}
            for spatial_dim, vars in self.model_settings["spatial_dims_var_source"].items():
                coord_dict = gu.get_coord_dict_from_var(ds, spatial_dim)

                coords_source[spatial_dim] = gu.get_coords_as_tensor(ds, lon=coord_dict['lon'], lat=coord_dict['lat']).unsqueeze(dim=0)
                for variable in vars:
                    data_source[variable] = torch.tensor(ds[variable].values[[ts]]).squeeze().to(device).unsqueeze(dim=0).unsqueeze(dim=-1)

            if ds_target is None:
                ds_target = ds

            coords_target = {}
            for spatial_dim, vars in self.model_settings["spatial_dims_var_target"].items():
                coord_dict = gu.get_coord_dict_from_var(ds_target, spatial_dim)
                coords_target[spatial_dim] = gu.get_coords_as_tensor(ds_target, lon=coord_dict['lon'], lat=coord_dict['lat']).unsqueeze(dim=0)
    
            self.set_input_mapper(mode="interpolation")

            with torch.no_grad():
                output = self(data_source, coords_target, coords_source=coords_source, norm=True)[0]
                
            return output, {}

    def check_model_dir(self):
        if 'km' not in self.model_settings.keys():
            self.model_settings['km']=False

        self.km = self.model_settings['km']

        self.model_dir = self.model_settings['model_dir']

        model_settings_path = os.path.join(self.model_dir,'model_settings.json')

        self.ckpt_dir = os.path.join(self.model_dir, 'ckpt')

        if 'log_dir' not in self.model_settings.keys():
            self.log_dir = os.path.join(self.model_dir, 'logs')
        else:
            self.log_dir = self.model_settings['log_dir']

        if not os.path.isdir(self.model_dir):
            os.makedirs(self.model_dir)

        if not self.eval_mode:
            with open(model_settings_path, 'w') as f:
                json.dump(self.model_settings, f, indent=4)



    def set_training_configuration(self, train_settings=None):
        self.train_settings = load_settings(train_settings, id='train')

        self.train_settings['log_dir'] = self.log_dir

        with open(os.path.join(self.model_dir,'train_settings.json'), 'w') as f:
            json.dump(self.train_settings, f, indent=4)


    def create_grids(self):
       
        self.radius_region_source_km = self.model_settings['radius_region_source_km']
        self.radius_region_target_km = self.model_settings['radius_region_target_km']

        if self.model_settings['km']:
            self.range_region_source_radx = [-self.radius_region_source_km/(6371), self.radius_region_source_km/(6371)]
            self.range_region_target_radx = [-self.radius_region_target_km/(6371), self.radius_region_target_km/(6371)]

            self.range_region_source_rady = self.range_region_source_radx
            self.range_region_target_rady = self.range_region_target_radx
        else:
            self.range_region_source_radx = [-math.pi, math.pi]
            self.range_region_target_radx = [-math.pi, math.pi]

            self.range_region_source_rady = [-math.pi/2, math.pi/2]
            self.range_region_target_rady = [-math.pi/2, math.pi/2]
 
        self.n_in, self.n_out = self.model_settings['n_regular']

        self.model_settings['range_region_source_radx'] = self.range_region_source_radx
        self.model_settings['range_region_target_radx'] = self.range_region_target_radx
        self.model_settings['range_region_source_rady'] = self.range_region_source_rady
        self.model_settings['range_region_target_rady'] = self.range_region_target_rady

    def set_normalizer(self):
        self.normalize = normalizer(self.model_settings['normalization'])

    def set_input_mapper(self, mode=None):
        if mode is None:
            self.input_mapper = None

        elif mode == 'quantdiscretizer':
            self.input_mapper = helpers.unstructured_to_reg_qdiscretizer(
                self.model_settings['n_regular'][0],
                self.model_settings['range_region_target_rad']
            )

        elif mode == 'interpolation': 
            self.input_mapper = helpers.unstructured_to_reg_interpolator(
                self.model_settings['n_regular'][0],
                self.model_settings['range_region_target_radx'],
                self.model_settings['range_region_target_rady'],
                method=self.model_settings['interpolation_method'] if 'interpolation_method' in self.model_settings else 'nearest' 
            )

    # -> high-level models first, cache results, then fusion
    def apply_serial(self):
        pass

    # feed data from all levels into the model at once
    def apply_parallel(self):
        pass
    
    def get_region_generator_settings(self, lon_trans=False):
        if lon_trans:
            region_gen_dict = {
                    'rect_source': False,
                    'radius_source': -1,
                    'rect_target': False,
                    'radius_target': -1,
                    "lon_range": [
                            -180,
                            180
                        ],
                        "lat_range": [
                            -0,
                            0.0001
                        ],
                    "batch_size": 1,
                    "generate_interval": 1
                }
        else:
            region_gen_dict = {
                    'rect_source': False,
                    'radius_source': self.radius_region_source_km,
                    'rect_target': False,
                    'radius_target': self.radius_region_target_km,
                    "lon_range": [
                            -180,
                            180
                        ],
                        "lat_range": [
                            -90,
                            90
                        ],
                    "batch_size": 1,
                    "generate_interval": 1
                }
        return region_gen_dict

    

    def train_(self, train_settings=None, subdir=None, pretrain_subdir=None, optimization=True):

        if train_settings is not None:
            self.set_training_configuration(train_settings)

        if subdir is not None:
            self.model_settings['model_dir'] = os.path.join(self.model_dir, subdir)
            self.check_model_dir()
            self.set_training_configuration(self.train_settings)

        if pretrain_subdir is not None:
            self.check_pretrained(os.path.join(self.model_dir, pretrain_subdir, 'logs'))

        train_settings = self.train_settings

        if "use_samples" in train_settings.keys() and train_settings["use_samples"]:
            train_settings["rel_coords"]=True
        else:
            if "random_region" not in self.train_settings.keys() and self.model_settings['km']:
                train_settings["random_region"] = self.get_region_generator_settings()
            
            if 'lon_trans' in self.train_settings.keys() and self.train_settings['lon_trans']:
                train_settings["random_region"] = self.get_region_generator_settings(lon_trans=True)

        train_settings["gauss_loss"] = self.model_settings['gauss'] 
        train_settings["variables_source"] = self.model_settings["variables_source"]
        train_settings["variables_target"] = self.model_settings["variables_target"]
        train_settings['model_dir'] = self.model_dir

        interpolation_dict = {
            'interpolate_source': True,
            'interpolate_target': False,
            'interpolation_size_s': self.n_in,
            'range_source_x': self.range_region_source_radx,
            'range_source_y': self.range_region_source_rady,
            'interpolation_method': self.model_settings['interpolation_method'] if 'interpolation_method' in self.model_settings else 'nearest' 
            }

        train_settings['interpolation'] = interpolation_dict

        self.set_input_mapper(mode=None)

        if optimization:
            trainer.train(self, train_settings, self.model_settings)
        else:
            trainer.no_train(self, train_settings, self.model_settings)


    def check_pretrained(self, log_dir_check=''):

        if len(log_dir_check)>0:
            ckpt_dir = os.path.join(log_dir_check, 'ckpts')
            weights_path = os.path.join(ckpt_dir, 'best.pth')
            if not os.path.isfile(weights_path):
                weights_paths = [f for f in os.listdir(ckpt_dir) if 'pth' in f]
                weights_paths.sort(key=getint)
                if len(weights_path)>0:
                    weights_path = os.path.join(ckpt_dir, weights_paths[-1])
            
            if os.path.isfile(weights_path):
                self.load_pretrained(weights_path)

    def load_pretrained(self, ckpt_path:str):
        device = 'cpu' if 'device' not in self.model_settings.keys() else self.model_settings['device']
        ckpt_dict = torch.load(ckpt_path, map_location=torch.device(device))
        load_model(ckpt_dict, self)


def getint(name):
    basename = name.partition('.')
    return int(basename[0])

def load_settings(dict_or_file, id='model'):
    if isinstance(dict_or_file, dict):
        return dict_or_file

    elif isinstance(dict_or_file, str):
        if os.path.isfile(dict_or_file):
            with open(dict_or_file,'r') as file:
                dict_or_file = json.load(file)
        else:
            dict_or_file = os.path.join(dict_or_file, f'{id}_settings.json')
            with open(dict_or_file,'r') as file:
                dict_or_file = json.load(file)

        return dict_or_file


def fetch_data(x, indices):
    b, n, e = x.shape
    n_k = indices.shape[-1]
    indices = indices.view(b,n_k,1).repeat(1,1,e)
    x = torch.gather(x, dim=1, index=indices)
    return x

def fetch_coords(rel_coords, indices):
    b, n_c, n, _ = rel_coords.shape
    n_k = indices.shape[-1]
    indices = indices.view(b,1,n_k,1).repeat(1,2,1,1)
    rel_coords = torch.gather(rel_coords, dim=2, index=indices)
    return rel_coords

    
def merge_debug_information(debug_info, debug_info_new):
    for key in debug_info_new.keys():
        if key in debug_info:
            if not isinstance(debug_info[key],list):
                debug_info[key] = [debug_info[key]]
            if isinstance(debug_info_new[key],list):
                debug_info[key] += debug_info_new[key]
            else:
                debug_info[key].append(debug_info_new[key])
        else:
            debug_info[key] = debug_info_new[key]

    return debug_info


