import json
import os
import copy
import xarray as xr
import math
import torch
import torch.nn as nn

from .. import transformer_training as trainer

import climatereconstructionai.model.transformer_helpers as helpers

from ..utils.io import load_ckpt
from ..utils import grid_utils as gu



class nh_spa_mapper_simple(nn.Module):
    def __init__(self, nh, model_dim, dropout=0, PE=None, polar=False, nh_batch_size=-1) -> None: 
        super().__init__()

        self.nh = nh
        self.md_nh = model_dim // nh

        self.nn_layer = helpers.nn_layer(nh, both_dims=False, batch_size=nh_batch_size)

        self.polar=polar

        self.PE = PE

        self.k_proj = nn.Sequential(nn.Linear(self.md_nh*self.nh, self.nh, bias=True))
        
        self.softm = nn.Softmax(dim=-1)

        self.v_proj = nn.Identity()
        
        self.mlp_layer_nh = nn.Identity()

        self.pe_dropout = nn.Dropout(dropout) if PE is not None else nn.Identity()

        self.norm1 = nn.LayerNorm(self.md_nh)

    def forward(self, x, coords_target, coords_source, d_mat=None, return_debug=False):
        
        pos_enc = None 

        #get nearest neighbours
        x_nh, _, cs_nh = self.nn_layer(x, coords_target, coords_source, d_mat=d_mat, skip_self=False)

        if self.polar:
            d_mat, phi = helpers.to_polar(cs_nh[:,0,:,:], cs_nh[:,1,:,:])
            cs_nh = torch.stack((d_mat,phi),dim=1)

        b, t, nh, e = x_nh.shape
        batched = cs_nh.shape[0] == b
        
        pe = self.pe_dropout(self.PE(cs_nh, batched=batched))
       
        k = self.norm1(pe).reshape(b*t,nh,self.md_nh)

        k = self.k_proj(k.reshape(b*t,self.nh*self.md_nh))
        k = self.softm(k).unsqueeze(dim=-1)

        x_nh = x_nh.reshape(b*t,nh,e)
        x_nh = x_nh*k

        x = x_nh.sum(dim=1).view(b,t,e)


        if return_debug:
            pos_enc = pe
            debug_information = {"atts": att.detach(),
                                 "pos_encs":pos_enc}
            
            return x, debug_information
        else:
            return x

        
class nu_grid_sample(nn.Module):
    
    def __init__(self, n_res=90, s=0.5, nh=5):
        super().__init__()

        nh_m = ((nh-1)/2) + 0.5

        self.pixel_offset_normal = nn.Parameter(torch.linspace(nh_m, -nh_m, n_res), requires_grad=False)
        self.pixel_offset_indices = nn.Parameter(torch.linspace(-(nh-1)//2, (nh-1)//2, nh).int(), requires_grad=False)

        self.s = s
        self.n_res = n_res
        self.nh = nh
        self.softmax2d = nn.Softmax2d()

    def forward(self, x, coords):
        b, n, nc = coords.shape
        _, c, nx, ny = x.shape

        positionsx = (coords[:,:,1])*(x.shape[-2]-1)
        positionsy = (coords[:,:,0])*(x.shape[-1]-1)    

        #fine grid for normal distribution
        p_x_o = positionsx.round().view(b,n,1) - self.pixel_offset_normal.view(1,-1)
        p_y_o = positionsy.round().view(b,n,1) - self.pixel_offset_normal.view(1,-1)

        p_x_o = torch.clamp(p_x_o, min=0, max=x.shape[-2])
        p_y_o = torch.clamp(p_y_o, min=0, max=x.shape[-2])

        weights_x = normal(p_x_o, positionsx.view(b,n,1), self.s)
        weights_y = normal(p_y_o, positionsy.view(b,n,1), self.s)

        weights_x = weights_x.reshape(x.shape[0], n, self.nh, self.n_res//self.nh).sum(axis=-1)
        weights_y = weights_y.reshape(x.shape[0], n, self.nh, self.n_res//self.nh).sum(axis=-1)

        weights_2d = torch.matmul(weights_x.view(b, n, self.nh, 1), weights_y.view(b,n,1,self.nh))
        weights_2d = weights_2d/weights_2d.view(b,n,self.nh*self.nh).sum(dim=-1).view(b,n,1,1)
        
        #course grid for pixel locations
        p_x_o = positionsx.round().view(b,n,1) - self.pixel_offset_indices.view(1,-1)
        p_y_o = positionsy.round().view(b,n,1) - self.pixel_offset_indices.view(1,-1)

        p_x_o = torch.clamp(p_x_o.round(), min=0, max=x.shape[-2]-1).long()
        p_y_o = torch.clamp(p_y_o.round(), min=0, max=x.shape[-2]-1).long()

        p_x_o = p_x_o.unsqueeze(dim=-1).repeat(1,1,1,self.nh)
        p_y_o = p_y_o.unsqueeze(dim=-2).repeat(1,1,self.nh,1)

        p_x_o = p_x_o.view(b,n*self.nh**2)
        p_y_o = p_y_o.view(b,n*self.nh**2)

        x = x.permute(2,3,0,1)[p_x_o, p_y_o]
        x = x.permute(0,2,1,-1)

        diag_m = torch.arange(b, device=x.device).long()
        x = x[diag_m, diag_m]
        x = x.permute(0,-1,1)

        x = x.view(b,c,n,self.nh,self.nh)

        x = x*weights_2d.unsqueeze(dim=1)

        x = x.view(b, c, n, self.nh**2).sum(dim=-1)

        return x
    
class input_net(nn.Module):
    def __init__(self, model_settings):
        super().__init__()

        model_dim = model_settings["model_dim"]
        ff_dim = model_settings["ff_dim"]
        nh = model_settings["nh"]

        self.spatial_dim_var_dict = model_settings['spatial_dims_var_source']

        n_mappers = model_settings['n_input_groups']
        
        nh_batch_size = -1 if 'nh_batch_size' not in model_settings.keys() else model_settings['nh_batch_size']

        polar = model_settings['polar']

        model_dim_nh = model_dim // nh
        ff_dim_nh = ff_dim // nh

        if model_settings['abs_pe']:
            self.abs_pe = helpers.RelativePositionEmbedder_mlp(model_dim, ff_dim, transform=model_settings['transform'])
        else:
            self.abs_pe = None

        PE = helpers.RelativePositionEmbedder_mlp(model_dim_nh, ff_dim_nh, transform=model_settings['transform'], polar=polar)

        self.nh_mapping = nn.ModuleList()
        for n in range(n_mappers):
            self.nh_mapping.append(
                nh_spa_mapper_simple(nh, model_dim, PE=PE, polar=polar, nh_batch_size=nh_batch_size)
            )
    
    def forward(self, x: dict, coords_source: dict, coords_source_reg):

        if self.abs_pe is not None:
            batched = coords_source.shape[0] == x.shape[0]
            ape_enc = self.dropout_ape_s(self.APE(coords_source, batched=batched))
            x = x + ape_enc

        x_spatial_dims = []
        nh_mapping_iter = 0
        for spatial_dim, vars in self.spatial_dim_var_dict.items():
            data = torch.concat([x[var] for var in vars], dim=-1)
            x_spatial_dims.append(self.nh_mapping[nh_mapping_iter](data, coords_source_reg, coords_source[spatial_dim]))
            nh_mapping_iter += 1
        x = torch.concat(x_spatial_dims, dim=-1)

        return x
    
def normal(x, mu, s):
    return torch.exp(-0.5*((x-mu)/s)**2)/(s*torch.tensor(math.sqrt(2*math.pi)))


def scale_coords(coords, mn, mx):
    coords_scaled = {}
    for spatial_dim, coords_ in coords.items():
        coords_scaled[spatial_dim] = (coords_ - mn)/(mx - mn)
        coords_scaled[spatial_dim] = torch.clamp(coords_scaled[spatial_dim], min=0, max=1)
    return coords_scaled


class interpolation_net(nn.Module):
    def __init__(self, model_settings):
        super().__init__()

        self.sample = nu_grid_sample(n_res=model_settings['interpolation_sample_pts'],
                                      s=model_settings['interpolation_std'],
                                      nh=model_settings["interpolation_nh"])


    def forward(self, x, coords_target):

        x = x.permute(0,3,1,2)

        x = self.sample(x, coords_target.transpose(-2,-1))

        x = x.permute(0,-1,1)

        return x

class output_net(nn.Module):
    def __init__(self, model_settings, use_gnlll=False):
        super().__init__()

        self.use_gnlll = use_gnlll
        self.grid_to_target = interpolation_net(model_settings)

        self.n_output_groups = model_settings['n_output_groups']
        self.output_dims = model_settings['output_dims']
        self.spatial_dim_var_dict = model_settings['spatial_dims_var_target']

        if use_gnlll:
            self.output_dims = [out_dim*2 for out_dim in self.output_dims]

        self.activation_mu = nn.Identity()

        total_dim_output = int(torch.sum(torch.tensor(self.output_dims)).sum())
        
        if model_settings['gauss']:
            total_dim_output *=2


    def forward(self, x, coords_target):
        
        data_out = {}

        x = torch.split(x, self.output_dims, dim=-1)

        idx = 0
        for spatial_dim, vars in self.spatial_dim_var_dict.items():
  
            data = x[idx]
            data = self.grid_to_target(data, coords_target[spatial_dim])

            if self.use_gnlll:
                data = torch.split(data, len(vars), dim=-1)
                data = torch.stack((self.activation_mu(data[0]), nn.functional.softplus(data[1])), dim=-1)
            else:
                data = self.activation_mu(data).unsqueeze(dim=-1)
            
            data_out.update(dict(zip(vars, torch.split(data, 1, dim=2))))

            idx += 1

        return data_out


class pyramid_step_model(nn.Module):
    def __init__(self, model_settings, load_pretrained=False):
        super().__init__()
        
        self.model_settings = load_settings(model_settings, 'model')

        self.fusion_modules = None

        self.model_settings['n_input_groups'] = len(self.model_settings['spatial_dims_var_source'])
        self.model_settings['input_dims'] = [len(values) for key, values in self.model_settings['spatial_dims_var_source'].items()]

        self.model_settings['n_output_groups'] = len(self.model_settings['spatial_dims_var_target'])
        self.model_settings['output_dims'] = [len(values) for key, values in self.model_settings['spatial_dims_var_target'].items()]

        self.predict_residual = self.model_settings['predict_residual']
        self.use_gnlll = self.model_settings['gauss']
        # core models operate on grids
        self.core_model = nn.Identity()
        
        self.create_grids()
        
        self.input_net = input_net(self.model_settings)
        
        self.output_net_pre = output_net(self.model_settings, use_gnlll=False)
        self.output_net_post = output_net(self.model_settings, use_gnlll=self.use_gnlll)

        self.check_model_dir()

        self.norm = nn.LayerNorm(len(self.model_settings["variables_source"])) if self.model_settings["norm_pre_core"] else nn.Identity()

        if load_pretrained:
            self.check_pretrained(model_dir_check=self.model_settings['model_dir'])
        
        if 'model_dir_pretrained' in self.model_settings.keys() and len(self.model_settings['model_dir_pretrained'])>0:
            self.check_pretrained(model_dir_check=self.model_settings['model_dir_pretrained'])
        
        
    def forward(self, x, coords_source, coords_target):
        
        x = self.input_net(x, coords_source, self.reg_coords_lr)

        b, n, c = x.shape
        x = x.view(b, int(math.sqrt(n)), int(math.sqrt(n)), c)
        
        x_res = x

        if not isinstance(self.core_model, nn.Identity):
            x = self.norm(x)
            x = x.permute(0,-1,1,2)#.unsqueeze(dim=1)

            if self.time_dim:
                x = x.unsqueeze(dim=1)

            x = self.core_model(x)

            if self.time_dim:
                x = x[:,0].permute(0,-2,-1,1)            
            else:
                x = x.permute(0,-2,-1,1)  
     
        coords_target_hr = scale_coords(coords_target, self.range_region_target_rad[0], self.range_region_target_rad[1])
        x = self.output_net_post(x, coords_target_hr)
        
        if self.predict_residual:
            coords_target_lr = scale_coords(coords_target, self.range_region_source_rad[0], self.range_region_source_rad[1])

            x_pre = self.output_net_pre(x_res, coords_target_lr)

            for var in x.keys():
                if self.use_gnlll:
                    mu, std = torch.split(x[var], 1, dim=-1)
                    mu = mu + x_pre[var][:,:,:,[0]] 
                    x[var] = torch.concat((mu,std), dim=-1)
                else:
                    x[var] = x[var] + x_pre[var]
        

        return x, x_res

    def check_model_dir(self):
        self.model_dir = self.model_settings['model_dir']

        model_settings_path = os.path.join(self.model_dir,'model_settings.json')

        self.ckpt_dir = os.path.join(self.model_dir, 'ckpt')
        self.log_dir = os.path.join(self.model_dir, 'log')

        if not os.path.isdir(self.model_dir):
            os.makedirs(self.model_dir)

        with open(model_settings_path, 'w') as f:
            json.dump(self.model_settings, f, indent=4)

        norm_stats_file = os.path.join(self.model_dir,'norm_stats.json')
        if os.path.isfile(norm_stats_file):
            self.norm_stats_file = norm_stats_file
        else:
            self.norm_stats_file = ''

    def set_training_configuration(self, train_settings=None):
        self.train_settings = load_settings(train_settings, id='train')

        self.train_settings['log_dir'] = os.path.join(self.model_dir, 'logs')

        if len(self.train_settings['norm_stats'])==0:
            self.train_settings['norm_stats'] = self.norm_stats_file

        with open(os.path.join(self.model_dir,'train_settings.json'), 'w') as f:
            json.dump(self.train_settings, f, indent=4)


    def create_grids(self):
        # dependend on resolution and fov
        # "static"
        #
        self.radius_region_source_km = self.model_settings['radius_region_source_km']
        self.range_region_source_rad = [-self.radius_region_source_km/(6371), self.radius_region_source_km/(6371)]

        self.radius_region_target_km = self.model_settings['radius_region_target_km']
        self.range_region_target_rad = [-self.radius_region_target_km/(6371), self.radius_region_target_km/(6371)]

        self.n_in, self.n_out = self.model_settings['n_regular']

        self.grid_size_in = (2*self.radius_region_source_km)/self.n_in
        self.grid_size_out = (2*self.radius_region_target_km)/self.n_out

        self.model_settings['range_region_source_rad'] = self.range_region_source_rad
        self.model_settings['range_region_target_rad'] = self.range_region_target_rad
        self.model_settings['grid_size_in'] = self.grid_size_in
        self.model_settings['grid_size_out'] = self.grid_size_out
       
        c_range_lr = torch.linspace(self.range_region_source_rad[0], self.range_region_source_rad[1], self.n_in)

        lon_lr = c_range_lr.view(-1,1).repeat(self.n_in,1)
        lat_lr = c_range_lr.view(-1,1).repeat(1, self.n_in).view(-1,1)

        self.reg_coords_lr = nn.Parameter(torch.stack((lon_lr, lat_lr)).squeeze(), requires_grad=False)


    # -> high-level models first, cache results, then fusion
    def apply_serial(self):
        pass

    # feed data from all levels into the model at once
    def apply_parallel(self):
        pass
    
    def get_region_generator_settings(self):
        region_gen_dict = {
                'rect_source': False,
                'radius_source': self.radius_region_source_km*math.sqrt(2),
                'rect_target': False,
                'radius_target': self.radius_region_target_km*0.9,
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


    def train_(self, train_settings=None, subdir=None, pretrain_subdir=None):

        if train_settings is not None:
            self.set_training_configuration(train_settings)

        if subdir is not None:
            self.model_settings['model_dir'] = os.path.join(self.model_dir, subdir)
            self.check_model_dir()
            self.set_training_configuration(self.train_settings)

        if pretrain_subdir is not None:
            self.check_pretrained(os.path.join(self.model_dir, pretrain_subdir))

        train_settings = self.train_settings

        if "use_samples" in train_settings.keys() and train_settings["use_samples"]:
            train_settings["rel_coords"]=True
        else:
            if "random_region" not in self.train_settings.keys():
                train_settings["random_region"] = self.get_region_generator_settings()


        train_settings["gauss_loss"] = self.model_settings['gauss'] 
        train_settings["variables_source"] = self.model_settings["variables_source"]
        train_settings["variables_target"] = self.model_settings["variables_target"]
        train_settings['model_dir'] = self.model_dir

        trainer.train(self, train_settings, self.model_settings)


    def check_pretrained(self, model_dir_check=''):

        if len(model_dir_check)>0:
            ckpt_dir = os.path.join(model_dir_check,'logs','ckpts')
            weights_path = os.path.join(ckpt_dir, 'best.pth')
            if not os.path.isfile(weights_path):
                weights_paths = [f for f in os.listdir(ckpt_dir) if 'pth' in f]
                weights_paths.sort(key=getint)
                if len(weights_path)>0:
                    weights_path = os.path.join(ckpt_dir, weights_paths[-1])
            
            if os.path.isfile(weights_path):
                self.load_pretrained(weights_path)

    def load_pretrained(self, ckpt_path:str, device=None):
        ckpt_dict = torch.load(ckpt_path)
        self.load_state_dict(ckpt_dict[ckpt_dict['labels'][-1]]['model'], strict=False)


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


