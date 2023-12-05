import json
import os
import copy
import xarray as xr
import math
import torch
import torch.nn as nn

from .. import transformer_training as trainer

import climatereconstructionai.model.transformer_helpers as helpers

from ..utils.io import load_ckpt, load_model
from ..utils import grid_utils as gu
from ..utils.normalizer import normalizer

def get_buckets_1d_batched(coords, n_q=2, equal_size=True):

    if equal_size: 
        keep =  coords.shape[-1] - coords.shape[-1] % (n_q)
        coords = coords[:,:keep]

    b,n = coords.shape

    offset = coords.shape[0]*torch.arange(coords.shape[0], device=coords.device).view(-1,1)
    coords = coords + offset

    quants = torch.linspace(1/n_q, 1-1/n_q, n_q-1, device=coords.device)
    qs = coords.quantile(quants, dim=-1)

    o_b = (coords.shape[0]+1)/2
    boundaries = torch.concat((offset.T-o_b, qs, offset.T+o_b))
    
    buckets = torch.bucketize(coords.flatten(), boundaries.flatten().sort().values, right=True)

    qs = qs - offset.T

    offset_buckets = boundaries.shape[0]*torch.arange(qs.shape[1], device=coords.device).view(-1,1)+1
    
    buckets = buckets.reshape(b,n) - offset_buckets

    idx_sort = buckets.argsort(dim=1) 
    
    indices = torch.stack(idx_sort.chunk(n_q, dim=-1),dim=1)

    return indices, qs, buckets


def get_field(n_output, coords, source, f=8):
    
    x = y = torch.linspace(0, 1, n_output, device=coords.device)
        
    b,n1,n2,c,nf = source.shape
    source_v = source.view(b,n1,n2,-1)

    coords_inter = torch.nn.functional.interpolate(coords, scale_factor=f, mode="bilinear", align_corners=True)
    source_inter = torch.nn.functional.interpolate(source_v.permute(0,-1,1,2), scale_factor=f, mode="bicubic", align_corners=True)
    source_inter = source_inter.view(b,c,nf,source_inter.shape[-2],source_inter.shape[-1])


    dev0 = ((coords_inter[:,[0]].unsqueeze(dim=-1) - x.view(1,-1))).abs()

    _, ind0 = dev0.min(dim=2)

    index_c = ind0.transpose(-2,-1).repeat(1,2,1,1)
    coords_inter0 = torch.gather(coords_inter, dim=2, index=index_c)

    source_inter = torch.gather(source_inter.mean(dim=1), dim=2, index=ind0.transpose(-2,-1).repeat(1,nf,1,1))

    dev1 = ((coords_inter0[:,[1]].unsqueeze(dim=-1) - y.view(1,-1))).abs()
    _, ind = dev1.min(dim=3)

    source_inter = torch.gather(source_inter, dim=3, index=ind.repeat(1,nf,1,1))

    return source_inter.transpose(-2,-1)


def scale_coords2(coords, min_val, max_val):
    c0 = coords[:,0]
    c1 = coords[:,1]
 
    scaled_c0 = (c0 - min_val)/(max_val-min_val)
    scaled_c1 = (c1 - min_val)/(max_val-min_val)

    mn0,mx0 = scaled_c0.min(dim=1).values, scaled_c0.max(dim=1).values
    mn1,mx1 = scaled_c1.min(dim=-1).values, scaled_c1.max(dim=-1).values

    zero_t = mn0.unsqueeze(dim=1) - 0.5
    one_t = mx0.unsqueeze(dim=1) + 0.5
    scaled_c0 = torch.concat((zero_t, scaled_c0, one_t),dim=1)
    scaled_c0 = torch.concat((scaled_c0[:,:,[0]], scaled_c0, scaled_c0[:,:,[-1]]),dim=2)


    zero_t = mn1.unsqueeze(dim=2) - 0.5
    one_t = mx1.unsqueeze(dim=2) + 0.5
    scaled_c1 = torch.concat((zero_t,scaled_c1, one_t),dim=2)
    scaled_c1 = torch.concat((scaled_c1[:,[0]], scaled_c1, scaled_c1[:,[-1]]),dim=1)

    return torch.stack((scaled_c0, scaled_c1), dim=1)


def batch_coords(coords, n_q):

    indices1, _, _ = get_buckets_1d_batched(coords[:,0], n_q, equal_size=True)

    c = coords[:,0]
    c1 = coords[:,1]

    b,nb,n = indices1.shape

    cs0 = torch.gather(c, dim=1, index=indices1.view(b,-1))
    cs0 = cs0.view(b,nb,n)
    cs1 = torch.gather(c1, dim=1, index=indices1.view(b,-1))
    cs1 = cs1.view(b,nb,n)

    b,n1,n = cs1.shape
    indices, _, _ = get_buckets_1d_batched(cs1.view(b*n1,n), n_q, equal_size=True)
    indices2 = indices.view(b,n1,n_q,-1)

    b,n1,n2,n = indices2.shape
    cs1_new = torch.gather(cs1, dim=2, index=indices2.view(b,n1,-1))
    cs1_new = cs1_new.view(b,n1,n2,n)

    cs0_new = torch.gather(cs0, dim=2, index=indices2.view(b,n1,-1))
    cs0_new = cs0_new.view(b,n1,n2,n)


    indices_tot = torch.gather(indices1, dim=-1, index=indices2.view(b,n1,n2*n)).view(b,n1,n2,n)
    cs2 = torch.gather(coords, dim=2, index=indices_tot.view(b,1,-1).repeat(1,2,1))
    cs2 = cs2.view(b,2,n1,n2,n)

    cs2_m = cs2.median(dim=-1).values
    #cs2_s = cs2.min(dim=-1).values
    #cs2_e = cs2.max(dim=-1).values

    return indices_tot, cs2_m


class quant_discretizer(nn.Module):
    def __init__(self, min_val, max_val, n) -> None: 
        super().__init__()

        self.min_nq = 5
        self.n_min = 4
        self.min_val = min_val
        self.max_val = max_val
        self.n = n


    def forward(self, x, coords_source):
                
        n = x.shape[1]

        n_q = int((x.shape[1] // self.n_min)**0.5)
        n_q = self.min_nq if n_q < self.min_nq else n_q
        n_q = self.n if n_q > self.n else n_q
        f = (self.n // n_q) + 1

        coords = coords_source
        data = x

        indices, cm = batch_coords(coords, n_q=n_q)
        b,n1,n2,n = indices.shape

        data_buckets  = torch.gather(data, dim=1, index=indices.view(b,-1,1).repeat(1,1,data.shape[-1]))
        data_buckets = data_buckets.view(b,n1,n2,n,data.shape[-1])

        scaled_m = scale_coords2(cm, self.min_val, self.max_val)

        data = torch.concat((data_buckets[:,:,[0]], data_buckets, data_buckets[:,:,[-1]]),dim=2)
        data = torch.concat((data[:,[0]], data, data[:,[-1]]),dim=1)

        b,n1,n2,n,nf = data.shape
        data = data.view(b,n1,n2,1,n*nf)

        x = get_field(self.n, scaled_m, data, f=f)

        b,_,n1,n2 = x.shape
        x = x.view(b,n,nf,n1,n2)

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

        self.spatial_dim_var_dict = model_settings['spatial_dims_var_source']

        n_mappers = model_settings['n_input_groups']

        self.nh_mapping = nn.ModuleList()

        for n in range(n_mappers):
            self.nh_mapping.append(
                quant_discretizer(model_settings['range_region_source_rad'][0], model_settings['range_region_source_rad'][1], model_settings['n_regular'][0])
            )    


    def forward(self, x: dict, coords_source: dict):

        x_spatial_dims = []
        nh_mapping_iter = 0
        for spatial_dim, vars in self.spatial_dim_var_dict.items():
            data = torch.concat([x[var] for var in vars], dim=-1)
            x_spatial_dims.append(self.nh_mapping[nh_mapping_iter](data, coords_source[spatial_dim]))
            nh_mapping_iter += 1
        x = torch.concat(x_spatial_dims, dim=-1)

        x = x.mean(dim=1)

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
    def __init__(self, model_settings):
        super().__init__()
        
        self.model_settings = load_settings(model_settings, 'model')
        if 'domain' in self.model_settings.keys():
            self.model_settings.update(load_settings(self.model_settings['domain'], 'model'))

        self.fusion_modules = None

        self.model_settings['n_input_groups'] = len(self.model_settings['spatial_dims_var_source'])
        self.model_settings['input_dims'] = [len(values) for key, values in self.model_settings['spatial_dims_var_source'].items()]

        self.model_settings['n_output_groups'] = len(self.model_settings['spatial_dims_var_target'])
        self.model_settings['output_dims'] = [len(values) for key, values in self.model_settings['spatial_dims_var_target'].items()]

        self.output_res_indices = {}
        for var_target in self.model_settings['variables_target']:
            var_in_source = [k for k, var_source in enumerate(self.model_settings['variables_source']) if var_source==var_target]
            if len(var_in_source)==1:
                self.output_res_indices[var_target] = var_in_source[0]

        self.predict_residual = self.model_settings['predict_residual']
        self.use_gnlll = self.model_settings['gauss']
        # core models operate on grids
        self.core_model = nn.Identity()
        
        self.create_grids()
        
        self.input_net = input_net(self.model_settings)
        
        self.output_net_pre = output_net(self.model_settings, use_gnlll=False)
        self.output_net_post = output_net(self.model_settings, use_gnlll=self.use_gnlll)

        self.check_model_dir()

        self.norm = nn.LayerNorm([self.model_settings['n_regular'][1], self.model_settings['n_regular'][1]], elementwise_affine=True) if self.model_settings["norm_pre_core"] else nn.Identity()
        
        
    def forward(self, x, coords_source, coords_target, norm=False):
        
        if norm:
            x = self.normalize(x)

        x = self.input_net(x, coords_source)

        x_reg_lr = x

        if not isinstance(self.core_model, nn.Identity):
            x = self.norm(x)

            if self.time_dim:
                x = x.unsqueeze(dim=1)

            x = self.core_model(x)

            if self.time_dim:
                x = x[:,0].permute(0,-2,-1,1)            
            else:
                x = x.permute(0,-2,-1,1)
            
            x_reg_hr = x
            coords_target_hr = scale_coords(coords_target, self.range_region_target_rad[0], self.range_region_target_rad[1])
            x = self.output_net_post(x, coords_target_hr)
        else:
            x = x.permute(0,-2,-1,1)

            coords_target_hr = scale_coords(coords_target, self.range_region_target_rad[0], self.range_region_target_rad[1])
            x = self.output_net_post(x[:,:,:,list(self.output_res_indices.values())], coords_target_hr)
        
        if self.predict_residual and not isinstance(self.core_model, nn.Identity):
            coords_target_lr = scale_coords(coords_target, self.range_region_source_rad[0], self.range_region_source_rad[1])

            x_reg_lr = x_reg_lr.permute(0,-2,-1,1)
            x_pre = self.output_net_pre(x_reg_lr[:,:,:,list(self.output_res_indices.values())], coords_target_lr)

            for var in self.output_res_indices.keys():
                if self.use_gnlll:
                    mu, std = torch.split(x[var], 1, dim=-1)
                    mu = mu + x_pre[var][:,:,:,[0]] 
                    x[var] = torch.concat((mu,std), dim=-1)
                else:
                    x[var] = x[var] + x_pre[var]
        
        if norm:
            x = self.normalize(x, denorm=True)

        return x, x_reg_lr, x_reg_hr

    def check_model_dir(self):
        self.model_dir = self.model_settings['model_dir']

        model_settings_path = os.path.join(self.model_dir,'model_settings.json')

        self.ckpt_dir = os.path.join(self.model_dir, 'ckpt')
        self.log_dir = os.path.join(self.model_dir, 'log')

        if not os.path.isdir(self.model_dir):
            os.makedirs(self.model_dir)

        with open(model_settings_path, 'w') as f:
            json.dump(self.model_settings, f, indent=4)


    def set_training_configuration(self, train_settings=None):
        self.train_settings = load_settings(train_settings, id='train')

        self.train_settings['log_dir'] = os.path.join(self.model_dir, 'logs')

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

    def set_normalizer(self):
        self.normalize = normalizer(self.model_settings['normalization'])

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

    

    def train_(self, train_settings=None, subdir=None, pretrain_subdir=None, optimization=True):

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

        if optimization:
            trainer.train(self, train_settings, self.model_settings)
        else:
            trainer.no_train(self, train_settings, self.model_settings)


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


