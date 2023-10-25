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


class nh_spa_mapper(nn.Module):
    def __init__(self, nh, input_dim, ff_dim, model_dim, output_dim, n_heads=4, dropout=0, PE=None, add_pe=False) -> None: 
        super().__init__()

        self.nh = nh
        self.md_nh = model_dim // nh
        ff_dim_nh = ff_dim // nh
        self.n_heads = n_heads

        self.add_pe = add_pe

        if PE is not None:
            self.nn_layer = helpers.nn_layer(nh, both_dims=False)
        else:
            self.nn_layer = helpers.nn_layer(nh, both_dims=True)

        self.PE = PE
        '''
        self.mlp_nh = nn.Sequential(
            nn.Linear(input_dim, self.md_nh, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            nn.LayerNorm(self.md_nh)
        )
        '''
        self.local_att = helpers.MultiHeadAttentionBlock(
            self.md_nh, self.md_nh, n_heads, logit_scale=True, qkv_proj=False
            )

        self.q_proj = nn.Linear(self.md_nh, self.md_nh, bias=False)
        self.k_proj = nn.Linear(self.md_nh, self.md_nh, bias=False)
        self.v_proj = nn.Linear(input_dim, self.md_nh, bias=False)
       
        
        self.mlp_layer_nh = nn.Sequential(
            nn.Linear(self.md_nh, ff_dim_nh, bias=True),
            nn.Dropout(dropout),
            #nn.LeakyReLU(inplace=True, negative_slope=0.2),
            nn.GELU(),
            nn.Linear(ff_dim_nh, 1, bias=True),
            #nn.LeakyReLU(inplace=True, negative_slope=0.2)
            nn.GELU()
        )

        self.mlp_layer_output = nn.Sequential(
            nn.Linear(model_dim, ff_dim, bias=True),
            nn.Dropout(dropout),
            #nn.LeakyReLU(inplace=True, negative_slope=0.2),
            nn.GELU(),
            nn.Linear(ff_dim, output_dim, bias=True),
            #nn.LeakyReLU(inplace=True, negative_slope=0.2)
            nn.GELU()
            )
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.pe_dropout = nn.Dropout(dropout) if PE is not None else nn.Identity()

        self.norm1 = nn.LayerNorm(self.md_nh)
    #    self.norm2 = nn.LayerNorm(self.md_nh)


    def forward(self, x, coords_target, coords_source, d_mat=None, return_debug=False):

        pos_enc = None 
        '''
        x = self.mlp_nh(x)
        '''
        #get nearest neighbours
        x_nh, _, cs_nh = self.nn_layer(x, coords_target, coords_source, d_mat=d_mat, skip_self=False)
        
        b, t, nh, e = x_nh.shape
        batched = cs_nh.shape[0] == b
        
        pe = self.pe_dropout(self.PE(cs_nh, batched=batched))

        if self.add_pe:
            v = q = k = self.norm1(x_nh + pe).reshape(b*t,nh,self.md_nh)
        else:
            q = k = self.norm1(x_nh + pe).reshape(b*t,nh,self.md_nh)
            v = x_nh.reshape(b*t,nh,x_nh.shape[-1])

        v = self.v_proj(v)
        q = self.q_proj(q)
        k = self.k_proj(k)

        x_nh = x_nh.reshape(b*t,nh,e)

        if return_debug:
            pos_enc = pe

        att_out, att = self.local_att(q, k, v, rel_pos_bias=None, return_debug=True)
            
        #x_nh = x_nh + self.dropout1(att_out)
        x_nh = self.dropout1(att_out)
        x_nh = x_nh + self.dropout2(self.mlp_layer_nh(x_nh))
    

        x_nh = x_nh.view(b,t,self.nh*self.md_nh)

        x = self.mlp_layer_output(x_nh)

        if return_debug:
            debug_information = {"atts": att.detach(),
                                 "pos_encs":pos_enc}
            
            return x, debug_information
        else:
            return x
        

class input_net(nn.Module):
    def __init__(self, model_settings):
        super().__init__()

        input_dim = model_settings["input_dim"]
        model_dim = model_settings["model_dim"]
        ff_dim = model_settings["ff_dim"]
        nh = model_settings["nh"]
        dropout = model_settings["dropout"]
        n_heads = model_settings["n_heads"]

        output_dim = input_dim
        if model_settings['gauss']:
            output_dim = output_dim*2

        model_dim_nh = model_dim // nh
        ff_dim_nh = ff_dim // nh

        if model_settings['abs_pe']:
            self.abs_pe = helpers.RelativePositionEmbedder_mlp(model_dim, ff_dim, transform='inv')
        else:
            self.abs_pe = None
        '''
        self.feature_net = nn.Sequential(
                nn.Linear(input_dim, ff_dim, bias=True),
                nn.Dropout(dropout),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Linear(ff_dim, model_dim, bias=True),
                nn.Dropout(dropout),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.LayerNorm(model_dim))
        '''
        PE = helpers.RelativePositionEmbedder_mlp(model_dim_nh, ff_dim_nh, transform='lin')

        self.nn_layer = nh_spa_mapper(nh, input_dim, ff_dim, model_dim, output_dim, n_heads=n_heads, PE=PE, add_pe=False)
    
    def forward(self, x, coords_source, coords_source_reg):

        #x = self.feature_net(x)

        if self.abs_pe is not None:
            batched = coords_source.shape[0] == x.shape[0]
            ape_enc = self.dropout_ape_s(self.APE(coords_source, batched=batched))
            x = x + ape_enc

        x = self.nn_layer(x, coords_source_reg, coords_source)

        return x
    
def normal(x, mu, s):
    return torch.exp(-0.5*((x-mu)/s)**2)/(s*torch.tensor(math.sqrt(2*math.pi)))

class nu_grid_sample(nn.Module):
    
    def __init__(self, n_res=9, s=0.5, nh=3):
        super().__init__()

        nh_m = ((nh-1)/2) + 0.5

        self.pixel_offset_normal = torch.linspace(nh_m, -nh_m, n_res)
        self.pixel_offset_indices = torch.linspace(-(nh-1)//2, (nh-1)//2, nh).int()

       # self.pixel_indices_xy = pixel_offset_indices.view(nh,1)*pixel_offset_indices.view(1,nh)

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
       



def scale_coords(coords, mn, mx):
    coords_scaled = (coords - mn)/(mx - mn)
    return coords_scaled


class interpolation_net(nn.Module):
    def __init__(self, model_settings):
        super().__init__()

        self.sample = nu_grid_sample(s=model_settings['interpolation_s'])
        
    def forward(self, x, coords_target, return_debug=False):

        x = x.permute(0,3,1,2)

        x = self.sample(x, coords_target.permute(0,2,1,-1).squeeze())

        x = x.permute(0,-1,1)
      #  x = self.mlp_out(x)

        return x

class output_net(nn.Module):
    def __init__(self, model_settings):
        super().__init__()
        self.gauss = model_settings['gauss']
        self.grid_to_target = interpolation_net(model_settings)
        self.activation_mu = nn.GELU()
        self.activation_std = nn.Softplus() if self.gauss else nn.Identity()

    def forward(self, x, coords_target):
        x = self.grid_to_target(x, coords_target)

        b, n, c = x.shape
        if self.gauss:
            x = x.view(b, n, c//2, 2)
            x = torch.stack((self.activation_mu(x[:,:,:,0]),self.activation_std(x[:,:,:,1])), dim=-1)
        else:
            x = x.view(b,n,c,1) 
            x = self.activation_mu(x)

        return x
        


class pyramid_step_model(nn.Module):
    def __init__(self, model_settings, load_pretrained=False):
        super().__init__()
        
        self.model_settings = load_settings(model_settings, 'model')

        self.grid_spacings = self.model_settings['grid_spacings']

        self.fusion_modules = None

        # core models operate on grids
        self.core_model = nn.Identity()
        
        self.create_grids()

        self.input_net = input_net(self.model_settings)
        #self.input_interpolation = interpolation_net(self.model_settings)

        self.output_net = output_net(self.model_settings)

        self.check_model_dir()

        if load_pretrained:
            self.check_pretrained(model_dir_check=self.model_settings['model_dir'])



    def forward(self, x, coords_dict):

        coords_source = coords_dict['rel']['source']
        coords_target = coords_dict['rel']['target']
        
        coords_target = scale_coords(coords_target, -self.region_tot_lr/(2.*6371.), self.region_tot_lr/(2.*6371.))

        x = self.input_net(x, coords_source, self.reg_coords_lr)
        
        
        b, n, c = x.shape
        x = x.view(b, int(math.sqrt(n)), int(math.sqrt(n)), c)

        if self.core_model is not nn.Identity():
            x = x.permute(0,-1,1,2).unsqueeze(dim=1)
            x = self.core_model(x)
            x = x[:,0].permute(0,-2,-1,1)

        x = self.output_net(x, coords_target)

        return x, None

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
        self.radius_region_km = self.model_settings['radius_region_km']

        n_lr = 2*self.radius_region_km/(self.grid_spacings[0]/1000)
        l = int(math.log2(n_lr))

        self.n_lr = 2**l

        n_hr = n_lr*self.grid_spacings[0]/self.grid_spacings[1]

        self.n_hr = 2**int(math.log2(n_hr))

        self.region_tot_lr = (self.grid_spacings[0]/1000)*self.n_lr
        self.region_tot_hr = (self.grid_spacings[1]/1000)*self.n_hr

        c_range_lr =torch.linspace(-self.region_tot_lr/2, self.region_tot_lr/2, self.n_lr)/6371.
        lon_lr = c_range_lr.view(-1,1).repeat(self.n_lr,1)
        lat_lr = c_range_lr.view(-1,1).repeat(1, self.n_lr).view(-1,1)

        self.reg_coords_lr = torch.stack((lon_lr, lat_lr))


    # -> high-level models first, cache results, then fusion
    def apply_serial(self):
        pass

    # feed data from all levels into the model at once
    def apply_parallel(self):
        pass
    
    def get_region_generator_settings(self):
        region_gen_dict = {
                'rect': True,
                'radius': self.region_tot_hr/2,
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

    def train_(self, train_settings=None, use_samples=False, subdir=None, pretrain_subdir=None):

        if train_settings is not None:
            self.set_training_configuration(train_settings)

        if subdir is not None:
            self.model_settings['model_dir'] = os.path.join(self.model_dir, subdir)
            self.check_model_dir()
            self.set_training_configuration(self.train_settings)

        if pretrain_subdir is not None:
            self.check_pretrained(os.path.join(self.model_dir, pretrain_subdir))

        train_settings = self.train_settings

        if not use_samples:
            if "random_region" not in self.train_settings.keys():
                train_settings["random_region"] = self.get_region_generator_settings()
        else:
            train_settings["rel_coords"]=True


        train_settings["gauss_loss"] = self.model_settings['gauss'] 

        train_settings["variables"] = self.model_settings["variables"]
        train_settings["coord_dict"] = self.model_settings["coord_dict"]
        train_settings['model_dir'] = self.model_dir

        trainer.train(self, train_settings, self.model_settings)



    def create_samples(self, sample_settings=None):
        
        sample_settings = load_settings(sample_settings, 'model')

        if "random_region" not in sample_settings.keys():
            sample_settings["random_region"] = self.get_region_generator_settings()

        sample_settings["variables"] = self.model_settings["variables"]
        sample_settings["coord_dict"] = self.model_settings["coord_dict"]
        sample_settings["model_dir"] = self.model_dir
        
        trainer.create_samples(sample_settings)


    def check_pretrained(self, model_dir_check=''):

        if len(model_dir_check)>0:
            ckpt_dir = os.path.join(model_dir_check,'logs','ckpts')
            weights_path = os.path.join(model_dir_check, 'best.pth')
            if not os.path.isfile(weights_path):
                weights_paths = sorted([f for f in os.listdir(ckpt_dir) if 'pth' in f])
                if len(weights_path)>0:
                    weights_path = os.path.join(ckpt_dir, weights_paths[-1])
                else:
                    raise Exception("No pretrained model was found")
                
            self.load_pretrained(weights_path)

    def load_pretrained(self, ckpt_path:str, device=None):
        ckpt_dict = load_ckpt(ckpt_path, device=device)
        self.load_state_dict(ckpt_dict, strict=False)

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


