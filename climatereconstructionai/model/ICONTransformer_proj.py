import json,os
import math
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr

import copy

from ..utils.io import load_ckpt, load_model
import climatereconstructionai.model.transformer_helpers as helpers
from climatereconstructionai.utils.grid_utils import get_distance_angle, get_coords_as_tensor, get_mapping_to_icon_grid, get_nh_variable_mapping_icon, get_adjacent_indices, icon_grid_to_mgrid, mapping_to_, scale_coordinates
from .. import transformer_training as trainer
from ..utils.normalizer import grid_normalizer

def dict_to_device(d, device):
    for key, value in d.items():
        if isinstance(value, dict):
            for key2, value2 in value.items():
                d[key][key2] = value2.to(device)
        else:
            d[key] = value.float().to(device)
    return d

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
    
def append_debug_dict(debug_dict_all, debug_dict):
    for type, debug_info in debug_dict.keys():
        if type in debug_dict_all.keys():
            debug_dict_all[type].append(debug_info)
        else:
            debug_dict_all[type] = []

    return debug_dict_all



class input_layer(nn.Module):
    def __init__(self, input_dim, model_dim) -> None: 
        super().__init__()

        self.input_mlp = nn.Sequential(
                        nn.Linear(input_dim, model_dim, bias=False)
                        )
        #self.input_mlp = nn.Identity()

    def forward(self, x):

        x = self.input_mlp(x)

        return x.squeeze()



class grid_layer(nn.Module):
    def __init__(self, global_level, adjc, adjc_mask, coordinates, projection_dict) -> None: 
        super().__init__()

        self.global_level = global_level
        self.register_buffer("coordinates", coordinates, persistent=False)
        self.register_buffer("adjc", adjc, persistent=False)
        self.register_buffer("adjc_mask", adjc_mask==False, persistent=False)
        self.register_buffer("fov_mask", ((adjc_mask==False).sum(dim=-1)==adjc_mask.shape[1]).view(-1,1),persistent=False)

        n_samples = torch.min(torch.tensor([self.adjc.shape[0]-1, 100]))
        nh_samples = self.adjc[:n_samples]
        coords_nh = self.get_coordinates_from_grid_indices(nh_samples)

        dists = get_relative_positions(coords_nh, coords_nh, polar=True)[0]

        self.min_dist = dists[dists>1e-10].min()
        self.max_dist = dists[dists>1e-10].max()
        self.mean_dist = dists[dists>1e-10].mean()
        self.median_dist = dists[dists>1e-10].median()
        
        self.n_dist = projection_dict["n_dist"]
        self.n_theta = projection_dict["n_theta"]

      #  self.kappa_vm = nn.Parameter(torch.tensor(projection_dict["kappa_vm"], dtype=float), requires_grad=True)
      #  self.sigma_d = nn.Parameter(torch.tensor(self.min_dist/10, dtype=float), requires_grad=True)

        # calc max dist to neighbours

    def get_nh(self, x, local_indices, sample_dict):
        indices_nh, mask = get_nh_indices(self.adjc, local_cell_indices=local_indices, global_level=int(self.global_level))
        adjc_mask = self.adjc_mask[local_indices]
        mask = torch.logical_or(mask, adjc_mask)
        x = gather_nh_data(x, indices_nh, sample_dict['sample'], sample_dict['sample_level'], int(self.global_level))
        coords = self.get_coordinates_from_grid_indices(indices_nh)
        return x, mask, coords

    def get_coordinates_from_grid_indices(self, local_indices):
        coords = self.coordinates[:, local_indices]
        return coords

    def get_sections(self, x, local_indices, section_level=1):
        indices = sequenize(local_indices, max_seq_level=section_level)
        x = sequenize(x, max_seq_level=section_level)
        coords = self.get_coordinates_from_grid_indices(indices)
        mask = self.fov_mask[indices]
        return x, mask, coords 
    
    def get_projection_nh(self, x, local_indices, sample_dict, sigma_d, kappa_vm):
        x_nh, _, coords_nh = self.get_nh(x, local_indices, sample_dict)

        distances, phis = get_relative_positions(coords_nh[:,:,:,0], coords_nh, polar=True)

        weights = get_spatial_projection_weights(phis, 
                                                 distances, 
                                                 kappa_vm=kappa_vm,
                                                 sigma_d=sigma_d,
                                                 n_theta=self.n_theta,
                                                 n_dist=self.n_dist,
                                                 max_dist=self.max_dist, 
                                                 min_dist=0, 
                                                 device=x_nh.device)

        x = proj_data(x_nh, weights)
        
        # x.shape = batch, N, phis, dists, feat
        return x

    def get_projection_cross(self, x, local_indices, sample_dict, coords_cross, sigma_d, kappa_vm):
        x_nh, mask_nh, coords_nh = self.get_nh(x, local_indices, sample_dict)

       # x_nh = x_nh[:,:,1:,:]
       # mask_nh = mask_nh[:,:,1:]
       # coords_nh = coords_nh[:,:,:,1:]
        distances_nh, phis_nh = get_relative_positions(coords_cross[:,:,:,0], coords_nh, polar=True)
        distances, phis = get_relative_positions(coords_cross[:,:,:,0], coords_cross, polar=True)

        #distances_nh[mask_nh.unsqueeze(dim=-2)]=1e5

        weights = get_spatial_projection_weights(phis_nh, 
                                                 distances_nh, 
                                                 kappa_vm=kappa_vm,
                                                 sigma_d=sigma_d,
                                                 n_theta=self.n_theta,
                                                 n_dist=self.n_dist,
                                                 max_dist=self.max_dist, 
                                                 min_dist=0, 
                                                 device=x_nh.device,
                                                 thetas_proj=phis,
                                                 dists_proj=distances)

        x = proj_data(x_nh, weights)
        
        # x.shape = batch, N, phis, dists, feat
        return x

#def get_proejction_levels


def von_mises(thetas, theta_offsets, kappa):

    if not torch.is_tensor(theta_offsets):
        theta_offsets = torch.tensor(theta_offsets)

    vm_norm = 1
    vm = vm_norm * torch.exp(kappa * torch.cos(thetas.unsqueeze(dim=-1) - theta_offsets.unsqueeze(dim=-2)))
    return vm


def normal_dist(distances, distances_offsets, sigma):

    if not torch.is_tensor(distances_offsets):
        distances_offsets = torch.tensor(distances_offsets)

    norm = 1
    nd = norm * torch.exp(-0.5 * ((distances.unsqueeze(dim=-1) - distances_offsets.unsqueeze(dim=-2)) / sigma) ** 2)
    return nd


def get_spatial_projection_weights(phis, dists, kappa_vm, sigma_d, n_theta, n_dist, max_dist, min_dist=0, device="cpu", thetas_proj=None, dists_proj=None):

    if dists_proj is None:
        dists_proj = torch.linspace(min_dist, max_dist, n_dist, device=device).unsqueeze(dim=-1).repeat_interleave(n_theta, dim=-1)

    if thetas_proj is None:
        thetas_proj = torch.linspace(-torch.pi, torch.pi, n_theta + 1, device=device)[:-1].unsqueeze(dim=0).repeat_interleave(n_dist, dim=0)


    vm_weights = von_mises(thetas_proj, phis, kappa_vm)
    dist_weights = normal_dist(dists_proj, dists, sigma_d)
    
    if dists_proj is None:
        weights = vm_weights.unsqueeze(dim=-2) * dist_weights.unsqueeze(dim=-3)
    else:
        weights = vm_weights * dist_weights

    weights_norm = weights.sum(dim=[-1], keepdim=True)

    return weights/(weights_norm+1e-10)


def proj_data(data, weights):
    if weights.dim() - data.dim() == 2:
        weights = weights.unsqueeze(dim=-1)
        data = data.unsqueeze(dim=2).unsqueeze(dim=2).unsqueeze(dim=2)

    elif weights.dim() - data.dim() == 1:
        weights = weights.unsqueeze(dim=-1)
        data = data.unsqueeze(dim=2).unsqueeze(dim=2)

    projection = weights * data
    return projection.sum(dim=-2)


class attention_block(nn.Module):
    def __init__(self, input_dim, model_dim, ff_dim, n_heads=4, dropout=0, output_dim=None, activation=nn.SiLU()) -> None: 
        super().__init__()

        self.model_dim = model_dim
        self.n_heads = n_heads
        
        #self.norm = nn.LayerNorm(model_dim, elementwise_affine=True) 
        self.norm = nn.LayerNorm(model_dim, elementwise_affine=True)

        if output_dim is not None:
            self.output_layer = nn.Sequential(
                    nn.Linear(model_dim, output_dim, bias=False),
                )
        else:
            self.output_layer = nn.Identity()

        self.MHA = helpers.MultiHeadAttentionBlock(
            model_dim, model_dim, n_heads, input_dim=input_dim
            )           

        self.mlp_layer = nn.Sequential(
            nn.LayerNorm(model_dim, elementwise_affine=True),
            nn.Linear(model_dim, ff_dim, bias=False),
            activation,
            nn.Linear(ff_dim, model_dim, bias=False)
        )
        self.dropout2 = nn.Dropout(dropout)

        self.dropout1 = nn.Dropout(dropout)

    def forward(self, x: torch.tensor, xq=None, xv=None, mask=None):    
    
        b, n, nh, e = x.shape
        x = x.reshape(b*n,nh,e)

        q = k = v = self.norm(x)

        if xq is not None:
            xq = self.norm(xq)
            b, nq = xq.shape[:2]
            q = xq.reshape(b*nq,-1,xq.shape[-1])

        if xv is not None:
            xv = self.norm(xv)
            b, nv = xv.shape[:2]
            v = xv.reshape(b*nv,-1,xv.shape[-1])

        if mask is not None:
            full_seq_mask = mask.sum(dim=-1)==mask.shape[-1]
            full_seq_mask_b, full_seq_mask_n, full_seq_mask_t = torch.where(full_seq_mask)
            mask[full_seq_mask_b, full_seq_mask_n, full_seq_mask_t, 0]=False
            mask = mask.view(b, n, mask.shape[-2],-1)

        att_out, att = self.MHA(q=q, k=k, v=v, return_debug=True, mask=mask) 
      
        x = x + self.dropout1(att_out)
        x = x + self.dropout2(self.mlp_layer(x))

        x = x.view(b,n,-1,e)

        return self.output_layer(x)

class decomp_layer(nn.Module):
    def __init__(self, grid_layers: dict) -> None: 
        super().__init__()

        self.grid_layers = grid_layers
        
        self.max_level = list(grid_layers.keys())[-1]

        self.global_levels = []
        for global_level in grid_layers.keys(): 
            if global_level != self.max_level:
                self.global_levels.append(int(global_level))
            
    def forward(self, x, indices_layers, sample_dict):
        
        x_levels={}
        for global_level in self.global_levels:
            
            b,n,e = x.shape

            x_sections = self.grid_layers[str(global_level)].get_sections(x, indices_layers[global_level], section_level=1)[0]

            x_sections_f = x_sections.mean(dim=-2, keepdim=True)

            x_levels[global_level] = (x_sections - x_sections_f).view(b,n,-1)
            
            x = x_sections_f.squeeze(dim=-2)

        if len(self.global_levels) == 0:
            x_levels[0] = x
        else:
            x_levels[global_level + 1] = x

        return x_levels



class nh_processing_layer(nn.Module):
    def __init__(self, grid_layers: dict, model_hparams: dict, periodic_fov=None) -> None: 
        super().__init__()

        model_dim = model_hparams['model_dim']
        ff_dim = model_dim
        n_heads = 1#model_hparams['n_heads']
        nh = model_hparams['nh']
        n_blocks = 4

        self.periodic_fov = periodic_fov

        self.grid_layers = grid_layers
    
        self.global_levels = []
        self.channel_att_layers = nn.ModuleDict()

        # potentially increase 
        self.simga_d = nn.ParameterDict()
        self.kappa_vm = nn.ParameterDict()
        for global_level in grid_layers.keys():

            n_dist, n_theta = grid_layers[global_level].n_dist, grid_layers[global_level].n_theta

            self.simga_d[global_level] = nn.Parameter(grid_layers[global_level].min_dist/2, requires_grad=True)
            self.kappa_vm[global_level] = nn.Parameter(torch.tensor(model_hparams["kappa_vm"], dtype=float), requires_grad=True)

            nh_channel_att_layers = nn.ModuleList()
            for _ in range(model_hparams['n_processing_layers']):
                nh_channel_att_layers.append(attention_block(n_dist*n_theta, n_dist*n_theta, n_dist*n_theta, n_heads=n_heads, output_dim=1))

            self.channel_att_layers[global_level] = nh_channel_att_layers

            self.global_levels.append(int(global_level))

            
    def forward(self, x_levels, indices_layers, sample_dict):
        

        for global_level in self.global_levels[::-1]:
            x = x_levels[global_level]
            b,n,f = x.shape

            for layer in self.channel_att_layers[str(global_level)]:
                x = self.grid_layers[str(global_level)].get_projection_nh(x, indices_layers[global_level], sample_dict, self.simga_d[str(global_level)], self.kappa_vm[str(global_level)])        
                x = x.view(b,n,-1,f).transpose(-1,-2)
                x = layer(x)
                x = x.view(b,n,f)
            x_levels[global_level] = x
        return x_levels
    

class nh_cross_processing_layer(nn.Module):
    def __init__(self, grid_layers: dict, model_hparams: dict, periodic_fov=None) -> None: 
        super().__init__()

        model_dim = model_hparams['model_dim']
        ff_dim = model_dim
        n_heads = 1#model_hparams['n_heads']
        nh = model_hparams['nh']
        n_blocks = 4

        self.periodic_fov = periodic_fov

        self.grid_layers = grid_layers
    
        self.global_levels = []
        self.channel_att_layers = nn.ModuleDict()

        # potentially increase 
        self.simga_d = nn.ParameterDict()
        self.kappa_vm = nn.ParameterDict()
        for global_level in grid_layers.keys():

            n_dist, n_theta = grid_layers[global_level].n_dist, grid_layers[global_level].n_theta

            self.simga_d[global_level] = nn.Parameter(grid_layers[global_level].min_dist/2, requires_grad=True)
            self.kappa_vm[global_level] = nn.Parameter(torch.tensor(model_hparams["kappa_vm"], dtype=float), requires_grad=True)

            nh_channel_att_layers = nn.ModuleList()
            for _ in range(model_hparams['n_processing_layers']):
                nh_channel_att_layers.append(attention_block(n_dist*n_theta, n_dist*n_theta, n_dist*n_theta, n_heads=n_heads, output_dim=1))

            self.channel_att_layers[global_level] = nh_channel_att_layers

            self.global_levels.append(int(global_level))

            
    def forward(self, x_levels, indices_layers, sample_dict):
        

        for global_level in self.global_levels[::-1]:
            x = x_levels[global_level]
            b,n,f = x.shape

            for layer in self.channel_att_layers[str(global_level)]:
                coords = self.grid_layers[str(global_level+1)].get_coordinates_from_grid_indices(indices_layers[global_level])

                x_lr = self.grid_layers[str(global_level+1)].get_projection_cross(x_levels[global_level+1], 
                                                                                  indices_layers[global_level+1], 
                                                                                  sample_dict, 
                                                                                  coords, 
                                                                                  self.simga_d[str(global_level+1)], 
                                                                                  self.kappa_vm[str(global_level+1)]) 

                x = self.grid_layers[str(global_level)].get_projection_nh(x, indices_layers[global_level], sample_dict, self.simga_d[str(global_level)], self.kappa_vm[str(global_level)])        
                x = x.view(b,n,-1,f).transpose(-1,-2)
                x = layer(x)
                x = x.view(b,n,f)

        return x

class projection_layer_output(nn.Module):
    def __init__(self, grid_layers: dict, model_hparams: dict, output_dim, output_mappings=None, periodic_fov=None) -> None: 
        super().__init__()

        model_dim = model_hparams['model_dim']

        self.periodic_fov = periodic_fov

        self.grid_layers = grid_layers
                
        self.global_levels = []
        self.layers = nn.ModuleDict()
        self.simga_d = nn.ParameterDict()
        self.kappa_vm = nn.ParameterDict()

        for global_level in grid_layers.keys():
            self.layers[global_level] = nn.Linear(model_dim, output_dim, bias=False)
            self.global_levels.append(int(global_level))
            self.simga_d[global_level] = nn.Parameter(grid_layers[global_level].min_dist/2, requires_grad=True)
            self.kappa_vm[global_level] = nn.Parameter(torch.tensor(model_hparams["kappa_vm"], dtype=float), requires_grad=True)
            
    def forward(self, x_levels, indices_layers, sample_dict, output_coords=None):
        
        if output_coords is None:
            output_coords = self.grid_layers["0"].get_coordinates_from_grid_indices(indices_layers[0])

        x_output = []

        for global_level in self.global_levels:
            
            output_coords = output_coords.reshape(2,output_coords.shape[1], -1, 4**global_level)

            x = self.layers[str(global_level)](x_levels[global_level])

            x = x.view(x.shape[0], -1, x.shape[-1])

            x = self.grid_layers[str(global_level)].get_projection_cross(x, indices_layers[global_level], sample_dict, output_coords, self.simga_d[str(global_level)], self.kappa_vm[str(global_level)]) 
            
            x = x.view(x.shape[0], -1, x.shape[-1])

            x_output.append(x)
        return x_output

class projection_layer_add(nn.Module):
    def __init__(self, grid_layers: dict, model_hparams: dict, output_dim, output_mappings=None, periodic_fov=None) -> None: 
        super().__init__()

        model_dim = model_hparams['model_dim']

        self.periodic_fov = periodic_fov

        self.grid_layers = grid_layers
                
        self.global_levels = []
        self.layers = nn.ModuleDict()
        for global_level in grid_layers.keys():
            self.layers[global_level] = nn.Linear(model_dim, output_dim, bias=False)
            self.global_levels.append(int(global_level))
            
    def forward(self, x_levels, indices_layers, output_coords=None):
        
        if output_coords is None:
            output_coords = self.grid_layers["0"].get_coordinates_from_grid_indices(indices_layers[0])

        x = self.layers["0"](x_levels[0])

        for global_level in self.global_levels[1:]:
            
            x = sequenize(x, max_seq_level=global_level) + self.layers[str(global_level)](x_levels[global_level]).unsqueeze(dim=-2)
            x = x.view(x.shape[0], -1, x.shape[-1])

        return x

def get_relative_positions(coords1, coords2, polar=False, periodic_fov=None):
    
    if coords2.dim() > coords1.dim():
        coords1 = coords1.unsqueeze(dim=-1)

    if coords1.dim() > coords2.dim():
        coords2 = coords2.unsqueeze(dim=-2)

    if coords1.dim() == coords2.dim():
        coords1 = coords1.unsqueeze(dim=-1)
        coords2 = coords2.unsqueeze(dim=-2)

    distances, phis = get_distance_angle(coords1[0], coords1[1], coords2[0], coords2[1], base="polar" if polar else "cartesian", periodic_fov=periodic_fov)

    return distances.float(), phis.float()


class ICON_Transformer(nn.Module):
    def __init__(self, model_settings) -> None: 
        super().__init__()

        self.model_settings = load_settings(model_settings, id='model')

        self.check_model_dir()

        self.var_model = self.model_settings["var_model"] if "var_model" in self.model_settings.keys() else False
        self.pos_emb_calc = self.model_settings["pos_emb_calc"]
        self.polar = True if "polar" in  self.pos_emb_calc else False

        self.grid = xr.open_dataset(self.model_settings['processing_grid'])
        

        n_grid_levels = self.model_settings['n_grid_levels']

        clon_fov = self.model_settings['clon_fov'] if 'clon_fov' in self.model_settings.keys() else None
        clat_fov = self.model_settings['clat_fov'] if 'clat_fov' in self.model_settings.keys() else None
        self.n_grid_levels_fov = self.model_settings['n_grid_levels_fov'] if 'n_grid_levels_fov' in self.model_settings.keys() else n_grid_levels

        self.scale_input = self.model_settings['scale_input'] if 'scale_input' in self.model_settings.keys() else 1
        self.scale_output = self.model_settings['scale_output'] if 'scale_output' in self.model_settings.keys() else 1
        self.periodic_fov = clon_fov if ('input_periodicty' in self.model_settings.keys() and self.model_settings['input_periodicty']) else None

        if 'mgrids_path' not in self.model_settings.keys():
            fov_extension = self.model_settings['fov_extension'] if 'fov_extension' in self.model_settings.keys() else 0.1
            mgrids = icon_grid_to_mgrid(self.grid, self.n_grid_levels_fov, clon_fov=clon_fov, clat_fov=clat_fov, nh=self.model_settings['nh'], extension=fov_extension)
            self.model_settings['mgrids_path'] = os.path.join(self.model_settings['model_dir'], 'mgrids.pt')
            torch.save(mgrids, self.model_settings['mgrids_path'])
        else:
            mgrids = torch.load(self.model_settings['mgrids_path'])

        self.register_buffer('global_indices', torch.arange(mgrids[0]['coords'].shape[1]).unsqueeze(dim=0), persistent=False)
        self.register_buffer('cell_coords_global', mgrids[0]['coords'], persistent=False)  

        self.input_data  = self.model_settings['variables_source']
        self.output_data = self.model_settings['variables_target']

        share_emb_every = self.model_settings['share_emb_every'] if 'share_emb_every' in self.model_settings.keys() else n_grid_levels
        pos_embedder = {}
        pos_embedder['pos_emb_dim'] = self.model_settings["pos_emb_dim"]
        pos_embedder['polar'] = self.polar
        pos_embedder['pos_embedder_handle'] = None
    
        grid_layers = nn.ModuleDict()
        self.global_levels = []
        pos_embedders = {}

        projection_dict = {"kappa_vm": 1,
                           "n_theta": 6,
                           "n_dist": self.model_settings["nh"]}
        
        self.model_settings.update(projection_dict)

        for grid_level_idx in range(n_grid_levels):
            global_level = grid_level_idx
            self.global_levels.append(grid_level_idx)
            grid_layers[str(global_level)] = grid_layer(global_level, mgrids[global_level]['adjc_lvl'], mgrids[global_level]['adjc_mask'], mgrids[global_level]['coords'], projection_dict)


        input_mapping, input_in_range, input_coordinates, output_mapping, output_in_range, output_coordinates = self.get_grid_mappings(mgrids[0]['coords'])

        self.input_layers = self.init_input_layers(self.model_settings['input_dim_var'])

        self.decomp_layer = decomp_layer(grid_layers)

        self.processing_layer = nh_processing_layer(grid_layers, self.model_settings)
        #self.proj_layer = projection_layer_add(grid_layers, self.model_settings, output_dim=len(self.model_settings['variables_target']['cell']))
        self.proj_layer = projection_layer_output(grid_layers, self.model_settings, output_dim=len(self.model_settings['variables_target']['cell']))    
        #self.processing_layer = processing_layer(grid_layers, self.model_settings, output_dim=len(self.model_settings['variables_target']['cell']))

        if self.model_settings['processing_grid'] != self.model_settings['output_grid']:
            self.register_buffer('output_mapping', output_mapping['cell']['cell'], persistent=False)  
            self.register_buffer('output_coords', output_coordinates['cell'], persistent=False)  
            self.register_buffer('output_in_range', output_in_range['cell']['cell'], persistent=False) 

        strict = self.model_settings['load_strict'] if 'load_strict' in self.model_settings.keys() else True

        trained_iterations = None
        if "pretrained_path" in self.model_settings.keys():
            trained_iterations = self.check_pretrained(log_dir_check=self.model_settings['pretrained_path'], strict=strict)

        if "pretrained_pos_embeddings_path" in self.model_settings.keys():
            self.check_pretrained(log_dir_check=self.model_settings['pretrained_pos_embeddings_path'], strict=False, match_list='pos_embedder')

        if "pretrained_model_wo_input" in self.model_settings.keys():
            self.check_pretrained(log_dir_check=self.model_settings['pretrained_model_wo_input'], strict=False, match_list='pos_embedder', not_match='input')

        self.trained_iterations = trained_iterations


    def forward(self, x, indices_batch_dict=None, debug=False, output_sum=False):
        # if global_indices are provided, batches in x are treated as independent
        debug_dict = {}

        if indices_batch_dict is None:
            indices_batch_dict = {'global_cell': self.global_indices,
                                  'local_cell': self.global_indices,
                                   'sample': None,
                                   'sample_level': None,
                                   'output_indices': None}
        else:
            indices_layers = dict(zip(self.global_levels,[self.get_global_indices_local(indices_batch_dict['sample'], indices_batch_dict['sample_level'], global_level) for global_level in self.global_levels]))
        
        input_data = []
        for key, values in x.items():
            
            input = self.input_layers[key](values)
            input_data.append(input) 
        
        x = torch.concat(input_data, dim=-1)

        x_levels = self.decomp_layer(x, indices_layers, indices_batch_dict)

        #x_levels = self.processing_layer(x_levels, indices_layers, indices_batch_dict)

        if debug:
            debug_dict['x_levels_input'] = x_levels

        #for layers in self.processing_layers:
        #    x_levels = layers(x_levels, indices_layers, indices_batch_dict)

        if 'output_mapping' in self.__dict__['_buffers']:
            output_coords = self.output_coords[:,self.output_mapping[indices_layers[0]]]
        else:
            output_coords = None

        x_levels = self.processing_layer(x_levels, indices_layers, indices_batch_dict)

        #x = self.proj_layer(x_levels, indices_layers, output_coords=output_coords)
        x = self.proj_layer(x_levels, indices_layers, indices_batch_dict, output_coords=None)

        x_var = []
        if output_sum:
            if self.var_model:
                x = torch.sum(torch.stack(x, dim=-1), dim=-1)
                x_var = torch.sum(torch.stack(x_var, dim=-1), dim=-1)
                output = {'cell': torch.stack([x, x_var],dim=-2)}
        
            else:
                x = torch.sum(torch.stack(x, dim=-1), dim=-1)
                x = self.output_mlp(x)

                output = {'cell': x.unsqueeze(dim=-2)}
        else:
            output = {'x': x, 'x_var': x_var}

        if debug:
            return output, debug_dict
        else:
            return output

        
    
    #currently cell only
    
    def apply_on_nc(self, ds, ts, sample_lvl=6, batch_size=8):
    
        normalizer = grid_normalizer(self.model_settings['normalization'])

        if isinstance(ds, str):
            ds = xr.open_dataset(ds)
        
        indices = self.global_indices.reshape(-1, 4**sample_lvl)

        indices_batches = indices.split(batch_size, dim=0)

        sample_id = 0
        outputs = []
        for indices_batch in indices_batches:
            sample_idx_min = (sample_id) * (batch_size)

            data = self.get_data_from_ds(ds, ts, self.model_settings["variables_source"], 0, indices_batch)

            sample_indices = torch.arange(sample_idx_min, sample_idx_min + (len(indices_batch)))

            indices_batch_dict = {'global_cell': indices_batch,
                    'local_cell': indices_batch // 4**self.model_settings['global_level_start'],
                        'sample': sample_indices,
                        'sample_level': sample_lvl* torch.ones((sample_indices.shape[0]), dtype=torch.int)}
            
            data = normalizer(data, self.model_settings["variables_source"])

            with torch.no_grad():
                output = self(data, indices_batch_dict=indices_batch_dict)

            output = normalizer(output, self.model_settings["variables_target"], denorm=True)
            outputs.append(output)
            sample_id+=1
        
        outputs_all = {}
        for output in  outputs:
            for key, output_batch in output.items():
                if key in outputs_all.keys():
                    outputs_all[key].append(output_batch)
                else:
                    outputs_all[key]= [output_batch]

        outputs = outputs_all

        for key, outputs in  outputs.items():
            outputs_all[key] = torch.concat(outputs, dim=0)

        #output = grid_dict_to_var_dict(outputs_all, self.model_settings["variables_target"])
        
        return outputs_all


    def get_data_from_ds(self, ds, ts, variables_dict, global_level_start, global_indices):
        
        sampled_data = {}
        for key, variables in variables_dict.items():
            data_g = []
            for variable in variables:
                data = torch.tensor(ds[variable][ts].values)
                data = data[0] if data.dim() > 1  else data
                data_g.append(data)

            data_g = torch.stack(data_g, dim=-1)

            indices = self.input_layers[key].input_mapping[global_indices // 4**global_level_start]

            data_g = data_g[indices]
            data_g = data_g.view(indices.shape[0], indices.shape[1], -1, len(variables))

            sampled_data[key] = data_g

        return sampled_data        

    def get_grid_mappings(self, mgrid_0_coords):
        
        indices_path = os.path.join(self.model_settings["model_dir"],"indices_data.pickle")

        if not os.path.isfile(indices_path):

            input_mapping, input_in_range = get_nh_variable_mapping_icon(self.model_settings['processing_grid'], ['cell'], 
                                        self.model_settings['input_grid'], self.input_data, 
                                        search_raadius=self.model_settings['search_raadius'], 
                                        max_nh=self.model_settings['nh_input'], 
                                        lowest_level=0,
                                        coords_icon=mgrid_0_coords,
                                        scale_input = self.scale_input,
                                        periodic_fov= self.model_settings['clon_fov'] if ('input_periodicty' in self.model_settings.keys() and self.model_settings['input_periodicty']) else None
                                        )

            output_mapping, output_in_range = get_nh_variable_mapping_icon(self.model_settings['processing_grid'], ['cell'], 
                                        self.model_settings['output_grid'], self.output_data, 
                                        search_raadius=self.model_settings['search_raadius'], 
                                        max_nh=1, 
                                        lowest_level=0,
                                        reverse_last=False,
                                        coords_icon=mgrid_0_coords,
                                        scale_input = self.scale_output,
                                        periodic_fov= self.model_settings['clon_fov'] if ('input_periodicty' in self.model_settings.keys() and self.model_settings['input_periodicty']) else None
                                        )
            
        else:
            with open(indices_path, 'rb') as handle:
                indices_data = pickle.load(handle)
            
            input_mapping = mapping_to_(indices_data['input_mapping'], to='pytorch')
            input_in_range = mapping_to_(indices_data['input_in_range'], to='pytorch')

            output_mapping = mapping_to_(indices_data['output_mapping'], to='pytorch')
            output_in_range = mapping_to_(indices_data['output_in_range'], to='pytorch')

        input_coordinates = {}
        for grid_type in self.input_data.keys():
            input_coordinates[grid_type] = scale_coordinates(get_coords_as_tensor(xr.open_dataset(self.model_settings['input_grid']), grid_type=grid_type), self.scale_input)
            

        output_coordinates = {}
        for grid_type in self.output_data.keys():
            output_coordinates[grid_type] = scale_coordinates(get_coords_as_tensor(xr.open_dataset(self.model_settings['output_grid']), grid_type=grid_type), self.scale_output)

        
        return input_mapping, input_in_range, input_coordinates, output_mapping, output_in_range, output_coordinates



    def init_processing_layer(self, global_level, proc_layer, x, x_att=[]):
        
        # in proc layer: attention to all in x_att
        # Maybe create "status_dict?" -> x with levels
        pass

    def forward_processing_layer(self, global_level, proc_layer, x, x_att=[]):
        
        # in proc layer: attention to all in x_att
        # Maybe create "status_dict?" -> x with levels
        pass


    def get_nh(self, x, indices, global_level, nh_level=None, nh=1):
        if nh_level is None:
            nh_level = global_level

        global_indices, _ ,cells_nh, mask = self.coarsen_indices(global_level, indices=indices['global_cell'], coarsen_level=nh_level, nh=nh)
        x_nh = helpers.get_nh_values(x, indices_nh=cells_nh, sample_indices=indices['sample'], coarsest_level=indices['sample_level'], global_level=global_level)

        global_indices_nh = helpers.get_nh_values(global_indices[:,:,[0]], indices_nh=cells_nh, sample_indices=indices['sample'], coarsest_level=indices['sample_level'], global_level=global_level)

        return x_nh, mask, global_indices, global_indices_nh


    def get_nh_indices(self, global_level, global_cell_indices=None, local_cell_indices=None, adjc_global=None):
        
        if adjc_global is not None:
            adjc_global = self.get_adjacent_global_cell_indices(global_level)

        if global_cell_indices is not None:
            local_cell_indices =  global_cell_indices // 4**global_level

        local_cell_indices_nh, mask = helpers.get_nh_of_batch_indices(local_cell_indices, adjc_global)

        return local_cell_indices_nh, mask



    def get_global_indices_global(self, batch_sample_indices, sampled_level_fov, global_level):

        global_indices_sampled  = self.global_indices.reshape(-1, 4**sampled_level_fov[0])[batch_sample_indices]
        
        return self.get_global_indices_relative(global_indices_sampled, global_level)
    
    def get_global_indices_local(self, batch_sample_indices, sampled_level_fov, global_level):

        global_indices_sampled  = self.global_indices.reshape(-1, 4**sampled_level_fov[0])[batch_sample_indices]
        global_indices_sampled = self.get_global_indices_relative(global_indices_sampled, global_level)    
        return global_indices_sampled // 4**global_level
    
    def get_global_indices_relative(self, sampled_indices, level):
        return sampled_indices.reshape(sampled_indices.shape[0], -1, 4**level)[:,:,0]
    

    def localize_global_indices(self, sample_indices_dict, level):
        
        b,n = sample_indices_dict['global_cell'].shape[:2]
        indices_offset_level = sample_indices_dict['sample']*4**(sample_indices_dict['sample_level']-level)
        indices_level = sample_indices_dict['global_cell'].view(b,n) - indices_offset_level.view(-1,1)

        return indices_level
    
    def coarsen_indices(self, global_level, coarsen_level=None, indices=None, nh=1):
        if indices is None:
            indices = self.global_indices

        global_cells, local_cells, cells_nh, out_of_fov_mask = helpers.coarsen_global_cells(indices, self.eoc, self.acoe, global_level=global_level, coarsen_level=coarsen_level, nh=nh)
        

        return global_cells, local_cells, cells_nh, out_of_fov_mask 
    

    def get_adjacent_global_cell_indices(self, global_level, nh=2):
        adjc, mask = get_adjacent_indices(self.acoe, self.eoc, nh=nh, global_level=global_level)

        return adjc, mask

    def get_coordinates_level(self, global_level):
        indices = self.global_indices.reshape(-1,4**int(global_level))[:,0]
        coords = self.cell_coords_global[:,indices]
        return coords

    def init_input_layers(self, model_dim_var):

        input_layers = nn.ModuleDict()
     
        n_input = len(self.input_data["cell"])

        layer = input_layer(
                input_dim = n_input,
                model_dim = model_dim_var)
        
        input_layers["cell"] = layer

        return input_layers
        
    def init_processing_layer(self):
        pass

    def check_model_dir(self):
      
        self.model_dir = self.model_settings['model_dir']

        self.ckpt_dir = os.path.join(self.model_dir, 'ckpt')

        if 'log_dir' not in self.model_settings.keys():
            self.log_dir = os.path.join(self.model_dir, 'logs')
        else:
            self.log_dir = self.model_settings['log_dir']

        if not os.path.isdir(self.model_dir):
            os.makedirs(self.model_dir)


    def set_training_configuration(self, train_settings=None):
        self.train_settings = load_settings(train_settings, id='train')

        self.train_settings['log_dir'] = self.log_dir

        with open(os.path.join(self.model_dir,'train_settings.json'), 'w') as f:
            json.dump(self.train_settings, f, indent=4)


    def train_(self, train_settings=None, subdir=None, pretrain_subdir=None, optimization=True):

        if train_settings is not None:
            self.set_training_configuration(train_settings)

        if subdir is not None:
            self.model_settings['model_dir'] = os.path.join(self.model_dir, subdir)
            self.check_model_dir()
            self.set_training_configuration(self.train_settings)

        if 'continue_training' in self.train_settings.keys() and self.train_settings['continue_training']:
            self.trained_iterations = self.check_pretrained(self.log_dir)

        train_settings = self.train_settings

        train_settings["variables_source"] = self.model_settings["variables_source"]
        train_settings["variables_target"] = self.model_settings["variables_target"]
        train_settings['model_dir'] = self.model_dir

     
        trainer.train(self, train_settings, self.model_settings)



    def check_pretrained(self, log_dir_check='', strict=True, match_list=None, not_match=None):
        iteration = None

        if len(log_dir_check)>0:
            ckpt_dir = os.path.join(log_dir_check, 'ckpts')
            weights_path = os.path.join(ckpt_dir, 'best.pth')
            if not os.path.isfile(weights_path):
                weights_paths = [f for f in os.listdir(ckpt_dir) if 'pth' in f]
                weights_paths.sort(key=getint)
                if len(weights_path)>0:
                    weights_path = os.path.join(ckpt_dir, weights_paths[-1])
            
            if os.path.isfile(weights_path):
                iteration = self.load_pretrained(weights_path, strict=strict, match_list=match_list, not_match=not_match)
        return iteration

    def load_pretrained(self, ckpt_path:str, strict=True, match_list=None, not_match=None):
        device = 'cpu' if 'device' not in self.model_settings.keys() else self.model_settings['device']
        ckpt_dict = torch.load(ckpt_path, map_location=torch.device(device))
        iteration = load_model(ckpt_dict, self, strict=strict, match_list=match_list, not_match=not_match)
        return iteration



def sequenize(tensor, max_seq_level):
    
    seq_len = tensor.shape[1]
    max_seq_level_seq = int(math.log(seq_len)/math.log(4))
    seq_level = min([max_seq_level_seq, max_seq_level])
    
    if tensor.dim()==3:
        tensor = tensor.view(tensor.shape[0], -1, 4**(seq_level), tensor.shape[-1])
    elif tensor.dim()==2:
        tensor = tensor.view(tensor.shape[0], -1, 4**(seq_level))
    elif tensor.dim()==4:
        tensor = tensor.view(tensor.shape[0], -1, 4**(seq_level), tensor.shape[-2], tensor.shape[-1])

    return tensor
    


def get_nh_indices(adjc_global, global_level, global_cell_indices=None, local_cell_indices=None):
        
    if global_cell_indices is not None:
        local_cell_indices =  global_cell_indices // 4**global_level

    local_cell_indices_nh, mask = helpers.get_nh_of_batch_indices(local_cell_indices, adjc_global)

    return local_cell_indices_nh, mask


def gather_nh_data(x, local_cell_indices_nh, batch_sample_indices, sampled_level, global_level):
    # x in batches sampled from local_cell_indices_nh
    if x.dim()<3:
        x = x.unsqueeze(dim=-1)

    b,n,e = x.shape
    nh = local_cell_indices_nh.shape[-1]

    local_cell_indices_nh_batch = local_cell_indices_nh - (batch_sample_indices*4**(sampled_level - global_level)).view(-1,1,1)

    return torch.gather(x.reshape(b,-1,e),1, index=local_cell_indices_nh_batch.reshape(b,-1,1).repeat(1,1,e)).reshape(b,n,nh,e)

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