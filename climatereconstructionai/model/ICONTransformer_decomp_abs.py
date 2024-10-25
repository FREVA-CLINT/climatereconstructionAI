import json,os
import math
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr

import copy

from ..utils.io import  load_model
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


def get_subdict(adict, keys):
    subdict = {k:adict[k] for k in keys if k in adict}
    return subdict


class grid_layer(nn.Module):
    def __init__(self, global_level, adjc, adjc_mask, coordinates, coord_system="polar", periodic_fov=None) -> None: 
        super().__init__()

        # introduce is_regid
        # if not add learnable parameters? like federkonstante, that are added onto the coords
        # all nodes have offsets, just the ones from the specified are learned
        self.global_level = global_level
        self.coord_system = coord_system
        self.periodic_fov = periodic_fov

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

    def get_nh(self, x, local_indices, sample_dict, relative_coordinates=True, coord_system=None, mask=None):

        indices_nh, adjc_mask = get_nh_indices(self.adjc, local_cell_indices=local_indices, global_level=int(self.global_level))
        
        x = gather_nh_data(x, indices_nh, sample_dict['sample'], sample_dict['sample_level'], int(self.global_level))

        if mask is not None:
            mask = gather_nh_data(mask, indices_nh, sample_dict['sample'], sample_dict['sample_level'], int(self.global_level))
            mask = torch.logical_or(mask.squeeze(dim=-1), adjc_mask)
        else:
            mask = adjc_mask

        if relative_coordinates:
            coords = self.get_relative_coordinates_from_grid_indices(indices_nh, coord_system=coord_system)
        else:
            coords = self.get_coordinates_from_grid_indices(indices_nh)

        return x, mask, coords
    
    def get_nh_indices(self, local_indices):
        return get_nh_indices(self.adjc, local_cell_indices=local_indices, global_level=int(self.global_level))


    def get_coordinates_from_grid_indices(self, local_indices):
        coords = self.coordinates[:, local_indices]
        return coords
    
    def get_relative_coordinates_from_grid_indices(self, local_indices, coords=None, coord_system=None):
        
        if coord_system is None:
            coord_system = self.coord_system
        
        if coords is None:
            coords = self.get_coordinates_from_grid_indices(local_indices)

        coords_rel = get_distance_angle(coords[0,:,:,[0]], coords[1,:,:,[0]], coords[0], coords[1], base=coord_system, periodic_fov=self.periodic_fov)

        return coords_rel
    
    def get_relative_coordinates_cross(self, local_indices, coords, coord_system=None):

        if coord_system is None:
            coord_system = self.coord_system
        
        coords_ref = self.get_coordinates_from_grid_indices(local_indices)

       # if coords.dim()>3:
       #     n_c, b, n, nh = coords.shape
       #     coords = coords.view(n_c, b ,-1)
        if coords_ref.dim()<4:
            coords_ref = coords_ref.unsqueeze(dim=-1)

        coords_rel = get_distance_angle(coords_ref[0,:,:,[0]], coords_ref[1,:,:,[0]], coords[0], coords[1], base=coord_system, periodic_fov=self.periodic_fov) 
        
    #    if coords.dim()>3:
     #       coords_rel = coords_rel.view(n_c, b, n, nh)
        
        return coords_rel

    def get_sections(self, x, local_indices, section_level=1, relative_coordinates=True, return_indices=True, coord_system=None):
        indices = sequenize(local_indices, max_seq_level=section_level)
        x = sequenize(x, max_seq_level=section_level)
        if relative_coordinates:
            coords = self.get_relative_coordinates_from_grid_indices(indices, coord_system=coord_system)
        else:
            coords = self.get_coordinates_from_grid_indices(indices)

        mask = self.fov_mask[indices]

        if return_indices:
            return x, mask, coords, indices
        else:
            return x, mask, coords
    
    def get_projection_cross_vm(self, x, local_indices, sample_dict, coords_cross, sigma_d, kappa_vm=None):
        x_nh, mask_nh, coords_nh = self.get_nh(x, local_indices, sample_dict)

       # x_nh = x_nh[:,:,1:,:]
       # mask_nh = mask_nh[:,:,1:]
       # coords_nh = coords_nh[:,:,:,1:]
        distances_nh, phis_nh = get_relative_positions(coords_cross[:,:,:,0], coords_nh, polar=True)
        distances, phis = get_relative_positions(coords_cross[:,:,:,0], coords_cross, polar=True)

        distances_nh[mask_nh.unsqueeze(dim=-2)]=1e10

 
        weights = get_spatial_projection_weights_vm_dist(phis, distances, phis_nh, kappa_vm, distances_nh, sigma_d)


        x = proj_data(x_nh, weights)

        return x
    
    def get_projection_cross_n(self, x, local_indices, sample_dict, coords_cross, sigma_lon, sigma_lat):
        x_nh, mask_nh, coords_nh = self.get_nh(x, local_indices, sample_dict, relative_coordinates=False)


        d_lon_0, d_lat_0 = get_relative_positions(coords_cross[:,:,:,0], coords_nh, polar=False)
        d_lon, d_lat = get_relative_positions(coords_cross[:,:,:,0], coords_cross, polar=False)

    
 
        weights = get_spatial_projection_weights_n_dist(d_lon, d_lat, d_lon_0, sigma_lon, d_lat_0, sigma_lat)


        x = proj_data(x_nh, weights)

        return x
    
    def get_projection_nh_vm(self, x, local_indices, sample_dict, phi_0, dist_0, sigma_d, kappa_vm):
        x_nh, _, coords_nh = self.get_nh(x, local_indices, sample_dict)
        
        distances, phis = get_relative_positions(coords_nh[:,:,:,0], coords_nh, polar=True)

        weights = get_spatial_projection_weights_vm_dist(phis, distances, phi_0, kappa_vm, dist_0, sigma_d)

        x_nh = proj_data(x_nh, weights)
        
        return x_nh
    
    def get_projection_nh_dist(self, x, local_indices, sample_dict, dist_0, sigma_d):
        x_nh, _, coords_nh = self.get_nh(x, local_indices, sample_dict)
        
        distances, phis = get_relative_positions(coords_nh[:,:,:,0], coords_nh, polar=True)

        weights = get_spatial_projection_weights_dists(distances, dist_0, sigma_d)

        x_nh = proj_data(x_nh, weights)
        
        return x_nh
    
    def get_position_embedding(self, indices, nh: bool, pos_embedder, coord_system, batch_dict=None, section_level=None):
        
        if nh:
            if isinstance(pos_embedder, position_embedder):
                indices = self.get_nh_indices(indices)[0]
                rel_coords = self.get_relative_coordinates_from_grid_indices(indices, coord_system=coord_system)
                pos_embeddings = pos_embedder(rel_coords[0], rel_coords[1])
            else:
                pos_embeddings = pos_embedder(self, indices, batch_dict)
        else:
            if isinstance(pos_embedder, position_embedder):
                indices = sequenize(indices, max_seq_level=section_level)
                coords = self.get_relative_coordinates_from_grid_indices(indices, coord_system=coord_system)
                pos_embeddings = pos_embedder(coords[0], coords[1])
            else:
                pos_embeddings = pos_embedder(self, indices)

        return pos_embeddings
    
class multi_grid_channel_attention(nn.Module):
    def __init__(self, n_grids ,model_hparams, chunks=4, output_reduction=True) -> None: 
        super().__init__()

        #without projection
        #output reduction reduces output to one 

        self.model_dim = model_dim = model_hparams['model_dim']
        ff_dim = model_dim
        n_heads = model_hparams['n_heads']

        model_dim_chunked = n_grids * model_hparams['model_dim']//chunks
        self.chunks = chunks

        if output_reduction:
            self.residual_layer = nn.Linear(n_grids * model_hparams['model_dim'], model_hparams['model_dim'], bias=False)
            model_dim_out_chunked = model_hparams['model_dim']//chunks
        else:
            model_dim_out_chunked = model_dim_chunked
            self.residual_layer = nn.Identity()

        self.model_dim = model_hparams['model_dim']

        self.layer_norms = nn.ModuleList()

        self.sep_lin_projections_out = nn.ModuleList()
        self.mlp_layers = nn.ModuleList()

        self.gammas = nn.ParameterList()
        self.gammas_mlp = nn.ParameterList()
        

        for _ in range(n_grids):
            
            self.layer_norms.append(nn.LayerNorm(model_dim, elementwise_affine=True))

        n_grids_out = 1 if output_reduction else n_grids

        for _ in range(n_grids_out):
            self.mlp_layers.append(nn.Sequential(          
                                        nn.LayerNorm(model_dim, elementwise_affine=True),
                                        nn.Linear(model_dim, ff_dim, bias=False),
                                        nn.SiLU(),
                                        nn.Linear(ff_dim, model_dim, bias=False)
                                        ))
            self.gammas.append(torch.nn.Parameter(torch.ones(model_dim) * 1e-6))
            self.gammas_mlp.append(torch.nn.Parameter(torch.ones(model_dim) * 1e-6))

        
        self.mha_attention = helpers.MultiHeadAttentionBlock(
            model_dim, model_dim_out_chunked, n_heads, input_dim=model_dim_chunked, qkv_proj=True
            )           



    def forward(self, x_levels):
        
        xs = []
        for i, x in enumerate(x_levels):
            xs.append(self.layer_norms[i](x))

        x = torch.concat(xs,dim=-1)

        b,n,f = x.shape
        x = x.view(b, n, self.chunks, -1)
        x = x.view(b*n, self.chunks, -1)

        att_out = self.mha_attention(q=x, k=x, v=x) 

        att_out = att_out.view(b,n,-1)

        outputs = att_out.split(self.model_dim, dim=-1)

        x_res = self.residual_layer(torch.concat(x_levels, dim=-1)).split(self.model_dim, dim=-1)

        x_output = []
        for i, x in enumerate(x_res):
        
            x = x + self.gammas[i] * outputs[i]
            x = x + self.gammas_mlp[i] * self.mlp_layers[i](x)

            x_output.append(x)


        return x_output


class multi_grid_attention_cross(nn.Module):
    def __init__(self, grid_layers, grid_layer_cross, model_hparams, nh_attention=True, continous_pos_embedding=True) -> None: 
        super().__init__()
        
        self.n_grids = len(grid_layers)
        self.model_dim = model_dim = model_hparams['model_dim']
        ff_dim = model_dim
        n_heads = model_hparams['n_heads']
        embedding_dim = model_hparams['pos_emb_dim']

        self.grid_layer_cross = grid_layer_cross
        self.grid_level_cross = grid_layer_cross.global_level
        self.grid_levels = [grid_layer.global_level for grid_layer in grid_layers.values()]

        self.nh_attention = nh_attention

        model_dim = model_hparams['model_dim']
        pos_emb_calc = model_hparams['pos_emb_calc']
        emb_table_bins = model_hparams['emb_table_bins']

        if 'cartesian' in pos_emb_calc:
            self.coord_system = 'cartesian'
        else:
            self.coord_system = 'polar'

        self.max_seq_level = model_hparams['max_seq_level']

        self.continous_pos_embedding=continous_pos_embedding

        if continous_pos_embedding:
            self.position_embedder_cross = position_embedder(0,0, emb_table_bins, model_dim, pos_emb_calc=pos_emb_calc)
        else:
            if nh_attention:
                self.position_embedder_cross = nh_pos_embedding(grid_layer_cross, model_hparams['nh'], model_dim)
            else:
                self.position_embedder_cross = seq_grid_embedding2(grid_layer_cross, 4, model_hparams['max_seq_level'], model_dim, constant_init=False)


        self.kv_projection = nn.Linear(model_dim, model_dim, bias=False)
        self.layer_norm_kv = nn.LayerNorm(model_dim, elementwise_affine=True)
        self.embedding_layer_cross = nn.Linear(model_dim, model_dim*2)

        self.layer_normsq = nn.ModuleList()
        self.q_projections = nn.ModuleList()
        self.position_embedders = nn.ModuleList()
        self.embedding_layers = nn.ModuleList()
        self.output_projections = nn.ModuleList()
        
        self.mlp_layers = nn.ModuleList()
        self.gammas = nn.ParameterList()
        self.gammas_mlp = nn.ParameterList()


        for grid_layer in grid_layers:

            if continous_pos_embedding:
                self.position_embedder = position_embedder(0,0, emb_table_bins, model_dim, pos_emb_calc=pos_emb_calc)
            else:
                self.position_embedder = seq_grid_embedding2(grid_layer, 4, model_hparams['max_seq_level'], model_dim, constant_init=False)
            
            self.layer_normsq.append(nn.LayerNorm(model_dim, elementwise_affine=True))

            self.q_projections.append(nn.Linear(model_dim, model_dim*2, bias=False))
            self.output_projections.append(nn.Linear(model_dim, model_dim*2, bias=False))

            self.embedding_layers.append(nn.Linear(model_dim, model_dim*2))

            self.mlp_layers.append(nn.Sequential(nn.LayerNorm(model_dim, elementwise_affine=True),
                                        nn.Linear(model_dim, ff_dim, bias=False),
                                        nn.SiLU(),
                                        nn.Linear(ff_dim, model_dim, bias=False)))
        
        
            self.gammas.append(torch.nn.Parameter(torch.ones(self.n_grids * model_dim) * 1e-6))
            self.gammas_mlp.append(torch.nn.Parameter(torch.ones(self.n_grids * model_dim) * 1e-6))

        self.mha_attention = helpers.MultiHeadAttentionBlock(
            model_dim, model_dim, n_heads, input_dim=model_dim, qkv_proj=False, v_proj=False
            )           


    def forward(self, x, x_levels, drop_mask_cross, indices_grid_layers, batch_dict):
        

        qs = []

        x_nh, mask , _ = self.grid_layer_cross.get_nh(x, indices_grid_layers[int(self.grid_level_cross)], batch_dict)
        if drop_mask_cross is not None:
            drop_mask_cross = self.grid_layer_cross.get_nh(drop_mask_cross, indices_grid_layers[int(self.grid_level_cross)], batch_dict)[0]
            mask = torch.logical_or(mask, drop_mask_cross)

        pos_embeddings_nh = self.grid_layer_cross.get_position_embedding(indices_grid_layers[int(self.grid_level_cross)], 
                                                                           True, 
                                                                           self.position_embedder_cross, 
                                                                           self.coord_system, 
                                                                           batch_dict=batch_dict, 
                                                                           section_level=self.max_seq_level)

        shift, scale = self.embedding_layer_cross(pos_embeddings_nh).chunk(2, dim=-1)

        x_nh = self.layer_norm_kv(x) * (scale + 1) + shift

        x_nh = self.kv_projection(x_nh)


        x_levels_output = []
        lens = []

        for i, x in enumerate(x_levels):
            
            grid_level = self.grid_levels[i]

            b,n,f = x.shape
            

            x = x.view(b,-1,f)
            
            # use nh instead of sequence?
     
            x_levels_output.append(x)
            x = sequenize(x, max_seq_level=self.grid_level_cross-grid_level)

            pos_embeddings = self.grid_layer_cross.get_position_embedding(indices_grid_layers[int(grid_level)], 
                                                                           False, 
                                                                           self.position_embedders[i], 
                                                                           self.coord_system, 
                                                                           batch_dict=batch_dict, 
                                                                           section_level=self.grid_level_cross-grid_level)
            
            shift, scale = self.embedding_layers[i](pos_embeddings).chunk(2, dim=-1)

            x = self.layer_normsq[i](x) * (scale + 1) + shift

            b,n,nh,f = x.shape
           
            q = self.q_projections[i](x).chunk(2, dim=-1)

            qs.append(q)
            lens.append(q.shape[1])
   
        q = torch.concat(qs, dim=1)
   
        att_out, att = self.mha_attention(q=q, k=x_nh, v=x_nh, return_debug=True, mask=mask)     

        outputs = att_out.split(lens, dim=-1)

        for i, x in enumerate(x_levels_output):
          
            output = self.output_projections[i](outputs[i]).view(x.shape)
        
            x = x + self.gammas[i] * output
            x = x + self.gammas_mlp[i] * self.mlp_layers[i](x)

            x_levels_output[i] = x

        return x_levels_output

class multi_grid_projection(nn.Module):
    def __init__(self, projection_level, cross_levels, grid_layers, model_hparams, nh_attention=False, projection_mode='') -> None: 
        super().__init__()
        
        # with interpolation to lowest grid

        self.model_dim = model_dim = model_hparams['model_dim']
        ff_dim = model_dim
        n_heads = model_hparams['n_heads']
        embedding_dim = model_hparams['pos_emb_dim']
        #projection_mode = model_hparams['mga_projection_mode']

        self.grid_layers = grid_layers
        self.projection_level = projection_level
        self.projection_grid_layer = grid_layers[projection_level]
        self.grid_levels = [grid_layer.global_level for grid_layer in grid_layers.values()]

        self.nh_attention = nh_attention

        
        self.max_seq_level = model_hparams['max_seq_level']
       
       
        self.projection_layers = nn.ModuleList()
        self.mlp_layers = nn.ModuleList()

        self.gammas = nn.ParameterList()
        self.gammas_mlp = nn.ParameterList()
        
        for k, cross_level in enumerate(cross_levels):
            cross_level = str(cross_level)

            if cross_level != projection_level:
                self.projection_layers.append(get_projection_layer(projection_mode, model_hparams, self.projection_grid_layer))
            else:
                self.projection_layers.append(nn.Identity())
                   


    def forward(self, x_levels, drop_masks_level, indices_grid_layers, batch_dict):
        
        
        b,n_proj,f = x_levels[int(self.projection_level)][-1].shape

        x_levels_output = {int(self.projection_level): []}
        drop_masks_output = {}
        i = 0

        drop_mask_proj_level = drop_masks_level[int(self.projection_level)]
        for global_level, x_ in x_levels.items():
            for x in x_:
                drop_mask_level = drop_masks_level[global_level]

                b,n,f = x.shape
                
                if n_proj//n > 1:
                    if isinstance(self.projection_layers[i], nn.Identity):
                        x = x.unsqueeze(dim=-2).repeat_interleave(n_proj//n, dim=-2)
                    else:
                        x, drop_mask_level = self.projection_layers[i](x, 
                                                        grid_layer=self.grid_layers[str(int(global_level))], 
                                                        grid_layer_out=self.grid_layers[str(int(self.projection_level))], 
                                                        indices_layer=indices_grid_layers[int(global_level)],
                                                        indices_layer_out = indices_grid_layers[int(self.projection_level)],
                                                        sample_dict=batch_dict,
                                                        nh_projection=True,
                                                        mask=drop_mask_level)
                        if drop_mask_proj_level is not None:
                            drop_mask_proj_level = torch.logical_and(drop_mask_level.view(drop_mask_proj_level.shape), drop_mask_proj_level)
                        elif drop_mask_level is not None:
                            drop_mask_proj_level = drop_mask_level.view(x.shape[0],-1,1)
                x = x.view(b,-1,f)
                
                x_levels_output[int(self.projection_level)].append(x)
                drop_masks_output[int(self.projection_level)] = (drop_mask_proj_level)

                i+=1
 
        return x_levels_output, drop_masks_output

class multi_grid_spatial_attention_ds(nn.Module):
    def __init__(self, projection_level, cross_levels, grid_layers, model_hparams, nh_attention=False, input_aggregation='concat', output_projection_overlap=False, projection_mode='', continous_pos_embedding=True) -> None: 
        super().__init__()
        
        # with interpolation to lowest grid

        self.model_dim = model_dim = model_hparams['model_dim']
        ff_dim = model_dim
        n_heads = model_hparams['n_heads']
        embedding_dim = model_hparams['pos_emb_dim']
        #projection_mode = model_hparams['mga_projection_mode']

        self.grid_layers = grid_layers
        self.projection_level = projection_level
        self.projection_grid_layer = grid_layers[projection_level]
        self.grid_levels = [grid_layer.global_level for grid_layer in grid_layers.values()]

        self.nh_attention = nh_attention

        self.input_aggregation = input_aggregation
        self.output_projection_overlap = output_projection_overlap

        if self.input_aggregation=='sum':
            model_dim_agg = model_dim

        elif self.input_aggregation=='concat':
            model_dim_agg = model_dim*len(cross_levels)
        
        self.model_dim_agg = model_dim_agg

        if output_projection_overlap:
            self.shared_lin_projection_out = nn.Linear(model_dim_agg, model_dim_agg, bias=False)
        else:
            self.shared_lin_projection_out = nn.Identity()

        self.max_seq_level = model_hparams['max_seq_level']

        self.continous_pos_embedding=continous_pos_embedding

        if continous_pos_embedding:
            model_dim = model_hparams['model_dim']
            pos_emb_calc = model_hparams['pos_emb_calc']
            emb_table_bins = model_hparams['emb_table_bins']

            if 'cartesian' in pos_emb_calc:
                self.coord_system = 'cartesian'
            else:
                self.coord_system = 'polar'
        
            self.position_embedder = position_embedder(0,0, emb_table_bins, model_dim, pos_emb_calc=pos_emb_calc)
        else:
            if nh_attention:
                self.position_embedder = nh_pos_embedding(self.projection_grid_layer, model_hparams['nh'], model_dim)
            else:
                self.position_embedder = seq_grid_embedding2(self.projection_grid_layer, 4, model_hparams['max_seq_level'], model_dim, constant_init=False)

       
        self.projection_layers = nn.ModuleList()

        self.layer_norms = nn.ModuleList()

        self.kv_projections = nn.ModuleList()
        self.q_projections = nn.ModuleList()

        self.embedding_layers = nn.ModuleList()
        self.sep_lin_projections_out = nn.ModuleList()

        self.mlp_layers = nn.ModuleList()

        self.gammas = nn.ParameterList()
        self.gammas_mlp = nn.ParameterList()
        
        for k, cross_level in enumerate(cross_levels):
            cross_level = str(cross_level)

            if cross_level != projection_level:
                self.projection_layers.append(get_projection_layer(projection_mode, model_hparams, self.projection_grid_layer))
            else:
                self.projection_layers.append(nn.Identity())
                    
            self.layer_norms.append(nn.LayerNorm(model_dim, elementwise_affine=True))

            self.kv_projections.append(nn.Linear(model_dim, model_dim*2, bias=False))

            self.q_projections.append(nn.Linear(model_dim, model_dim, bias=False))

            self.embedding_layers.append(nn.Linear(model_dim, model_dim*2))

            if not output_projection_overlap:
                self.sep_lin_projections_out.append(nn.Linear(model_dim, model_dim, bias=False))
            else:
                self.sep_lin_projections_out.append(nn.Identity())

            self.gammas.append(torch.nn.Parameter(torch.ones(model_dim) * 1e-6))
            self.gammas_mlp.append(torch.nn.Parameter(torch.ones(model_dim) * 1e-6))

            self.mlp_layers.append(nn.Sequential(          
                                        nn.LayerNorm(model_dim, elementwise_affine=True),
                                        nn.Linear(model_dim, ff_dim, bias=False),
                                        nn.SiLU(),
                                        nn.Linear(ff_dim, model_dim, bias=False)
                                        ))
        
        
        self.mha_attention = helpers.MultiHeadAttentionBlock(
            model_dim_agg, model_dim_agg, n_heads, input_dim=model_dim_agg, qkv_proj=False, v_proj=False
            )           


    def forward(self, x_levels, drop_masks_level, indices_grid_layers, batch_dict):
        

        qs = []
        ks = []
        vs = [] 
        
        b,n_proj,f = x_levels[int(self.projection_level)][-1].shape

        pos_embeddings = self.projection_grid_layer.get_position_embedding(indices_grid_layers[int(self.projection_level)], 
                                                                           self.nh_attention, 
                                                                           self.position_embedder, 
                                                                           self.coord_system, 
                                                                           batch_dict=batch_dict, 
                                                                           section_level=self.max_seq_level)
        x_levels_output = {int(self.projection_level): []}
        drop_masks_output = {}
        i = 0

        drop_mask_proj_level = drop_masks_level[int(self.projection_level)]
        for global_level, x_ in x_levels.items():
            drop_mask_level = drop_masks_level[global_level]
            for x in x_:

                b,n,f = x.shape
                
                if n_proj//n > 1:
                    if isinstance(self.projection_layers[i], nn.Identity):
                        x = x.unsqueeze(dim=-2).repeat_interleave(n_proj//n, dim=-2)
                    else:
                        x, drop_mask_level = self.projection_layers[i](x, 
                                                      grid_layer=self.grid_layers[str(int(global_level))], 
                                                      grid_layer_out=self.grid_layers[str(int(self.projection_level))], 
                                                      indices_layer=indices_grid_layers[int(global_level)],
                                                      indices_layer_out = indices_grid_layers[int(self.projection_level)],
                                                      sample_dict=batch_dict,
                                                      nh_projection=True,
                                                      mask=drop_mask_level)
                        if drop_mask_proj_level is not None:
                            drop_mask_proj_level = torch.logical_and(drop_mask_level.view(drop_mask_proj_level.shape), drop_mask_proj_level)
                        else:
                            drop_mask_proj_level = drop_mask_level.view(x.shape[0],-1,1)

                x = x.view(b,-1,f)
                
                # use nh instead of sequence?
                if self.nh_attention:
                    x, mask, _ = self.projection_grid_layer.get_nh(x, indices_grid_layers[int(self.projection_level)], batch_dict, mask=drop_mask_proj_level)
                    x_levels_output[int(self.projection_level)].append(x[:,:,0])
                    if drop_mask_proj_level is not None:
                        mask_update = mask.clone()
                        mask_update = mask_update.sum(dim=-1)==mask_update.shape[-1]
                else:
                    x_levels_output[int(self.projection_level)].append(x)

                    x = sequenize(x, max_seq_level=self.max_seq_level)

                    if drop_mask_proj_level is not None:
                        mask = sequenize(drop_mask_proj_level, max_seq_level=self.max_seq_level)
                        mask_update = mask.clone()
                        mask_update[mask_update.sum(dim=-1)!=mask_update.shape[-1]]=False
                        mask_update = mask_update.view(drop_mask_proj_level.shape)
                    else:
                        mask_update=mask=None

                    drop_masks_output[int(self.projection_level)] = mask_update

                shift, scale = self.embedding_layers[i](pos_embeddings).chunk(2, dim=-1)

                x = self.layer_norms[i](x) * (scale + 1) + shift

                b,n,nh,f = x.shape

                x = x.view(b*n,nh,f)

                if mask is not None:
                    mask = mask.view(b*n,nh)

                if self.nh_attention:
                    q = x[:,[0]]
                    kv = x
                else:
                    q = kv = x

                kv = self.kv_projections[i](kv).chunk(2, dim=-1)
                q = self.q_projections[i](q)

                qs.append(q)
                ks.append(kv[0])
                vs.append(kv[1])

                i+=1

        if self.input_aggregation=='concat':
            q = torch.concat(qs, dim=-1)
            k = torch.concat(ks, dim=-1)
            v = torch.concat(vs, dim=-1)

        elif self.input_aggregation=='sum':
            q = torch.stack(qs, dim=-1).sum(dim=-1)
            k = torch.stack(ks, dim=-1).sum(dim=-1)
            v = torch.stack(vs, dim=-1).sum(dim=-1)

        
        att_out, att = self.mha_attention(q=q, k=k, v=v, return_debug=True, mask=mask) 

        outputs = self.shared_lin_projection_out(att_out)

        outputs = outputs.split(self.model_dim, dim=-1)


        for i, x in enumerate(x_levels_output[int(self.projection_level)]):
          
            output = outputs[0] if self.input_aggregation=='sum' else outputs[i]
            output = self.sep_lin_projections_out[i](output).view(b,n_proj,-1)
        
            x = x + self.gammas[i] * output
            x = x + self.gammas_mlp[i] * self.mlp_layers[i](x)

            x_levels_output[int(self.projection_level)][i] = x

        return x_levels_output, drop_masks_output
    



class input_projection_layer(nn.Module):
    def __init__(self, mapping, in_range_mask, coordinates, projection_grid_layer: grid_layer, model_hparams, projection_mode='learned_cont') -> None: 
        super().__init__()
        
        mask_out_of_range = model_hparams['mask_out_of_range'] if 'mask_out_of_range' in model_hparams.keys() else False

        self.register_buffer("mapping", mapping, persistent=False)
        self.register_buffer("coordinates", coordinates, persistent=False)
        
        if mask_out_of_range:
            self.register_buffer("out_of_range_mask", ~in_range_mask.bool().squeeze(), persistent=False)
        else:
            self.register_buffer("out_of_range_mask", torch.ones_like(in_range_mask.squeeze(), dtype=bool), persistent=False)
            
        model_dim = model_hparams['model_dim']
        pos_emb_calc = model_hparams['pos_emb_calc']
        emb_table_bins = model_hparams['emb_table_bins']

        if 'cartesian' in pos_emb_calc:
            self.coord_system = 'cartesian'
        else:
            self.coord_system = 'polar'

        model_dim = model_hparams['model_dim']

        if int(projection_grid_layer.global_level)>0:
            self.projection_layer = get_projection_layer(projection_mode, model_hparams, projection_grid_layer)
        else:
            self.projection_layer=nn.Identity()

        self.projection_grid_layer = projection_grid_layer
        self.input_seq_level = model_hparams['input_seq_level']
        
        self.lin_projection = nn.Linear(len(model_hparams['variables_source']['cell']), model_dim) 
        

    def forward(self, x, indices_grid_layer0, indices_grid_layer, drop_mask=None):    
        
        if indices_grid_layer.dim()<3:
            indices_grid_layer = indices_grid_layer.unsqueeze(dim=-1)

        f_in = x.shape[-1]
        b, n_out,_ = indices_grid_layer.shape

        x = x.view(b, -1, f_in)

        x = self.lin_projection(x)

        #x = sequenize(x, max_seq_level=self.projection_grid_layer.global_level)
        #indices0 = sequenize(indices_grid_layer0, max_seq_level=self.projection_grid_layer.global_level)
        #drop_mask = sequenize(drop_mask, max_seq_level=self.projection_grid_layer.global_level)

        drop_mask = torch.logical_or(self.out_of_range_mask[indices_grid_layer0], drop_mask)

        if not isinstance(self.projection_layer, nn.Identity):
            coords_input = self.coordinates[:,self.mapping[indices_grid_layer0]]

            n = x.shape[1]
            coords_input = coords_input.view(2,b,n,-1)

            x, drop_mask = self.projection_layer(x, coordinates=coords_input, grid_layer_out=self.projection_grid_layer, indices_layer_out=indices_grid_layer, mask=drop_mask, nh_projection=False)

        x = x.view(b, n_out, -1)
        x = {int(self.projection_grid_layer.global_level): x}

        return x, drop_mask


class shifting_cross_layer(nn.Module):
    def __init__(self, global_levels, grid_layers, model_hparams, grid_window=1) -> None: 
        super().__init__()
        
        self.grid_window = grid_window
        self.grid_layers = grid_layers
        self.x_level_indices = []
        self.gobal_levels = global_levels
        
        self.mga_layers_levels = nn.ModuleDict()
    
        self.global_levels_windows = []
  
        for idx, global_level in enumerate(global_levels):

            global_levels_window = [idx+k for k in range(grid_window)]
            self.global_levels_windows.append(global_levels_window)

            grid_layers_keys = [str(global_levels[k]) for k in global_levels_window]
            grid_layers_ = dict(zip(grid_layers_keys, [grid_layers[key] for key in grid_layers_keys]))

            self.mga_layers_levels[str(global_level)] = nn.ModuleList(
                [multi_grid_attention_cross(grid_layers_, grid_layers[str(global_level)], model_hparams, continous_pos_embedding=False) for _ in range(model_hparams['n_processing_layers'])]
                )

    def forward(self, x_levels, indices_levels, batch_dict):
        # how to deal with overlapping layers? average and project back in mga layers?
        n = 0
        x_levels_output = []
        for level_idx, x_cross in enumerate(x_levels):
            
            x = [x_levels[idx] for idx in self.global_levels_windows[level_idx]]
            
            for layer in self.mga_layers_levels[str(self.gobal_levels[level_idx])]:
                x = layer(x_cross, x, None, indices_levels, batch_dict)

            x_levels_output += x

            n += 1
        
        return x_levels_output
    

class multi_grid_layer(nn.Module):
    def __init__(self, global_levels, grid_layers, model_hparams, input_aggregation='concat', output_projection_overlap=False, nh_attention=False, projection_mode='', reduction_mode=None, with_spatial_attention=True, cascading=True) -> None: 
        super().__init__()
        
        mode = 'simultaneous' # cascading increases the window size as going higher in resolution
        inter_level_reduction = True
        self.inter_level_reduction = inter_level_reduction

        if cascading:
            n_layers = len(global_levels)-1
            step = 1 
        else:
            n_layers = 1
            step = len(global_levels)
        
        self.reduction_mode = reduction_mode

        self.grid_layers = grid_layers
        self.x_level_indices = []
        
        self.inter_mgaca_layers_levels = nn.ModuleDict()   

        self.mga_layers_levels = nn.ModuleDict()      
        self.global_levels_cross = []

        global_levels_processed = global_levels.clone()

        for k in range(n_layers):

            initial_layer = k if inter_level_reduction else 0
            global_levels_cross = global_levels_processed[initial_layer:(k+step+1)].tolist()
            self.global_levels_cross.append(global_levels_cross)

            global_level_projection = str(global_levels_cross[-1])

            if with_spatial_attention:
                self.mga_layers_levels[global_level_projection] = multi_grid_spatial_attention_ds(global_level_projection,
                                                    global_levels_cross,
                                                    grid_layers, 
                                                    model_hparams, 
                                                    input_aggregation=input_aggregation, 
                                                    output_projection_overlap=output_projection_overlap, 
                                                    nh_attention=nh_attention, 
                                                    projection_mode=projection_mode)
            else:
                self.mga_layers_levels[global_level_projection] = multi_grid_projection(global_level_projection,
                                                    global_levels_cross,
                                                    grid_layers, 
                                                    model_hparams,
                                                    projection_mode=projection_mode)

            global_levels_processed = global_levels_processed.clamp(max=int(global_level_projection))

            if len(global_levels_cross)>1 and inter_level_reduction:
                if reduction_mode == 'channel_attention':
                    self.inter_mgaca_layers_levels[global_level_projection] = multi_grid_channel_attention(len(global_levels_cross), model_hparams, chunks=4, output_reduction=True)

        if not inter_level_reduction:
            if reduction_mode == 'channel_attention':
                self.output_mgaca_layers_levels = multi_grid_channel_attention(len(global_levels_cross), model_hparams, chunks=4, output_reduction=True)

    def forward(self, x_levels, drop_mask_levels, indices_levels, batch_dict):
        
        x_levels_output = {}
        drop_mask_levels_output = {}

        for n, global_levels in enumerate(self.global_levels_cross):
            
            x_levels_output = {**get_subdict(x_levels, global_levels),**x_levels_output}
      
            mga_layer = self.mga_layers_levels[str(global_levels[-1])]

            x_levels_output, drop_mask_levels_output = mga_layer(x_levels_output, drop_mask_levels, indices_levels, batch_dict)
            
            drop_mask_levels.update(drop_mask_levels_output)

            if str(global_levels[-1]) in self.inter_mgaca_layers_levels.keys():
                x_levels_output = {global_levels[-1]: [self.inter_mgaca_layers_levels[str(global_levels[-1])](x_levels_output[global_levels[-1]])[0]]}

            elif self.inter_level_reduction and self.reduction_mode == 'sum':
                x_levels_output = {global_levels[-1]: [torch.stack(x_levels_output[global_levels[-1]], dim=-1).sum(dim=-1)]}

            n += 1

        if not self.inter_level_reduction:
            if self.reduction_mode == 'channel_attention':
                x = self.output_mgaca_layers_levels(x_levels_output[global_levels[-1]])[0]
            elif self.reduction_mode == None:
                x = torch.stack(x_levels_output[global_levels[-1]], dim=-1).sum(dim=-1)
        else:
            x = x_levels_output[global_levels[-1]][0]

        return {global_levels[-1]: x}, drop_mask_levels




class position_embedder(nn.Module):
    def __init__(self, min_dist, max_dist, emb_table_bins, emb_dim, pos_emb_calc="polar", phi_table=None) -> None: 
        super().__init__()
        self.pos_emb_calc = pos_emb_calc

        self.operation = None
        self.transform = None
        self.proj_layer = None
        self.cartesian = False

        if "descrete" in pos_emb_calc and "polar" in pos_emb_calc:
            self.pos1_emb = helpers.PositionEmbedder_phys_log(min_dist, max_dist, emb_table_bins, n_heads=emb_dim)
            if phi_table is not None:
                self.pos2_emb = phi_table
            else:
                self.pos2_emb = helpers.PositionEmbedder_phys(-torch.pi, torch.pi, emb_table_bins, n_heads=emb_dim, special_token=True)

        if "semi" in pos_emb_calc and "polar" in pos_emb_calc:
            self.pos1_emb = nn.Sequential(nn.Linear(1, emb_dim), nn.SiLU())
            if phi_table is not None:
                self.pos2_emb = phi_table
            else:
                self.pos2_emb = helpers.PositionEmbedder_phys(-torch.pi, torch.pi, emb_table_bins, n_heads=emb_dim, special_token=True)

        if "cartesian" in pos_emb_calc:
            self.proj_layer = nn.Sequential(nn.Linear(2, emb_dim, bias=True),
                                        nn.SiLU(),
                                        nn.Linear(emb_dim, emb_dim, bias=False),
                                        nn.Sigmoid())
            
            self.cartesian = True


        if "learned" in pos_emb_calc and "polar" in pos_emb_calc:
            self.proj_layer = nn.Sequential(nn.Linear(2*emb_dim, emb_dim, bias=True),
                                        nn.SiLU(),
                                        nn.Linear(emb_dim, emb_dim, bias=False),
                                        nn.Sigmoid())
        self.km_transform = False
        if 'km' in pos_emb_calc:
            self.km_transform = True
        
        
        if 'inverse' in pos_emb_calc:
            self.transform = helpers.conv_coordinates_inv

        elif 'sig_log' in pos_emb_calc:
            self.transform = helpers.conv_coordinates_sig_log

        elif 'sig_inv_log' in pos_emb_calc:
            self.transform = helpers.conv_coordinates_inv_sig_log  

        elif 'log' in pos_emb_calc:
            self.transform = helpers.conv_coordinates_log
       
        if 'sum' in pos_emb_calc:
            self.operation = 'sum'

        elif 'product' in pos_emb_calc:
            self.operation = 'product'


    def forward(self, pos1, pos2):
        if self.cartesian:
            if self.km_transform:
                pos1 = pos1*6371.
                pos2 = pos2*6371.
                pos1[pos1.abs()<0.01]=0
                pos2[pos2.abs()<0.01]=0
            else:
                pos1[pos1.abs()<1e-6]=0
                pos2[pos2.abs()<1e-6]=0

            if self.transform is not None:
                pos1 = self.transform(pos1)
                pos2 = self.transform(pos2)
            
            return 16*self.proj_layer(torch.stack((pos1, pos2), dim=-1))    

        else:
            if self.km_transform:
                pos1 = pos1*6371.
                dist_0 = pos1 < 0.01
            else:
                dist_0 = pos1 < 1e-6

            if self.transform is not None:
                pos1 = self.transform(pos1)
            
            if isinstance(self.pos1_emb, nn.Sequential):
                pos1 = pos1.unsqueeze(dim=-1)

            pos1_emb = self.pos1_emb(pos1)
            pos2_emb = self.pos2_emb(pos2, special_token_mask=dist_0)

            if self.proj_layer is not None:
                return 16*self.proj_layer(torch.concat((pos1_emb, pos2_emb), dim=-1))
                        
            if self.operation == 'sum':
                return pos1_emb + pos2_emb
            
            elif self.operation == 'product':
                return pos1_emb * pos2_emb


def proj_data(data, weights):


    if weights.dim() - data.dim() == 2:
        data = data.unsqueeze(dim=2).unsqueeze(dim=2)

    projection = weights * data
    return projection.sum(dim=-2)


def get_spatial_projection_weights_dists(dists, dists_0, sigma):

    dist_weights = normal_dist(dists, sigma, dists_0)
    
    weights = F.softmax(dist_weights, dim=-1)

    #return weights/(weights_norm+1e-10)
    return weights

def get_spatial_projection_weights_n_dist(d_lons, d_lats, dlon_0, sigma_lon, dlat_0, sigma_lat):

    lon_weights = normal_dist(d_lons, sigma_lon, dlon_0)
    lat_weights = normal_dist(d_lats, sigma_lat, dlat_0)
    
    weights = lon_weights * lat_weights

    weights = F.softmax(weights, dim=-2)

    #return weights/(weights_norm+1e-10)
    return weights

def get_spatial_projection_weights_vm_dist(phis, dists, phi_0, kappa_vm, dists_0, sigma, mask=None):

    vm_weights = von_mises(phis, kappa_vm, phi_0)
    dist_weights = normal_dist(dists, sigma, dists_0)
    
    vm_weights[dist_weights[:,:,:,:,0]==1] = torch.exp(kappa_vm)

    weights = vm_weights * dist_weights

    if mask is not None:
        weights = weights.transpose(2,3)
        weights[mask] = -1e30 if weights.dtype == torch.float32 else -1e4
        weights = weights.transpose(2,3)

    weights = F.softmax(weights, dim=-2)

    return weights

def cosine(thetas, wavelengths, distances, theta_offsets=None):

    freq = 2*torch.pi/wavelengths

    if theta_offsets is not None:
        Z = torch.cos(freq * (torch.cos(thetas.unsqueeze(dim=-1)-theta_offsets)*distances.unsqueeze(dim=-1)).unsqueeze(dim=-1))
    else:
        Z = torch.cos(freq * distances)

    return Z


def von_mises(thetas, kappa, theta_offsets=None):

    if theta_offsets is not None:
        if not torch.is_tensor(theta_offsets):
            theta_offsets = torch.tensor(theta_offsets)
        vm = torch.exp(kappa * torch.cos(thetas.unsqueeze(dim=-1) - theta_offsets.unsqueeze(dim=-2)).unsqueeze(dim=-1))
    else:
        vm = torch.exp(kappa * torch.cos(thetas.unsqueeze(dim=-1)).unsqueeze(dim=-1))

    return vm


def normal_dist(distances, sigma, distances_offsets=None, sigma_cross=True):

    #if sigma.dim()==3 and sigma_cross:
    #    sigma = sigma.unsqueeze(dim=-1).unsqueeze(dim=-1)

    if distances_offsets is not None:
        if not torch.is_tensor(distances_offsets):
            distances_offsets = torch.tensor(distances_offsets)

        diff = distances.unsqueeze(dim=-1) - distances_offsets.unsqueeze(dim=-2)
    else:
        diff = distances
    
    if sigma_cross: 
        diff = diff.unsqueeze(dim=-1)

    nd = torch.exp(-0.5 * (diff / sigma) ** 2)

    return nd

class angular_embedder(nn.Module):
    def __init__(self, n_bins, emb_dim) -> None: 
        super().__init__()
 
        self.thata_embedder = helpers.PositionEmbedder_phys(-torch.pi, torch.pi, n_bins, n_heads=emb_dim, special_token=True)

    def forward(self, thetas, dist_0_mask):
        return  self.thata_embedder(thetas, special_token_mask=dist_0_mask)

class decomp_layer_diff_(nn.Module):
    def __init__(self, global_levels, grid_layers: dict, model_hparams, aggregation_fcn='n') -> None: 
        super().__init__()
        # not learned

        self.grid_layers = grid_layers
        
        self.max_level = list(grid_layers.keys())[-1]
        self.global_levels = global_levels.sort().values
        self.global_level_diff = self.global_levels.diff()

        self.aggregation_layers = nn.ModuleDict()

        for global_level in global_levels[:-1]:
            global_level = str(int(global_level))
            
            self.aggregation_layers[global_level] = (get_projection_layer(aggregation_fcn, model_hparams, grid_layers[global_level]))
            
                

            

    def forward(self, x, indices_layers, drop_mask=None):   
        
        x_levels={}
        drop_masks_level = {}
        #from fine to coarse

        for k, global_level in enumerate(self.global_levels):
            global_level = int(global_level)
            global_level_prior = int(self.global_levels[k-1])
            
            if k == 0:
                x = x[global_level]
                x_levels[global_level] = [x]
                drop_masks_level[global_level] = drop_mask
            
            else:
                x, drop_mask = self.aggregation_layers[str(global_level)](x, 
                                                                        indices_layer=indices_layers[global_level_prior], 
                                                                        grid_layer=self.grid_layers[str(global_level_prior)], 
                                                                        indices_layer_out=indices_layers[global_level], 
                                                                        grid_layer_out=self.grid_layers[str(global_level)],
                                                                        mask=drop_mask,
                                                                        nh_projection=False)

                x_levels[global_level] = [x.squeeze(dim=-2)]
                
                drop_masks_level[global_level] = drop_mask

        return x_levels, drop_masks_level


class decomp_layer_diff(nn.Module):
    def __init__(self, global_levels, grid_layers: dict) -> None: 
        super().__init__()
        # not learned

        self.grid_layers = grid_layers
        
        self.max_level = list(grid_layers.keys())[-1]
        self.global_levels = global_levels.sort().values
        self.global_level_diff = self.global_levels.diff()

    def forward(self, x, indices_layers, drop_mask=None):
        
        x_levels={}
        drop_masks_level = {}
        #from fine to coarse
        for k, global_level in enumerate(self.global_levels[:-1]):

            global_level = int(global_level)
            drop_masks_level[global_level] = drop_mask

            if k == 0:
                x = x[global_level]

            b,n,e = x.shape

            x_sections, _, _ = self.grid_layers[str(global_level)].get_sections(x, indices_layers[global_level], section_level=self.global_level_diff[k], return_indices=False)

            weights = torch.ones(x_sections.shape[:-1])

            if drop_mask is not None:
                drop_mask = sequenize(drop_mask, max_seq_level=self.global_level_diff[k])
                weights[drop_mask] = 0
                x_sections[drop_mask]=0

            weights = weights/(weights.sum(dim=-1, keepdim=True)+1e-10)
            x_sections_f = (x_sections*weights.unsqueeze(dim=-1)).sum(dim=-2, keepdim=True)
            
            x_res = (x_sections - x_sections_f)
                        
            if drop_mask is not None:
                x_res[drop_mask] = 0 
                drop_mask = drop_mask.sum(dim=-1)==4**self.global_level_diff[k]

            x_levels[global_level] = [x_res.view(b,n,-1)]
            
            x = x_sections_f.squeeze(dim=-2)

        x_levels[int(self.global_levels[-1])] = [x]
        drop_masks_level[int(self.global_levels[-1])] = drop_mask

        if len(self.global_levels) == 0:
            x_levels[0] = x

        return x_levels, drop_masks_level


class processing_layer(nn.Module):
    #on each grid layer

    def __init__(self, global_levels, grid_layers: dict, model_hparams, mode='nh_VM') -> None: 
        super().__init__()

        output_dim = len(model_hparams['variables_target']['cell'])*(1+model_hparams['var_model'])
        kernel_dim = 4 if "projection_kernel_dim" not in model_hparams.keys() else model_hparams["projection_kernel_dim"]
        n_chunks = 8 if "projection_n_chunks" not in model_hparams.keys() else model_hparams["projection_n_chunks"]
        residual = True if "projection_residual" not in model_hparams.keys() else model_hparams["projection_residual"]

        self.var_projection = model_hparams['var_model']

        self.global_levels = global_levels
        self.mode = mode

        self.processing_layers = nn.ModuleDict()
        self.gammas = nn.ParameterDict()

        for global_level in global_levels:
            global_level = str(int(global_level))
            

            if mode == 'spatial_seq':
                self.processing_layers[global_level] = multi_grid_spatial_attention_ds(global_level, global_level, grid_layers, model_hparams, nh_attention=False, continous_pos_embedding=True)

            elif mode == 'spatial_nh':
                self.processing_layers[global_level] = multi_grid_spatial_attention_ds(global_level, global_level, grid_layers, model_hparams, nh_attention=True, continous_pos_embedding=True)

            elif mode == 'nh_ca_VM':
                self.processing_layers[global_level] = projection_layer_VM_multi_ca(model_hparams, grid_layers[global_level].min_dist, grid_layers[global_level].min_dist/2, kernel_dim=kernel_dim, n_chunks=n_chunks, residual=False)
                self.gammas[global_level] = nn.Parameter(torch.ones(model_hparams["model_dim"])*1e-6, requires_grad=True)

            elif mode == 'n_ll_ca':
                self.processing_layers[global_level] = projection_layer_n_lon_lat_multi_ca(model_hparams, grid_layers[global_level].min_dist, grid_layers[global_level].min_dist/2, kernel_dim=kernel_dim, n_chunks=n_chunks, residual=residual)
                self.gammas[global_level] = nn.Parameter(torch.ones(model_hparams["model_dim"])*1e-6, requires_grad=True)

            elif mode == 'linear':
                self.processing_layers[global_level] = nn.Linear(model_hparams['model_dim'], model_hparams['model_dim'], bias=False)
                self.gammas[global_level] = nn.Parameter(torch.ones(model_hparams["model_dim"])*1e-6, requires_grad=True)

        self.grid_layers = grid_layers

            
    def forward(self, x_levels, indices_layers, batch_dict, drop_masks_level=None):
        
        for k, global_level in enumerate(self.global_levels):
            global_level = int(global_level)
            if self.mode == 'linear' or self.mode == 'ca':
                x = self.processing_layers[str(global_level)](x_levels[global_level])

            elif 'spatial' in self.mode:
                x, drop_mask_level = self.processing_layers[str(global_level)](get_subdict(x_levels, [global_level]), drop_masks_level, indices_layers, batch_dict)

            else:
                x_in = x_levels[global_level][0]
                x, drop_mask_level = self.processing_layers[str(global_level)](x_in, 
                                                                               indices_layer=indices_layers[global_level], 
                                                                               grid_layer=self.grid_layers[str(global_level)],
                                                                               grid_layer_out=self.grid_layers[str(global_level)], 
                                                                               indices_layer_out = indices_layers[global_level],
                                                                               sample_dict=batch_dict, 
                                                                               nh_projection=True)
                if str(global_level) in self.gammas.keys():
                    x = x_in + self.gammas[str(global_level)]*x.view(x_in.shape)

                x = {global_level: [x]}
                drop_mask_level = {global_level: drop_mask_level}

            x_levels.update(x)
            drop_masks_level.update(drop_mask_level)

        return x_levels, drop_masks_level


class channel_attention(nn.Module):
    def __init__(self, grid_layer, model_hparams, model_dim=None, n_chunks=None) -> None: 
        super().__init__()

        self.grid_layer = grid_layer

        model_dim = model_hparams['model_dim']

        # channel attention is good here!
        self.n_chunks = 4

        input_model_dim = model_hparams['model_dim'] // self.n_chunks

        self.norm = nn.LayerNorm(input_model_dim, elementwise_affine=True)
        
        self.MHA = helpers.MultiHeadAttentionBlock(
            model_dim, input_model_dim, model_hparams['n_heads'], input_dim=input_model_dim, qkv_proj=True
            )   
        
        self.mlp_layer = nn.Sequential(
            nn.LayerNorm(model_dim, elementwise_affine=True),
            nn.Linear(model_dim, model_dim, bias=False),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim, bias=False)
        )
        self.gamma = nn.Parameter(torch.ones(model_dim)*1e-6, requires_grad=True)
        self.gamma_mlp = nn.Parameter(torch.ones(model_dim)*1e-6, requires_grad=True)

    def forward(self, x_level):
        b,n,f = x_level.shape

        x = x_level.view(b*n,f)

        x = x.view(b*n,self.n_chunks,-1)

        x = self.norm(x)

        x = self.MHA(q=x, k=x, v=x)

        x = x_level + self.gamma * x.view(b,n,f)
        x = x + self.gamma_mlp * self.mlp_layer(x)

        x = x.view(b,n,f)

        return x


class nh_channel_attention(nn.Module):
    def __init__(self, grid_layer, model_hparams: dict) -> None: 
        super().__init__()

        self.grid_layer = grid_layer

        model_dim = model_hparams['model_dim']

        # channel attention is good here!
        self.n_chunks = 4

        input_model_dim = model_hparams['model_dim'] // self.n_chunks

        self.norm = nn.LayerNorm(input_model_dim, elementwise_affine=True)
        
        self.pos_embedding = nh_pos_embedding(grid_layer, nh=model_hparams['nh'], emb_dim=model_hparams['model_dim'], softmax=True)

        self.MHA = helpers.MultiHeadAttentionBlock(
            model_dim, input_model_dim, model_hparams['n_heads'], input_dim=input_model_dim, qkv_proj=True
            )
        
        self.mlp_layer = nn.Sequential(
            nn.LayerNorm(model_dim, elementwise_affine=True),
            nn.Linear(model_dim, model_dim, bias=False),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim, bias=False)
        )

        self.gamma = nn.Parameter(torch.ones(model_dim)*1e-6, requires_grad=True)
        self.gamma_mlp = nn.Parameter(torch.ones(model_dim)*1e-6, requires_grad=True)

    def forward(self, x_level, indices_layer, batch_dict):
        
        x_nh = self.pos_embedding(x_level, indices_layer, batch_dict, add_to_x=True)
        
        x_nh = x_nh.sum(dim=-2)

        b,n,f = x_nh.shape
    
        x_nh = x_nh.view(b*n,self.n_chunks,-1)
        
        x_nh = self.norm(x_nh)

        x_nh = self.MHA(q=x_nh, k=x_nh, v=x_nh)

        x = x_level + self.gamma*x_nh.view(b,n,f)
        x = x + self.gamma_mlp*self.mlp_layer(x)

        x = x.view(b,n,f)

        return x


class nh_vm_channel_attention(nn.Module):
    def __init__(self, grid_layer, model_hparams: dict) -> None: 
        super().__init__()

        self.grid_layer = grid_layer

        model_dim = model_hparams['model_dim']

        # channel attention is good here!
        self.n_vm = 4
        self.n_d  = 2

        self.kappa_scan = nn.Parameter(torch.tensor(model_hparams["kappa_vm"], dtype=float), requires_grad=False) 
        self.simga_scan = nn.Parameter(grid_layer.min_dist, requires_grad=True)
                       
        self.min_val = self.grid_layer.min_dist/10
        self.max_val = self.grid_layer.max_dist
                    
        self.dists_0 = nn.Parameter(torch.linspace(0, self.max_val, self.n_d).unsqueeze(dim=-1).repeat_interleave(self.n_vm, dim=-1), requires_grad=False)
        self.phi_0 = nn.Parameter(torch.linspace(-torch.pi, torch.pi, self.n_vm + 1)[:-1].unsqueeze(dim=0).repeat_interleave(self.n_d, dim=0), requires_grad=False)

        self.n_chunks = 4

        input_model_dim = self.n_vm * self.n_d * model_hparams["model_dim"] // self.n_chunks
        self.norm = nn.LayerNorm(input_model_dim, elementwise_affine=True)

        self.MHA = helpers.MultiHeadAttentionBlock(
            model_dim, model_dim//self.n_chunks, model_hparams['n_heads'], input_dim=input_model_dim, qkv_proj=True
            )   
        
        self.mlp_layer = nn.Sequential(
            nn.LayerNorm(model_dim, elementwise_affine=True),
            nn.Linear(model_dim, model_dim, bias=False),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim, bias=False)
        )

        self.gamma_mlp = nn.Parameter(torch.ones(model_dim)*1e-6, requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(model_dim)*1e-6, requires_grad=True)

    def forward(self, x_level, indices_layer, batch_dict):
        
        x_vm = self.grid_layer.get_projection_nh_vm(x_level, indices_layer, batch_dict, self.phi_0, self.dists_0, self.simga_scan, self.kappa_scan)

        b,n,nd,nvm,f = x_vm.shape
        x_vm = x_vm.view(b*n,nd,nvm,f)

        x_vm = x_vm.view(b*n,nd,nvm,-1,self.n_chunks).view(b*n,-1,self.n_chunks).transpose(-1,-2)
        
        x_vm = self.norm(x_vm)
        x_vm = self.MHA(q=x_vm, k=x_vm, v=x_vm)

        x = x_level + self.gamma*x_vm.view(b,n,f)
        x = x + self.gamma_mlp*self.mlp_layer(x)

        x = x.view(b,n,f)

        return x



# make similar hier nh grid embedding (hier vielleicht noch nicht wichtig)
class nh_pos_embedding(nn.Module):
    def __init__(self, grid_layer: grid_layer, nh, emb_dim, softmax=False) -> None: 
        super().__init__()
        #uses hierarical grid embeddings
        #vm to interpolate -> for nh attention

        self.n_nh  = [4,6,20]

        # number of bins per neighbours
        n_bins_nh = {1:4, 2:6, 3:12} 

        self.grid_layer = grid_layer
        self.nh = nh
        
        self.softmax = softmax

        self.angular_embedders= nn.ParameterList()
        for n in range(nh):
            self.angular_embedders.append(helpers.PositionEmbedder_phys(-torch.pi, torch.pi, n_bins_nh[n+1], n_heads=emb_dim, special_token=True, constant_init=False))


    def forward(self, x, indices_layer, batch_dict, add_to_x=True, drop_mask=None):
      
        x_nh, _, rel_coords = self.grid_layer.get_nh(x, indices_layer, batch_dict, coord_system='polar', relative_coordinates=True)
        rel_coords = torch.stack(rel_coords, dim=0)
        b,n,n_nh,f = x_nh.shape
        
        rel_coords = rel_coords.split(tuple(self.n_nh[:self.nh]), dim=-1)
        
        embeddings = []
        for i in range(self.nh):

            embeddings.append(self.angular_embedders[i](rel_coords[i][1], special_token_mask=rel_coords[i][0]<1e-6))

        embeddings = torch.concat(embeddings, dim=-2)

        if self.softmax:
            embeddings = F.softmax(embeddings, dim=-2)

        if add_to_x:
            x_nh = x_nh * embeddings  
            return x_nh.view(b,n,n_nh,f)
        else:
            return embeddings.view(b,n,n_nh,f)


class seq_grid_embedding2(nn.Module):
    def __init__(self, grid_layer: grid_layer, max_seq_level, n_bins, emb_dim, constant_init=False) -> None: 
        super().__init__()
        #uses hierarical grid embeddings
        #vm to interpolate -> for nh attention

        self.grid_layer = grid_layer
        self.max_seq_level = max_seq_level

        self.angular_embedders= nn.ParameterList()
        for _ in range(max_seq_level):
            self.angular_embedders.append(helpers.PositionEmbedder_phys(-torch.pi, torch.pi, n_bins, n_heads=emb_dim, special_token=True, constant_init=constant_init))

    def forward(self, x, indices_layer):
        b,n,f = x.shape
        
        seq_level = min([get_max_seq_level(x), self.max_seq_level])

        all_embeddings = []
        for i in range(seq_level):
            x, mask, rel_coords, indices_layer = self.grid_layer.get_sections(x, indices_layer, section_level=1, relative_coordinates=True, return_indices=True, coord_system="polar")
            
            embeddings = self.angular_embedders[i](rel_coords[1], special_token_mask=rel_coords[0]<1e-6)

            if i > 0:
                scale = scale.view(x.shape) + embeddings
            else:
                scale = embeddings

            if seq_level>1:
                indices_layer = indices_layer[:,:,[0]]

            all_embeddings.append(scale.view(b,n,f))
        return scale.view(b,n,f)

# make similar hier nh grid embedding (hier vielleicht noch nicht wichtig)
class seq_grid_embedding(nn.Module):
    def __init__(self, grid_layer: grid_layer, n_bins, seq_level, emb_dim, softmax=True,  constant_init=False) -> None: 
        super().__init__()
        #uses hierarical grid embeddings
        #vm to interpolate -> for nh attention


        self.grid_layer = grid_layer
        self.seq_level = seq_level

        self.softmax = softmax

        self.angular_embedders= nn.ParameterList()
        for _ in range(seq_level):
            self.angular_embedders.append(helpers.PositionEmbedder_phys(-torch.pi, torch.pi, n_bins, n_heads=emb_dim, special_token=True, constant_init=constant_init))
            #initi to 0.25

    def forward(self, x, indices_layer,  add_to_x=True, drop_mask=None):
        b,n,f = x.shape
        
        seq_level = min([get_max_seq_level(x), self.seq_level])

        for i in range(seq_level):
            x, mask, rel_coords, indices_layer = self.grid_layer.get_sections(x, indices_layer, section_level=1, relative_coordinates=True, return_indices=True, coord_system="polar")
            
            embeddings = self.angular_embedders[i](rel_coords[1], special_token_mask=rel_coords[0]<1e-6)

            if self.softmax:
                if drop_mask is not None and i==seq_level-1:
                    embeddings[drop_mask.view(x.shape[:-1],1)] = -100
                embeddings = F.softmax(embeddings, dim=-2-i)

            if i > 0:
                scale = scale.view(x.shape) + embeddings
            else:
                scale = embeddings

            if seq_level>1:
                # keep midpoints of previous
                indices_layer = indices_layer[:,:,[0]]

        if add_to_x:
            x = x * scale  
            return x.view(b,n,f)
        else:
            return scale.view(b,n,f)



class lin_projection(nn.Module):
    def __init__(self, grid_layers: dict, model_hparams: dict) -> None: 
        super().__init__()

        model_dim = model_hparams['model_dim']

        self.grid_layers = grid_layers

        self.layers = nn.ModuleDict()
        self.global_levels = []
        for global_level in grid_layers.keys(): 
            self.layers[str(global_level)] = nn.Linear(model_dim, model_dim, bias=False)
            self.global_levels.append(int(global_level))
            
    def forward(self, x_levels):
        
        for global_level in self.global_levels:
            x_levels[global_level] = self.layers[str(global_level)](x_levels[global_level])

        return x_levels


class output_layer(nn.Module):
    def __init__(self, mapping, coordinates, global_levels, grid_layers: dict, model_hparams, mode='learned') -> None: 
        super().__init__()

        self.register_buffer("mapping", mapping, persistent=False)
        self.register_buffer("coordinates", coordinates, persistent=False)

        output_dim = len(model_hparams['variables_target']['cell'])*(1+model_hparams['var_model'])
        self.var_projection = model_hparams['var_model']
        self.mode = mode
        self.global_levels = global_levels

        self.projection_layers = nn.ModuleDict()
        self.lin_projection_layers = nn.ModuleDict()

        for global_level in global_levels:
            if global_level >0:
                global_level = str(int(global_level))
                self.projection_layers[global_level] = get_projection_layer(mode, model_hparams, grid_layers[global_level])
            
            self.lin_projection_layers[str(int(global_level))] = nn.Linear(model_hparams['model_dim'], output_dim, bias=False)


        self.grid_layers = grid_layers

            
    def forward(self, x_levels, indices_grid_layer0, indices_layers, batch_dict):
        
        x_output_mean = []
        x_output_var = []

        coords_output = self.coordinates[:,self.mapping[indices_grid_layer0]]
        n_c, b, n, n_nh = coords_output.shape

        for k, global_level in enumerate(self.global_levels):
            global_level = int(global_level)
            if global_level>0:
                if self.mode == 'simple':
                    x = x_levels[k]
                else:
                    x = self.projection_layers[str(global_level)](x_levels[global_level], indices_layer=indices_layers[global_level], grid_layer=self.grid_layers[str(global_level)],coordinates_out=coords_output, sample_dict=batch_dict, nh_projection=True)[0]
            else:
                x = x_levels[k]

            x = x.view(b, n, -1)
            x = self.lin_projection_layers[str(global_level)](x)
           
            if self.var_projection:
                x, x_var = x.split(x.shape[-1] // 2, dim=-1)
                x_output_mean.append(x)
                x_output_var.append(nn.functional.softplus(x_var))
            else:
                x_output_mean.append(x)

        return x_output_mean, x_output_var


def get_coordinates_grid_layers(grid_layers, indices, grid_levels=None):
    if grid_levels is None:
        grid_levels = list(grid_layers.keys())
    
    coordinates = {}
    for grid_level in grid_levels:
        coordinates[grid_level] = grid_layers[grid_level].get_coordinates_from_grid_indices(indices[grid_level])
    
    return coordinates


def get_projection_layer(projection_mode, model_hparams, grid_layer):

    kernel_dim = 4 if "projection_kernel_dim" not in model_hparams.keys() else model_hparams["projection_kernel_dim"]
    n_chunks = 8 if "projection_n_chunks" not in model_hparams.keys() else model_hparams["projection_n_chunks"]
    residual = True if "projection_residual" not in model_hparams.keys() else model_hparams["projection_residual"]
    
    if projection_mode == 'vm':
        projection_layer = projection_layer_vm(model_hparams, grid_layer.min_dist/2)

    elif projection_mode == 'n':
        projection_layer = projection_layer_n(model_hparams, grid_layer.min_dist, grid_layer.min_dist/2)

    elif projection_mode == 'n_ca':
        projection_layer = (projection_layer_n_multi_ca(model_hparams, grid_layer.min_dist, grid_layer.min_dist/2, kernel_dim=kernel_dim, n_chunks=n_chunks, residual=residual))
    
    elif projection_mode == 'vm_ca':
        projection_layer = (projection_layer_VM_multi_ca(model_hparams, grid_layer.min_dist, grid_layer.min_dist/2, kernel_dim=kernel_dim, n_chunks=n_chunks, residual=residual))

    elif projection_mode == 'vm_ca_red':
        projection_layer = (projection_layer_VM_multi_ca_red(model_hparams, grid_layer.min_dist, grid_layer.min_dist/2, kernel_dim=kernel_dim, n_chunks=n_chunks, residual=residual))

    elif projection_mode == 'cosine_ca':
        projection_layer = (projection_layer_cosine_multi_ca(model_hparams, grid_layer.max_dist, grid_layer.min_dist, kernel_dim=kernel_dim, n_chunks=n_chunks, residual=residual))

    elif projection_mode == 'learned_cont':
        projection_layer = (projection_layer_learned_cont(model_hparams))

    elif projection_mode == 'n_ll_ca':
        projection_layer = (projection_layer_n_lon_lat_multi_ca(model_hparams, grid_layer.min_dist, grid_layer.min_dist/2, kernel_dim=kernel_dim, n_chunks=n_chunks, residual=residual))

    elif projection_mode == 'mha':
        projection_layer = projection_layer_mha(model_hparams, mean_res=False)
    
    model_dim_out = model_hparams['model_dim']
    return projection_layer

class projection_layer(nn.Module):
    def __init__(self, model_hparams, polar=None, requires_arel_positions=True, residual=True, channel_attention=True, output_model_dim=None) -> None: 
        super().__init__()

        self.periodic_fov = model_hparams['periodic_fov']
        
        if polar is None:
            if 'cartesian' in model_hparams['pos_emb_calc']:
                self.polar = False
                self.coord_system = 'cartesian'
            else:
                self.polar = True
                self.coord_system = 'polar'
        else:
            self.polar=polar
            self.coord_system = 'polar' if polar else 'cartesian'
        
        self.requires_arel_positions = requires_arel_positions

        if output_model_dim is None:
            output_model_dim = model_hparams['model_dim']

        '''
        if channel_attention:
        
            self.MHA = helpers.MultiHeadAttentionBlock(
                mha_dim, model_dim, model_hparams['n_heads'], input_dim=mha_dim, qkv_proj=True
                )   
            self.layer_norm = nn.LayerNorm(mha_dim, elementwise_affine=True)

            self.mlp_layer = nn.Sequential(
                nn.Linear(self.n_chunks*model_dim, model_dim, bias=False),
                nn.SiLU(),
                nn.Linear(model_dim, model_dim, bias=False)
            )

            if residual:
                self.mlp_norm = nn.LayerNorm(self.n_chunks*model_dim, elementwise_affine=True)
                self.gamma = nn.Parameter(torch.ones(model_dim)*1e-6, requires_grad=True)
        '''

    def forward(self, 
                x, 
                grid_layer: grid_layer=None, 
                grid_layer_out: grid_layer=None, 
                indices_layer=None, 
                indices_layer_out=None,
                nh_projection=True,
                sample_dict=None,
                coordinates=None, 
                coordinates_out=None,
                mask=None):
        
        if coordinates is None and grid_layer is not None and not nh_projection:
            coordinates = grid_layer.get_coordinates_from_grid_indices(indices_layer)

        elif coordinates is None and grid_layer is not None and nh_projection:
            x, mask, coordinates = grid_layer.get_nh(x, indices_layer, sample_dict, relative_coordinates=False, coord_system=self.coord_system, mask=mask)
        
        if coordinates_out is None and grid_layer_out is not None:
            coordinates_out = grid_layer_out.get_coordinates_from_grid_indices(indices_layer_out)
        
        n_c, b, seq_dim_in = coordinates.shape[:3]
        seq_dim_out = coordinates_out.shape[2]
        
        if seq_dim_in > seq_dim_out:
            coordinates = coordinates.view(n_c, b, seq_dim_out, -1)
            x = x.view(b, seq_dim_out, coordinates.shape[-1], -1)
            if mask is not None:
                mask = mask.view(b, seq_dim_out, -1)
            coordinates_out = coordinates_out.view(n_c, b, seq_dim_out, -1)
        else:
            coordinates = coordinates.view(n_c, b, seq_dim_in,-1)
            coordinates_out = coordinates_out.view(n_c, b, seq_dim_in, -1)
            x = x.view(b, seq_dim_in, coordinates.shape[-1], -1)
            if mask is not None:
                mask = mask.view(b, seq_dim_in, -1)

        if self.requires_arel_positions:
            coords_ref = coordinates[:,:,:,[0]]
            coordinates_rel = get_distance_angle(coords_ref[0], coords_ref[1], coordinates[0], coordinates[1], base=self.coord_system, periodic_fov=self.periodic_fov)
            coordinates_rel_out = get_distance_angle(coords_ref[0], coords_ref[1], coordinates_out[0], coordinates_out[1], base=self.coord_system, periodic_fov=self.periodic_fov)

            return self.project(x, coordinates_rel, coordinates_rel_out, mask=mask)
        
        else:
            coordinates = coordinates.unsqueeze(dim=-2)
            coordinates_out = coordinates_out.unsqueeze(dim=-1)
            coordinates_rel = get_distance_angle(coordinates[0], coordinates[1], coordinates_out[0], coordinates_out[1], base=self.coord_system, periodic_fov=self.periodic_fov)

            return self.project(x, coordinates_rel, mask=mask)      


    def project(self, x, coordinates_rel, coordinates_rel_out=None, mask=None):
        return x, mask


class projection_layer_learned_cont(projection_layer):
    def __init__(self, model_hparams) -> None: 
        super().__init__(model_hparams)
    
        model_dim = model_hparams['model_dim']
        pos_emb_calc = model_hparams['pos_emb_calc']
        emb_table_bins = model_hparams['emb_table_bins']      
      
        self.positon_embedder = position_embedder(0,0, emb_table_bins, model_dim, pos_emb_calc=pos_emb_calc)

            
    def project(self, x, coordinates_rel, coordinates_rel_out, mask=None):
         
        pos_embeddings_in = self.positon_embedder(coordinates_rel[0], coordinates_rel[1])
        pos_embeddings_out = self.positon_embedder(coordinates_rel_out[0], coordinates_rel_out[1])

        b, n_in, n_seq, f = x.shape 
        n_out = pos_embeddings_out.shape[-2]
        
        if mask is not None:
            pos_embeddings_in[mask] = -1e30 if pos_embeddings_in.dtype == torch.float32 else -1e4

        pos_embeddings_out = pos_embeddings_out.view(b, n_in, -1, f)
        
        weights = pos_embeddings_in.unsqueeze(dim=-3) * pos_embeddings_out.unsqueeze(dim=-2)

        weights = F.softmax(weights, dim=-2)

        x = (weights * x.unsqueeze(dim=-3)).sum(dim=-2)

        if mask is not None:
            mask_update = mask.sum(dim=-1)==mask.shape[2]

            if x.shape[1]*x.shape[2] > mask_update.shape[-1]:
                mask_update = mask_update.unsqueeze(dim=-1).repeat_interleave(x.shape[-2], dim=-1)
        else:
            mask_update = mask

        return x, mask_update
 

class projection_layer_vm(projection_layer):
    # define as projection_layer
    # to be done
    def __init__(self, model_hparams: dict, dist_min) -> None: 
        super().__init__(model_hparams, polar=True, requires_arel_positions=False)

        model_dim = model_hparams['model_dim']

        self.register_buffer('phi_0', torch.linspace(-torch.pi, torch.pi, model_dim+1)[:-1])

        self.kappa_vm = nn.Parameter(torch.ones(1) * model_hparams["kappa_vm"], requires_grad=True)
        self.sigma = nn.Parameter(torch.ones(1)*dist_min, requires_grad=True)


    def project(self, x: torch.tensor, coordinates_rel, mask=None):
        
        b, n, nh, f = x.shape

        dists = coordinates_rel[0]
        thetas = coordinates_rel[1]

        weights_dist = normal_dist(dists, self.sigma, sigma_cross=True)
        weights_vm = von_mises(thetas, self.kappa_vm, self.phi_0).squeeze(dim=-1)/torch.exp(self.kappa_vm)

        weights_vm = weights_vm.masked_fill(dists.unsqueeze(dim=-1)==0, 1)
 
        weights = weights_dist.unsqueeze(dim=-1)*weights_vm

        if mask is not None:
            weights = weights.masked_fill(mask.unsqueeze(dim=2).unsqueeze(dim=-1), -1e30 if x.dtype == torch.float32 else -1e4)

        weights = F.softmax(weights, dim=-2)

        x = (x.unsqueeze(dim=2)*weights).sum(dim=-2)
   
        if mask is not None:
            mask_update = mask.sum(dim=-1)==mask.shape[-1]

            if x.shape[1]*x.shape[2] > mask_update.shape[-1]:
                mask_update = mask_update.unsqueeze(dim=-1).repeat_interleave(x.shape[-2], dim=-1)
        else:
            mask_update = mask

        return x, mask_update


class projection_layer_mha(projection_layer):
    # define as projection_layer
    def __init__(self, model_hparams: dict, mean_res=True) -> None: 
        super().__init__(model_hparams, requires_arel_positions=True)

        model_dim = model_hparams['model_dim']
        pos_emb_calc = model_hparams['pos_emb_calc']
        emb_table_bins = model_hparams['emb_table_bins']

        self.position_embedder = position_embedder(0,0, emb_table_bins, model_dim, pos_emb_calc=pos_emb_calc)
        self.layer_norm = nn.LayerNorm(model_dim, elementwise_affine=True)
        self.layer_normq = nn.LayerNorm(model_dim, elementwise_affine=True)

        self.mha = helpers.MultiHeadAttentionBlock(model_dim, model_dim, model_hparams['n_heads'],qkv_proj=False)

        self.kv_proj = nn.Linear(model_dim, model_dim*2, bias=False)
        self.q_proj = nn.Linear(model_dim, model_dim, bias=False)

        self.embedding_layer = nn.Linear(model_dim, model_dim*2, bias=False)

        self.mlp_layer = nn.Sequential(
            nn.LayerNorm(model_dim, elementwise_affine=True),
            nn.Linear(model_dim, model_dim, bias=True),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim, bias=True)
        )

        self.gamma = nn.Parameter(torch.ones(model_dim)*1e-6, requires_grad=True)
        self.gamma_mlp = nn.Parameter(torch.ones(model_dim)*1e-6, requires_grad=True)

        self.mean_ref = mean_res


    def project(self, x, coordinates_rel, coordinates_rel_out, mask=None):
        
        pos_embeddings_rel = self.position_embedder(coordinates_rel[0],coordinates_rel[1])
        pos_embeddings_rel_out = self.position_embedder(coordinates_rel_out[0],coordinates_rel_out[1])

        b,n,nh,f = x.shape
        nt = pos_embeddings_rel_out.shape[-2]

        x_res = x.mean(dim=-2)

        if self.mean_ref:
            q = x.mean(dim=-2)
        else:
            q = x[:,:,0,:]

        q_shift, q_scale = self.embedding_layer(pos_embeddings_rel_out).chunk(2, dim=-1)
        q = self.layer_normq(q)
        q = q.unsqueeze(dim=-2) * (q_scale + 1) + q_shift

        shift, scale = self.embedding_layer(pos_embeddings_rel).chunk(2, dim=-1)

        x = self.layer_norm(x) * (scale + 1) + shift
        
        k, v = self.kv_proj(x).chunk(2, dim=-1)
        q = self.q_proj(q)

        q = q.view(b*n,nt,f)
        k = k.view(b*n,nh,f)
        v = v.view(b*n,nh,f)

        out = self.mha(q=q,k=k,v=v)

        out = out.view(b,n,nt,f)
        x = x_res.unsqueeze(dim=-2) + self.gamma*out

        x = x + self.gamma_mlp*self.mlp_layer(x)

        if mask is not None:
            mask_update = mask.sum(dim=-1)==mask.shape[-1]

            if x.shape[1]*x.shape[2] > mask_update.shape[-1]:
                mask_update = mask_update.unsqueeze(dim=-1).repeat_interleave(x.shape[-2], dim=-1)
        else:
            mask_update = mask

        return x, mask_update

'''
class projection_layer_spatial_kernel_polar(projection_layer):
    # define as projection_layer
    def __init__(self, model_hparams: dict, n_phis, sigmas_start, dists_start=None, cross_calculation=False) -> None: 
        super().__init__(model_hparams, polar=True, requires_arel_positions=False)

        #sigma = torch.linspace(sigma_min, sigma_max, model_hparams['model_dim'])
        self.sigma = nn.Parameter(sigmas_start, requires_grad=True)

        if dists_start is not None:
            self.dists_start = nn.Parameter(dists_start, requires_grad=True)


    def project(self, x, coordinates_rel, mask=None):
       
        dists = coordinates_rel[0]
 
        weights = normal_dist(dists, self.sigma)
        
        if mask is not None:
            weights = weights.masked_fill(mask.unsqueeze(dim=2).unsqueeze(dim=-1), -1e30 if x.dtype == torch.float32 else -1e4)

        weights = F.softmax(weights, dim=-2)

        x = (weights * x.unsqueeze(dim=2)).sum(dim=[-2])

        if mask is not None:
            mask_update = mask.sum(dim=-1)==mask.shape[-1]

            if x.shape[1]*x.shape[2] > mask_update.shape[-1]:
                mask_update = mask_update.unsqueeze(dim=-1).repeat_interleave(x.shape[-2], dim=-1)
        else:
            mask_update = mask

        return x, mask_update
'''

class projection_layer_n(projection_layer):
    # define as projection_layer
    def __init__(self, model_hparams: dict, sigma_max, sigma_min) -> None: 
        super().__init__(model_hparams, polar=True, requires_arel_positions=False)

        sigma = torch.linspace(sigma_min, sigma_max, model_hparams['model_dim'])
        self.sigma = nn.Parameter(sigma, requires_grad=True)


    def project(self, x, coordinates_rel, mask=None):
       
        dists = coordinates_rel[0]
 
        weights = normal_dist(dists, self.sigma)
        
        if mask is not None:
            weights = weights.masked_fill(mask.unsqueeze(dim=2).unsqueeze(dim=-1), -1e30 if x.dtype == torch.float32 else -1e4)

        weights = F.softmax(weights, dim=-2)

        x = (weights * x.unsqueeze(dim=2)).sum(dim=[-2])

        if mask is not None:
            mask_update = mask.sum(dim=-1)==mask.shape[-1]

            if x.shape[1]*x.shape[2] > mask_update.shape[-1]:
                mask_update = mask_update.unsqueeze(dim=-1).repeat_interleave(x.shape[-2], dim=-1)
        else:
            mask_update = mask

        return x, mask_update


class projection_layer_cosine_multi_ca(projection_layer):
    # define as projection_layer
    def __init__(self, model_hparams: dict, sigma_max, sigma_min, kernel_dim=4, n_chunks=4, residual=True) -> None: 
        super().__init__(model_hparams, polar=True, requires_arel_positions=False)

        model_dim = model_hparams['model_dim']

        self.register_buffer('phi_0', torch.linspace(-torch.pi, torch.pi, kernel_dim+1)[:-1])

        #dist = torch.linspace(0, sigma_max, model_dim)
        #sigma = dist.clamp(min=sigma_min)/2

        #self.dist_0 = nn.Parameter(dist, requires_grad=True)
        self.min_wl = sigma_min*2
       
        #self.sigma = nn.Parameter(torch.randn(model_dim).abs() * sigma_min, requires_grad=True)
        self.wavelengths = nn.Parameter(torch.linspace(self.min_wl, sigma_max*10, model_dim), requires_grad=True)

        mha_dim = kernel_dim

        self.n_chunks = n_chunks
        total_dim = kernel_dim * model_dim

        mha_dim = total_dim//n_chunks
        self.mha_dim = mha_dim

        self.MHA = helpers.MultiHeadAttentionBlock(
            mha_dim, model_dim, model_hparams['n_heads'], input_dim=mha_dim, qkv_proj=True
            )   
        
        self.layer_norm = nn.LayerNorm(mha_dim, elementwise_affine=True)

        self.mlp_layer = nn.Sequential(
            
            nn.Linear(self.n_chunks*model_dim, model_dim, bias=False),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim, bias=False)
        )

        if residual:
            self.mlp_norm = nn.LayerNorm(self.n_chunks*model_dim, elementwise_affine=True)
            self.gamma = nn.Parameter(torch.ones(model_dim)*1e-6, requires_grad=True)
        
        self.residual = residual

    def project(self, x: torch.tensor, coordinates_rel, mask=None):
        
        b, n, nh, f = x.shape

        dists = coordinates_rel[0]
        thetas = coordinates_rel[1]

        wavelengths = self.wavelengths.clamp(min=self.min_wl)
        weights = cosine(thetas, wavelengths, dists, theta_offsets=self.phi_0)

        if mask is not None:
            #min_val = 1e10 if x.dtype == torch.float32 else 1e4
            weights = weights.masked_fill(mask.unsqueeze(dim=2).unsqueeze(dim=-1).unsqueeze(dim=-1), 0)
            scaling = (mask==False).sum(dim=-1)
       # scaling =     
        #weights = F.softmax(weights, dim=-3)

            x_p = (x.unsqueeze(dim=2).unsqueeze(dim=-2) * weights).sum(dim=-3)/(scaling.view(b,n,1,1,1)+1e-10)

            #x_p[scaling==0] = 0
        else:
            x_p = (x.unsqueeze(dim=2).unsqueeze(dim=-2) * weights).mean(dim=-3)

        if self.residual:
            x_res = x_p.mean(dim=-2)

        b,n,nt,f,n_k = x_p.shape

        x_p = x_p.view(b,n,nt,-1)
        x_p = x_p.view(b*n,nt,-1)
        x_p = x_p.view(b*n*nt,-1,self.n_chunks).transpose(-1,-2)

        q = k = v = self.layer_norm(x_p)
        
        if not self.residual:
            v = x_p

        x_mha = self.MHA(q=q, k=k, v=v)
        x_mha = x_mha.view(b*n,nt,-1)
        x_mha = x_mha.view(b,n,nt,-1)

        if self.residual:
            x_mha = self.mlp_norm(x_mha)
            x = x_res + self.gamma* self.mlp_layer(x_mha)
        else:
            x = self.mlp_layer(x_mha)
   
        if mask is not None:
            mask_update = mask.sum(dim=-1)==mask.shape[-1]

            if x.shape[1]*x.shape[2] > mask_update.shape[-1]:
                mask_update = mask_update.unsqueeze(dim=-1).repeat_interleave(x_mha.shape[-2], dim=-1)
        else:
            mask_update = mask

        return x, mask_update

class projection_layer_VM_multi_ca_red(projection_layer):
    # define as projection_layer
    def __init__(self, model_hparams: dict, sigma_max, sigma_min, kernel_dim=4, n_chunks=4, residual=True) -> None: 
        super().__init__(model_hparams, polar=True, requires_arel_positions=False)

        model_dim = model_hparams['model_dim']

        self.register_buffer('phi_0', torch.linspace(-torch.pi, torch.pi, kernel_dim+1)[:-1])

        #dist = torch.linspace(0, sigma_max, model_dim)
        #sigma = dist.clamp(min=sigma_min)/2

        #self.dist_0 = nn.Parameter(dist, requires_grad=True)

        self.kappa_vm = nn.Parameter(torch.ones(1) * model_hparams["kappa_vm"], requires_grad=True)
        self.sigma = nn.Parameter(torch.randn(model_dim).abs() * sigma_min, requires_grad=True)

        mha_dim = kernel_dim

        self.n_chunks = n_chunks
        total_dim = kernel_dim * model_dim

        mha_dim = total_dim//n_chunks
        self.mha_dim = mha_dim

        self.MHA = helpers.MultiHeadAttentionBlock(
            mha_dim, model_dim, model_hparams['n_heads'], input_dim=mha_dim, qkv_proj=True
            )   
        
        self.layer_norm = nn.LayerNorm(mha_dim, elementwise_affine=True)

        self.mlp_layer = nn.Sequential(
            
            nn.Linear(self.n_chunks*model_dim, model_dim, bias=False),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim, bias=False)
        )

        if residual:
            self.mlp_norm = nn.LayerNorm(self.n_chunks*model_dim, elementwise_affine=True)
            self.gamma = nn.Parameter(torch.ones(model_dim)*1e-6, requires_grad=True)
        
        self.residual = residual

    def project(self, x: torch.tensor, coordinates_rel, mask=None):
        
        b, n, nh, f = x.shape

        min_val = 1e10 if x.dtype == torch.float32 else 1e4
        sigma = self.sigma.clamp(min=min_val)

        dists = coordinates_rel[0]
        thetas = coordinates_rel[1]

        weights_dist = normal_dist(dists.unsqueeze(dim=-1), sigma, sigma_cross=False)
        weights_vm = von_mises(thetas, self.kappa_vm, self.phi_0).squeeze(dim=-1)/torch.exp(self.kappa_vm)

        weights_vm = weights_vm.masked_fill(dists.unsqueeze(dim=-1)==0, 1)

        weights = weights_dist.unsqueeze(dim=-1)*weights_vm.unsqueeze(dim=-2)

        if mask is not None:
            weights = weights.masked_fill(mask.unsqueeze(dim=2).unsqueeze(dim=-1).unsqueeze(dim=-1), -min_val)
            
        weights = F.softmax(weights, dim=-3)

        x_p = (x.unsqueeze(dim=2).unsqueeze(dim=-1) * weights).sum(dim=-3)

        if self.residual:
            x_res = x_p.mean(dim=-1)

        b,n,nt,f,n_k = x_p.shape

        x_p = x_p.view(b,n,nt,-1)
        x_p = x_p.view(b*n,nt,-1)
        x_p = x_p.view(b*n*nt,-1,self.n_chunks).transpose(-1,-2)

        q = k = v = self.layer_norm(x_p)
        
        if not self.residual:
            v = x_p

        x_mha = self.MHA(q=q, k=k, v=v)
        x_mha = x_mha.view(b*n,nt,-1)
        x_mha = x_mha.view(b,n,nt,-1)

        if self.residual:
            x_mha = self.mlp_norm(x_mha)
            x = x_res + self.gamma* self.mlp_layer(x_mha)
        else:
            x = self.mlp_layer(x_mha)
   
        if mask is not None:
            mask_update = mask.sum(dim=-1)==mask.shape[-1]

            if x.shape[1]*x.shape[2] > mask_update.shape[-1]:
                mask_update = mask_update.unsqueeze(dim=-1).repeat_interleave(x_mha.shape[-2], dim=-1)
        else:
            mask_update = mask

        return x, mask_update


class projection_layer_VM_multi_ca(projection_layer):
    # define as projection_layer
    def __init__(self, model_hparams: dict, sigma_max, sigma_min, kernel_dim=4, n_chunks=4, residual=True) -> None: 
        super().__init__(model_hparams, polar=True, requires_arel_positions=False)

        model_dim = model_hparams['model_dim']

        self.register_buffer('phi_0', torch.linspace(-torch.pi, torch.pi, kernel_dim+1)[:-1])

        dist = torch.linspace(0, sigma_max, kernel_dim)
        sigma = dist.clamp(min=sigma_min)/2

        self.dist_0 = nn.Parameter(dist, requires_grad=True)

        self.kappa_vm = nn.Parameter(torch.ones(1) * model_hparams["kappa_vm"], requires_grad=True)
        self.sigma = nn.Parameter(sigma, requires_grad=True)

        mha_dim = kernel_dim**2

        self.n_chunks = n_chunks
        total_dim = kernel_dim**2 * model_dim

        mha_dim = total_dim//n_chunks
        self.mha_dim = mha_dim

        self.MHA = helpers.MultiHeadAttentionBlock(
            mha_dim, model_dim, model_hparams['n_heads'], input_dim=mha_dim, qkv_proj=True
            )   
        
        self.layer_norm = nn.LayerNorm(mha_dim, elementwise_affine=True)

        self.mlp_layer = nn.Sequential(
            
            nn.Linear(self.n_chunks*model_dim, model_dim, bias=False),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim, bias=False)
        )

        if residual:
            self.mlp_norm = nn.LayerNorm(self.n_chunks*model_dim, elementwise_affine=True)
            self.gamma = nn.Parameter(torch.ones(model_dim)*1e-6, requires_grad=True)
        
        self.residual = residual

    def project(self, x: torch.tensor, coordinates_rel, mask=None):
        
        b, n, nh, f = x.shape

        dists = coordinates_rel[0]
        thetas = coordinates_rel[1]

        min_val = 1e10 if x.dtype == torch.float32 else 1e4
        sigma = self.sigma.clamp(min=min_val)

        weights_dist = normal_dist(dists, sigma, self.dist_0, sigma_cross=False)
        weights_vm = von_mises(thetas, self.kappa_vm, self.phi_0).squeeze(dim=-1)/torch.exp(self.kappa_vm)

        weights_vm = weights_vm.masked_fill(dists.unsqueeze(dim=-1)==0, 1)
 
        weights = weights_dist.unsqueeze(dim=-2)*weights_vm.unsqueeze(dim=-1)

        b,n,nt,nh,_,_ = weights.shape

        weights = weights.view(b,n,nt,nh,-1)

        if mask is not None:
            weights = weights.masked_fill(mask.unsqueeze(dim=2).unsqueeze(dim=-1), -1e30 if x.dtype == torch.float32 else -1e4)

        weights = F.softmax(weights, dim=-2)

        x_p = (x.unsqueeze(dim=2).unsqueeze(dim=-2) * weights.unsqueeze(dim=-1)).sum(dim=-3)
        
        if self.residual:
            x_res = x_p.mean(dim=-2)

        x_p = x_p.view(b,n,nt,-1)
        x_p = x_p.view(b*n,nt,-1)
        x_p = x_p.view(b*n*nt,-1,self.n_chunks).transpose(-1,-2)

        q = k = v = self.layer_norm(x_p)
        
        if not self.residual:
            v = x_p

        x_mha = self.MHA(q=q, k=k, v=v)
        x_mha = x_mha.view(b*n,nt,-1)
        x_mha = x_mha.view(b,n,nt,-1)

        if self.residual:
            x_mha = self.mlp_norm(x_mha)
            x = x_res + self.gamma* self.mlp_layer(x_mha)
        else:
            x = self.mlp_layer(x_mha)
   
        if mask is not None:
            mask_update = mask.sum(dim=-1)==mask.shape[-1]

            if x.shape[1]*x.shape[2] > mask_update.shape[-1]:
                mask_update = mask_update.unsqueeze(dim=-1).repeat_interleave(x_mha.shape[-2], dim=-1)
        else:
            mask_update = mask

        return x, mask_update

class projection_layer_n_lon_lat_multi_ca(projection_layer):
    # define as projection_layer
    def __init__(self, model_hparams: dict, sigma_max, sigma_min, kernel_dim=4, n_chunks=4, residual=True) -> None: 
        super().__init__(model_hparams, polar=False, requires_arel_positions=False)

        model_dim = model_hparams['model_dim']

        dist_lat = torch.linspace(0, (sigma_min), kernel_dim)
        dist_lon = torch.linspace(0, (sigma_min), kernel_dim)
        sigma = dist_lat.clamp(min=sigma_min)/2

        self.dist_lat = nn.Parameter(dist_lat, requires_grad=True)
        self.dist_lon = nn.Parameter(dist_lon, requires_grad=True)
        self.sigma = nn.Parameter(sigma, requires_grad=True)

        mha_dim = kernel_dim**2

        self.n_chunks = n_chunks
        total_dim = kernel_dim**2 * model_dim

        mha_dim = total_dim//n_chunks
        self.mha_dim = mha_dim

        self.MHA = helpers.MultiHeadAttentionBlock(
            mha_dim, model_dim, model_hparams['n_heads'], input_dim=mha_dim, qkv_proj=True
            )   
        
        self.layer_norm = nn.LayerNorm(mha_dim, elementwise_affine=True)

        self.mlp_layer = nn.Sequential(
            
            nn.Linear(self.n_chunks*model_dim, model_dim, bias=False),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim, bias=False)
        )

        if residual:
            self.mlp_norm = nn.LayerNorm(self.n_chunks*model_dim, elementwise_affine=True)
            self.gamma = nn.Parameter(torch.ones(model_dim)*1e-6, requires_grad=True)
        
        self.residual = residual

    def project(self, x: torch.tensor, coordinates_rel, mask=None):
        
        b, n, nh, f = x.shape

        dists_lon = coordinates_rel[0]
        dists_lat = coordinates_rel[1]

        min_val = 1e10 if x.dtype == torch.float32 else 1e4
        sigma = self.sigma.clamp(min=min_val)

        weights_lon = normal_dist(dists_lon, sigma, self.dist_lat, sigma_cross=False)
        weights_lat = normal_dist(dists_lat, sigma, self.dist_lon, sigma_cross=False)

        weights = weights_lon.unsqueeze(dim=-1)*weights_lat.unsqueeze(dim=-2)

        b,n,nt,nh,_,_ = weights.shape

        weights = weights.view(b,n,nt,nh,-1)

        if mask is not None:
            weights = weights.masked_fill(mask.unsqueeze(dim=2).unsqueeze(dim=-1), -1e30 if x.dtype == torch.float32 else -1e4)

        weights = F.softmax(weights, dim=-2)

        x_p = (x.unsqueeze(dim=2).unsqueeze(dim=-2) * weights.unsqueeze(dim=-1)).sum(dim=-3)
        
        if self.residual:
            x_res = x_p.mean(dim=-2)

        x_p = x_p.view(b,n,nt,-1)
        x_p = x_p.view(b*n,nt,-1)
        x_p = x_p.view(b*n*nt,-1,self.n_chunks).transpose(-1,-2)

        q = k = v = self.layer_norm(x_p)
        
        if not self.residual:
            v = x_p

        x_mha = self.MHA(q=q, k=k, v=v)
        x_mha = x_mha.view(b*n,nt,-1)
        x_mha = x_mha.view(b,n,nt,-1)

        if self.residual:
            x_mha = self.mlp_norm(x_mha)
            x = x_res + self.gamma* self.mlp_layer(x_mha)
        else:
            x = self.mlp_layer(x_mha)
   
        if mask is not None:
            mask_update = mask.sum(dim=-1)==mask.shape[-1]

            if x.shape[1]*x.shape[2] > mask_update.shape[-1]:
                mask_update = mask_update.unsqueeze(dim=-1).repeat_interleave(x_mha.shape[-2], dim=-1)
        else:
            mask_update = mask

        return x, mask_update


class projection_layer_n_multi_ca(projection_layer):
    # define as projection_layer
    def __init__(self, model_hparams: dict, sigma_max, sigma_min, kernel_dim=4, n_chunks=4, residual=True) -> None: 
        super().__init__(model_hparams, polar=True, requires_arel_positions=False)

        model_dim = model_hparams['model_dim']

        dist = torch.linspace(0, sigma_min, kernel_dim)
        sigma = dist.clamp(min=sigma_min)/2

        self.dist_0 = nn.Parameter(dist, requires_grad=True)
        self.sigma = nn.Parameter(sigma, requires_grad=True)

        mha_dim = kernel_dim

        self.n_chunks = n_chunks
        total_dim = kernel_dim * model_dim

        mha_dim = total_dim//n_chunks
        self.mha_dim = mha_dim

        self.MHA = helpers.MultiHeadAttentionBlock(
            mha_dim, model_dim, model_hparams['n_heads'], input_dim=mha_dim, qkv_proj=True
            )   
        
        self.layer_norm = nn.LayerNorm(mha_dim, elementwise_affine=True)

        self.mlp_layer = nn.Sequential(
            
            nn.Linear(self.n_chunks*model_dim, model_dim, bias=False),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim, bias=False)
        )

        if residual:
            self.mlp_norm = nn.LayerNorm(self.n_chunks*model_dim, elementwise_affine=True)
            self.gamma = nn.Parameter(torch.ones(model_dim)*1e-6, requires_grad=True)
        
        self.residual = residual

    def project(self, x: torch.tensor, coordinates_rel, mask=None):
        
        b, n, nh, f = x.shape

        dists = coordinates_rel[0]

        min_val = 1e10 if x.dtype == torch.float32 else 1e4
        sigma = self.sigma.clamp(min=min_val)

        weights = normal_dist(dists, sigma, self.dist_0, sigma_cross=False)
        
        b,n,nt,nh,_ = weights.shape

        if mask is not None:
            weights = weights.masked_fill(mask.unsqueeze(dim=2).unsqueeze(dim=-1), -1e30 if x.dtype == torch.float32 else -1e4)

        weights = F.softmax(weights, dim=-2)

        x_p = (x.unsqueeze(dim=2).unsqueeze(dim=-2) * weights.unsqueeze(dim=-1)).sum(dim=-3)
        
        if self.residual:
            x_res = x_p.mean(dim=-2)

        x_p = x_p.view(b,n,nt,-1)
        x_p = x_p.view(b*n,nt,-1)
        x_p = x_p.view(b*n*nt,-1,self.n_chunks).transpose(-1,-2)

        q = k = v = self.layer_norm(x_p)
        
        if not self.residual:
            v = x_p

        x_mha = self.MHA(q=q, k=k, v=v)
        x_mha = x_mha.view(b*n,nt,-1)
        x_mha = x_mha.view(b,n,nt,-1)

        if self.residual:
            x_mha = self.mlp_norm(x_mha)
            x = x_res + self.gamma* self.mlp_layer(x_mha)
        else:
            x = self.mlp_layer(x_mha)
   
        if mask is not None:
            mask_update = mask.sum(dim=-1)==mask.shape[-1]

            if x.shape[1]*x.shape[2] > mask_update.shape[-1]:
                mask_update = mask_update.unsqueeze(dim=-1).repeat_interleave(x_mha.shape[-2], dim=-1)
        else:
            mask_update = mask

        return x, mask_update
    


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


class MultiGridBlock(nn.Module):
    def __init__(self, grid_layers, global_levels, model_hparams, decomp_settings, multi_grid_settings, processing_settings=None) -> None: 
        super().__init__()      
        
        self.decomp_layer = decomp_layer_diff_(global_levels, grid_layers, model_hparams, aggregation_fcn=decomp_settings['proj_mode'])

        if processing_settings is not None:
            global_levels_proc = global_levels[global_levels >= processing_settings['lowest_processing_level']]
            self.processing_layer = processing_layer(global_levels_proc, grid_layers, model_hparams, mode=processing_settings['mode'])

        self.mg_layer = multi_grid_layer(global_levels, 
                                         grid_layers, 
                                         model_hparams, 
                                         nh_attention = multi_grid_settings['nh_attention'], 
                                         input_aggregation = 'concat', 
                                         output_projection_overlap = False, 
                                         projection_mode = multi_grid_settings['proj_mode'], 
                                         reduction_mode = 'channel_attention',
                                         with_spatial_attention = multi_grid_settings['with_spatial_attention'],
                                         cascading = multi_grid_settings['cascading'])

    def forward(self, x, indices_layers, indices_batch_dict, mask=None):

        x_levels, drop_mask_levels = self.decomp_layer(x, indices_layers, drop_mask=mask)

        if hasattr(self, 'processing_layer'):
            x_levels, drop_mask_levels = self.processing_layer(x_levels, indices_layers, indices_batch_dict, drop_mask_levels)

        x, drop_mask_levels = self.mg_layer(x_levels, drop_mask_levels, indices_layers, indices_batch_dict)

        return x, drop_mask_levels


def check_value(value, n_repeat):
    if not isinstance(value, list):
        value = [value]*n_repeat
    return value


class ICON_Transformer(nn.Module):
    def __init__(self, model_settings) -> None: 
        super().__init__()

        self.model_settings = load_settings(model_settings, id='model')

        self.check_model_dir()

        self.var_model = self.model_settings["var_model"] if "var_model" in self.model_settings.keys() else False
        self.pos_emb_calc = self.model_settings["pos_emb_calc"]
        self.coord_system = "polar" if "polar" in  self.pos_emb_calc else "cartesian"

        self.grid = xr.open_dataset(self.model_settings['processing_grid'])

        n_grid_levels = self.model_settings['n_grid_levels']

        clon_fov = self.model_settings['clon_fov'] if 'clon_fov' in self.model_settings.keys() else None
        clat_fov = self.model_settings['clat_fov'] if 'clat_fov' in self.model_settings.keys() else None
        self.n_grid_levels_fov = self.model_settings['n_grid_levels_fov'] if 'n_grid_levels_fov' in self.model_settings.keys() else n_grid_levels

        self.scale_input = self.model_settings['scale_input'] if 'scale_input' in self.model_settings.keys() else 1
        self.scale_output = self.model_settings['scale_output'] if 'scale_output' in self.model_settings.keys() else 1
        self.periodic_fov = clon_fov if ('input_periodicty' in self.model_settings.keys() and self.model_settings['input_periodicty']) else None
        self.model_settings['periodic_fov'] = self.periodic_fov

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

        projection_dict = {"kappa_vm": 0.5,
                           "n_theta": 6,
                           "n_dist": self.model_settings["nh"]}

        self.model_settings.update(projection_dict)

        global_levels = torch.tensor(self.model_settings['grid_indices'])
        self.register_buffer('global_levels', global_levels, persistent=False)

        grid_layers = nn.ModuleDict()
        for global_level in global_levels:
            grid_layers[str(int(global_level))] = grid_layer(global_level, mgrids[global_level]['adjc_lvl'], mgrids[global_level]['adjc_mask'], mgrids[global_level]['coords'], coord_system=self.coord_system, periodic_fov=self.periodic_fov)

        input_mapping, input_in_range, input_coordinates, output_mapping, output_in_range, output_coordinates = self.get_grid_mappings(mgrids[0]['coords'], mgrids[0]['coords'])

        self.input_mapping = input_mapping
  
     
        input_projection = self.model_settings['input_projection']

        self.input_layer = input_projection_layer(input_mapping['cell']['cell'], input_in_range['cell']['cell'], input_coordinates['cell'], grid_layers[str(int(global_levels[-1]))], self.model_settings, projection_mode=input_projection)
        
    
        global_levels_block = self.model_settings['global_levels_blocks']
        n_blocks = len(global_levels_block)

        decomp_projection = check_value(self.model_settings['decomp_projection'], n_blocks)

        processing_type = check_value(self.model_settings['processing_type'], n_blocks)
        lowest_processing_levels = check_value(self.model_settings['lowest_processing_levels'], n_blocks)

        multi_nh_attention = check_value(self.model_settings['multi_nh_attention'], n_blocks)
        multi_grids_projection = check_value(self.model_settings['multi_grids_projection'], n_blocks)
        multi_grids_spatial_attention = check_value(self.model_settings['multi_grids_spatial_attention'], n_blocks)
        multi_grids_cascading = check_value(self.model_settings['multi_grids_cascading'], n_blocks)

        self.MGBlocks = nn.ModuleList()

        for k in range(n_blocks):
            decomp_layer_settings = {'proj_mode': decomp_projection[k]}


            processing_settings = {'mode':processing_type[k],
                                 'lowest_processing_level': lowest_processing_levels[k]}


            multi_grid_settings = {'nh_attention': multi_nh_attention[k],
                                'proj_mode': multi_grids_projection[k],
                                'with_spatial_attention': multi_grids_spatial_attention[k],
                                'cascading': multi_grids_cascading[k]}
            
            self.MGBlocks.append(MultiGridBlock(grid_layers, torch.tensor(global_levels_block[k]), self.model_settings, decomp_layer_settings, multi_grid_settings, processing_settings=processing_settings))
        
        

        output_projection = self.model_settings['output_projection']
        self.output_layer = output_layer(output_mapping['cell']['cell'], output_coordinates['cell'], [self.global_levels[-1]], grid_layers, self.model_settings, mode=output_projection)

        self.global_res = self.model_settings['global_res'] if 'global_res' in self.model_settings.keys() else False
     
        self.input_projection = nn.Identity()
  
          # if output mapping is not same as processing:
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
            self.check_pretrained(log_dir_check=self.model_settings['pretrained_model_wo_input'], strict=False, not_match='input')

        self.trained_iterations = trained_iterations


    def forward(self, x, indices_batch_dict=None, debug=False, output_sum=False, drop_mask=None):
        # if global_indices are provided, batches in x are treated as independent
        debug_dict = {}

        if indices_batch_dict is None:
            indices_batch_dict = {'global_cell': self.global_indices,
                                  'local_cell': self.global_indices,
                                   'sample': None,
                                   'sample_level': None,
                                   'output_indices': None}
        else:
            indices_layers = dict(zip(self.global_levels.tolist(),[self.get_global_indices_local(indices_batch_dict['sample'], indices_batch_dict['sample_level'], global_level) for global_level in self.global_levels]))
            indices_0 = self.get_global_indices_local(indices_batch_dict['sample'], indices_batch_dict['sample_level'], 0)

     
        x, drop_mask = self.input_layer(x['cell'], indices_0, indices_layers[self.global_levels.tolist()[-1]] ,drop_mask=drop_mask)

        for k, multi_grid_block in enumerate(self.MGBlocks):
            
            if k==0:
                x, drop_mask = multi_grid_block(x, indices_layers, indices_batch_dict, mask=drop_mask)
            else:
                x, drop_mask = multi_grid_block(x, indices_layers, indices_batch_dict)

        x, x_var = self.output_layer(x, indices_0, indices_layers, indices_batch_dict)
            

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

    
    def apply_on_nc(self, ds, ts, sample_lvl=6, batch_size=8):
    
        normalizer = grid_normalizer(self.model_settings['normalization'])

        if isinstance(ds, str):
            ds = xr.open_dataset(ds)
        
        indices = self.global_indices.view(-1, 4**sample_lvl)

        indices_batches = indices.split(batch_size, dim=0)

        sample_id = 0
        outputs = [[] for _ in range(self.model_settings['n_grid_levels'])]
        for indices_batch in indices_batches:
            sample_idx_min = (sample_id) * (batch_size)

            data = self.get_data_from_ds(ds, ts, self.model_settings["variables_source"], 0, indices_batch)

            sample_indices = torch.arange(sample_idx_min, sample_idx_min + (len(indices_batch)))

            indices_batch_dict = {'global_cell': indices_batch,
                    'local_cell': indices_batch,
                        'sample': sample_indices,
                        'sample_level': sample_lvl* torch.ones((sample_indices.shape[0]), dtype=torch.int)}
            
            data = normalizer(data, self.model_settings["variables_source"])

            with torch.no_grad():
                output = self(data, indices_batch_dict=indices_batch_dict)

            for k, x_lvl in enumerate(output['x']):
                x_lvl = normalizer({'cell': x_lvl}, self.model_settings["variables_target"], denorm=True)
                outputs[k].append(x_lvl['cell'])

            sample_id+=1
        
        output_lvls = []
        for output_lvl in outputs:
           output_lvls.append(torch.concat(output_lvl, dim=0))
        
        output_lvls = torch.stack(output_lvls, dim=0)
        return output_lvls


    def get_data_from_ds(self, ds, ts, variables_dict, global_level_start, global_indices):
        
        sampled_data = {}
        for key, variables in variables_dict.items():
            data_g = []
            for variable in variables:
                data = torch.tensor(ds[variable][ts].values)
                data = data[0] if data.dim() > 1  else data
                data_g.append(data)

            data_g = torch.stack(data_g, dim=-1)

            indices = global_indices

            data_g = data_g[self.input_mapping['cell']['cell'][indices]]
            data_g = data_g.view(indices.shape[0], indices.shape[1], -1, len(variables))

            sampled_data[key] = data_g

        return sampled_data        

    def get_grid_mappings(self, mgrid_coords_input, mgrid_coords_output):
        
        indices_path = os.path.join(self.model_settings["model_dir"],"indices_data.pickle")

        if not os.path.isfile(indices_path):

            input_mapping, input_in_range = get_nh_variable_mapping_icon(self.model_settings['processing_grid'], ['cell'], 
                                        self.model_settings['input_grid'], self.input_data, 
                                        search_raadius=self.model_settings['search_raadius'], 
                                        max_nh=self.model_settings['nh_input'], 
                                        lowest_level=0,
                                        coords_icon=mgrid_coords_input,
                                        scale_input = self.scale_input,
                                        periodic_fov= self.model_settings['clon_fov'] if ('input_periodicty' in self.model_settings.keys() and self.model_settings['input_periodicty']) else None
                                        )

            output_mapping, output_in_range = get_nh_variable_mapping_icon(self.model_settings['processing_grid'], ['cell'], 
                                        self.model_settings['output_grid'], self.output_data, 
                                        search_raadius=self.model_settings['search_raadius'], 
                                        max_nh=1, 
                                        lowest_level=0,
                                        reverse_last=False,
                                        coords_icon=mgrid_coords_output,
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


    def get_nh_indices(self, global_level, global_cell_indices=None, local_cell_indices=None, adjc_global=None):
        
        if adjc_global is not None:
            adjc_global = self.get_adjacent_global_cell_indices(global_level)

        if global_cell_indices is not None:
            local_cell_indices =  global_cell_indices // 4**global_level

        local_cell_indices_nh, mask = helpers.get_nh_of_batch_indices(local_cell_indices, adjc_global)

        return local_cell_indices_nh, mask



    def get_global_indices_global(self, batch_sample_indices, sampled_level_fov, global_level):

        global_indices_sampled  = self.global_indices.view(-1, 4**sampled_level_fov[0])[batch_sample_indices]
        
        return self.get_global_indices_relative(global_indices_sampled, global_level)
    
    def get_global_indices_local(self, batch_sample_indices, sampled_level_fov, global_level):

        global_indices_sampled  = self.global_indices.view(-1, 4**sampled_level_fov[0])[batch_sample_indices]
        global_indices_sampled = self.get_global_indices_relative(global_indices_sampled, global_level)    
        return global_indices_sampled // 4**global_level
    
    def get_global_indices_relative(self, sampled_indices, level):
        return sampled_indices.view(sampled_indices.shape[0], -1, 4**level)[:,:,0]
    

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
        indices = self.global_indices.view(-1,4**int(global_level))[:,0]
        coords = self.cell_coords_global[:,indices]
        return coords


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
    
    seq_level = min([get_max_seq_level(tensor), max_seq_level])
    
    if tensor.dim()==2:
        tensor = tensor.view(tensor.shape[0], -1, 4**(seq_level))
    elif tensor.dim()==3:
        tensor = tensor.view(tensor.shape[0], -1, 4**(seq_level), tensor.shape[-1])
    elif tensor.dim()==4:
        tensor = tensor.view(tensor.shape[0], -1, 4**(seq_level), tensor.shape[-2], tensor.shape[-1])
    elif tensor.dim()==5:
        tensor = tensor.view(tensor.shape[0], -1, 4**(seq_level), tensor.shape[-3], tensor.shape[-2], tensor.shape[-1])
    elif tensor.dim()==6:
        tensor = tensor.view(tensor.shape[0], -1, 4**(seq_level), tensor.shape[-4], tensor.shape[-3], tensor.shape[-2], tensor.shape[-1])

    return tensor

def get_max_seq_level(tensor):
    seq_len = tensor.shape[1]
    max_seq_level_seq = int(math.log(seq_len)/math.log(4))
    return max_seq_level_seq


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

    return torch.gather(x.view(b,-1,e),1, index=local_cell_indices_nh_batch.view(b,-1,1).repeat(1,1,e)).view(b,n,nh,e)

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