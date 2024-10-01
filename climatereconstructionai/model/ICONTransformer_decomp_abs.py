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

    def get_nh(self, x, local_indices, sample_dict, relative_coordinates=True, coord_system=None):

        indices_nh, mask = get_nh_indices(self.adjc, local_cell_indices=local_indices, global_level=int(self.global_level))
        adjc_mask = self.adjc_mask[local_indices]
        mask = torch.logical_or(mask, adjc_mask)
        x = gather_nh_data(x, indices_nh, sample_dict['sample'], sample_dict['sample_level'], int(self.global_level))

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


class multi_grid_attention(nn.Module):
    def __init__(self, projection_level, grid_layers, model_hparams, nh_attention=False, input_aggregation='concat', output_projection_overlap=False) -> None: 
        super().__init__()
        
        # with interpolation to lowest grid

        self.n_grids = len(grid_layers)
        self.model_dim = model_dim = model_hparams['model_dim']
        ff_dim = model_dim
        n_heads = model_hparams['n_heads']
        embedding_dim = model_hparams['pos_emb_dim']
        projection_mode = model_hparams['mga_projection_mode']

        self.projection_level = projection_level
        self.projection_grid_layer = grid_layers[projection_level]
        self.grid_levels = [grid_layer.global_level for grid_layer in grid_layers.values()]

        bins=4
        self.nh_attention = nh_attention

        self.input_aggregation = input_aggregation
        self.output_projection_overlap = output_projection_overlap

        if self.input_aggregation=='sum':
            model_dim_agg = model_dim

        elif self.input_aggregation=='concat':
            model_dim_agg = model_dim*self.n_grids
        
        self.model_dim_agg = model_dim_agg

        if output_projection_overlap:
            self.shared_lin_projection_out = nn.Linear(model_dim_agg, model_dim_agg, bias=False)
        else:
            self.shared_lin_projection_out = nn.Identity()

        self.max_seq_level = model_hparams['max_seq_level']

        if nh_attention:
            self.position_embedder = nh_pos_embedding(self.projection_grid_layer, model_hparams['nh'], model_dim)
        else:
            self.position_embedder = seq_grid_embedding(self.projection_grid_layer, bins, model_hparams['max_seq_level'], model_dim, softmax=False, constant_init=False)

        self.processing=True
        if self.processing:
            self.projection_layers = nn.ModuleList()

            self.layer_norms = nn.ModuleList()

            self.kv_projections = nn.ModuleList()
            self.q_projections = nn.ModuleList()

            self.embedding_layers = nn.ModuleList()
            self.sep_lin_projections_out = nn.ModuleList()

            self.mlp_layers = nn.ModuleList()

            self.gammas = nn.ParameterList()
            self.gammas_mlp = nn.ParameterList()
            
            for k in range(self.n_grids):
                
                if projection_mode == 'vm' and k < self.n_grids-1:
                    self.projection_layers.append(projection_layer_vm(grid_layer_in=grid_layers[list(grid_layers.keys())[k]], 
                                                                      grid_layer_out=self.projection_grid_layer, 
                                                                      model_hparams=model_hparams,
                                                                      uniform_kappa=True,
                                                                      uniform_simga=True))
                elif projection_mode == 'n' and k < self.n_grids-1:
                    self.projection_layers.append(projection_layer_n(grid_layer_in=grid_layers[list(grid_layers.keys())[k]], 
                                                                      grid_layer_out=self.projection_grid_layer, 
                                                                      model_hparams=model_hparams,
                                                                      uniform_simga=True))
                    
                elif projection_mode == 'learned_cont' and k < self.n_grids-1:
                    self.projection_layers.append(layer_layer_projection_nh(grid_layer_in=grid_layers[list(grid_layers.keys())[k]], 
                                                                      grid_layer_out=self.projection_grid_layer, 
                                                                      model_hparams=model_hparams))
                    
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


   # def forward(self, x_levels, nh_att=False, indices_grid_layer=None, batch_dict=None):
   #     if not nh_att:
   #         return self.forward_seq(x_levels, indices_grid_layer=indices_grid_layer, batch_dict=batch_dict, nh_att=nh_att)
   #     else:
   #         return self.forward_nh(x_levels,  indices_grid_layer=None, batch_dict=batch_dict, nh=nh_att)

    def forward(self, x_levels, drop_mask_level, indices_grid_layers, batch_dict):
        
        #grid_levels_in as parameter? -> implement vm projection/simple projection

        '''
        tbd: either softmax embeddings and sum to keep layer dims, for residual, use mean, for gamma-1 and gamma or not
        if nh attention, apply continuous vm embeddings, also for output layers

        angular projection sum -> concat with low res, channel attention -> project back with angles and res add

        simple children -> parent attentions after decomp, xlayers deep, sliding, add new embeddings if more layers


        where channel attention of chunks of feature vectors? after decomp layers!
        Hier channel attention?  Is channel attention immer noch möglich? -> nur nach sum

        '''



        qs = []
        ks = []
        vs = [] 
        
        b,n_proj,f = x_levels[-1].shape

        if self.nh_attention:
            #indices = self.projection_grid_layer.get_nh_indices(indices_grid_layer)[0]
            #rel_coords = self.projection_grid_layer.get_relative_coordinates_from_grid_indices(indices)
            pos_embeddings = self.position_embedder(x_levels[-1], indices_grid_layers[int(self.projection_level)], batch_dict, add_to_x=False)
        else:
            #x, mask, coords = self.projection_grid_layer.get_sections(x_levels[-1], indices_grid_layer, section_level=self.max_seq_level, relative_coordinates=True, return_indices=False)
            #pos_embeddings = self.position_embedder(coords[0], coords[1]).view(b,n_proj,f)
                        
            pos_embeddings = self.position_embedder(x_levels[-1], indices_grid_layers[int(self.projection_level)], add_to_x=False)
            pos_embeddings = sequenize(pos_embeddings,max_seq_level=self.max_seq_level)

        x_levels_output = []


        for i, x in enumerate(x_levels):
            
            drop_mask = drop_mask_level

            b,n,f = x.shape
            
            # should be just dependend on the different shapes

            if isinstance(self.projection_layers[i], nn.Identity):
                x = x.unsqueeze(dim=-2).repeat_interleave(n_proj//n, dim=-2)
            else:
               # x = self.projection_layers[i](x, indices_grid_layers[int(self.grid_levels[i])], indices_grid_layers[int(self.projection_level)], batch_dict)
                x = self.projection_layers[i](x, indices_grid_layers[int(self.grid_levels[i])], indices_grid_layers[int(self.projection_level)], batch_dict)


            x = x.view(b,-1,f)
            
            # use nh instead of sequence?
            if self.nh_attention:
                x, mask, _ = self.projection_grid_layer.get_nh(x, indices_grid_layers[int(self.projection_level)], batch_dict)
                mask = mask.view(b * n_proj, -1)

                if drop_mask is not None:
                    mask = torch.logical_or(mask, drop_mask.view(b * n_proj, -1))

                x_levels_output.append(x[:,:,0])

            else:
                x_levels_output.append(x)
                x = sequenize(x, max_seq_level=self.max_seq_level)

                if drop_mask is not None:
                    mask = sequenize(drop_mask, max_seq_level=self.max_seq_level)
                else:
                    mask=None

            if self.processing:

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

        if self.processing:

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


        for i, x in enumerate(x_levels_output):
            if self.processing:

                output = outputs[0] if self.input_aggregation=='sum' else outputs[i]
                output = self.sep_lin_projections_out[i](output).view(b,n_proj,-1)
         
                x = x + self.gammas[i] * output
                x = x + self.gammas_mlp[i] * self.mlp_layers[i](x)

            x_levels_output[i] = x


        return x_levels_output

   # def forward_nh(self, x_levels, emb_level=None):
   #     pass


class input_layer_simple(nn.Module):
    def __init__(self,  model_hparams) -> None: 
        super().__init__()

        self.v_projection = nn.Linear(len(model_hparams['variables_source']['cell']), model_hparams['model_dim']) 


    def forward(self, x):    
        
        b,n,_,f = x.shape
        
        x = self.v_projection(x)
        
        x = x.view(b,n,-1)

      #  rem_mask = drop_mask.sum(dim=1)==4**self.input_seq_level
       # rem_mask = rem_mask.view(b,-1)
        # clear nans where sequence doesnt have at least on valid value
       # x_nan = x.isnan()

       # x = sequenize(x, max_seq_level=self.input_seq_level)
       # x[rem_mask]=0
        return x



class input_projection_layer(nn.Module):
    def __init__(self, mapping, in_range_mask, coordinates, projection_grid_layer: grid_layer, model_hparams) -> None: 
        super().__init__()

        self.register_buffer("mapping", mapping, persistent=False)
        self.register_buffer("in_range_mask", ~in_range_mask, persistent=False)
        self.register_buffer("coordinates", coordinates, persistent=False)

        model_dim = model_hparams['model_dim']
        pos_emb_calc = model_hparams['pos_emb_calc']
        emb_table_bins = model_hparams['emb_table_bins']

        if 'cartesian' in pos_emb_calc:
            self.coord_system = 'cartesian'
        else:
            self.coord_system = 'polar'

        model_dim = model_hparams['model_dim']
                
        self.projection_layer = projection_layer_learned_cont(model_hparams)

        self.projection_grid_layer = projection_grid_layer
        self.input_seq_level = model_hparams['input_seq_level']
        
        self.lin_projection = nn.Linear(len(model_hparams['variables_source']['cell']), model_dim) 
        

    def forward(self, x, indices_grid_layer, drop_mask=None):    
        
        b, n_out, _, f_in = x.shape

        x = x.view(b, -1, f_in)

        x = self.lin_projection(x)

        x = sequenize(x, max_seq_level=self.input_seq_level)
        indices = sequenize(indices_grid_layer, max_seq_level=self.input_seq_level)
        drop_mask = sequenize(drop_mask, max_seq_level=self.input_seq_level)

        coords_input = self.coordinates[:,self.mapping[indices]]

        n = x.shape[1]
        coords_input = coords_input.view(2,b,n,-1)

        rel_coords = self.projection_grid_layer.get_relative_coordinates_cross(indices, coords_input, coord_system=self.coord_system)

        indices_layers_out_seq = sequenize(indices_grid_layer, max_seq_level=self.input_seq_level)
        rel_coords_out = self.projection_grid_layer.get_relative_coordinates_from_grid_indices(indices_layers_out_seq, coord_system=self.coord_system)
       
        x = self.projection_layer(x, rel_coords, rel_coords_out, mask=drop_mask)

        drop_mask_update = drop_mask.clone()
        drop_mask_update[drop_mask.sum(dim=-1)!=drop_mask.shape[-1]] = False

        x = x.view(b, n_out, -1)
        return x, drop_mask_update.view(b,n_out)


class shifting_mga_layer(nn.Module):
    def __init__(self, global_levels, grid_layers, model_hparams, grid_window=1, nh_attention=True) -> None: 
        super().__init__()
        
        #to do: implement grid window + output projections
        #self.grid_window = model_hparams['max_grid_window']

        self.grid_window = grid_window
        self.grid_layers = grid_layers
        self.x_level_indices = []
        
        self.mga_layers_levels = nn.ModuleDict()
    
        self.global_levels = global_levels
  
        for global_level in self.global_levels:
            grid_layers_ = dict(zip([str(global_level)], [grid_layers[str(global_level)]]))
            self.mga_layers_levels[str(global_level)] = nn.ModuleList(
                [multi_grid_attention(str(global_level), grid_layers_, model_hparams, nh_attention=nh_attention) for _ in range(model_hparams['n_processing_layers'])]
                )

    def forward(self, x_levels, indices_levels, batch_dict):
        # how to deal with overlapping layers? average and project back in mga layers?
        n = 0
        x_levels_output = []
        for global_level in self.global_levels:
            
            mga_layers_level = self.mga_layers_levels[str(global_level)]

            x = x_levels[n:n+(self.grid_window)]
            for mga_layer in mga_layers_level:
                x = mga_layer(x, None, indices_levels, batch_dict)

            x_levels_output += x

            n += 1
        
        return x_levels_output
'''
class shifting_mga_layers(nn.Module):
    def __init__(self, global_levels, grid_layers, model_hparams, n_shifts=1) -> None: 
        super().__init__()

        self.shifting_mga_layers = nn.ModuleList()

        for shift_idx in range(n_shifts):
            self.shifting_mga_layers.append(shifting_mga_layer(global_levels[shift_idx:], grid_layers, model_hparams, grid_window=shift_idx+1))

    def forward(self, x_levels, indices_levels, batch_dict):
        for layer in self.shifting_mga_layers:
            x_levels = layer(x_levels, indices_levels, batch_dict)
        
        return x_levels
'''


class cascading_layer_reduction(nn.Module):
    def __init__(self, global_levels, grid_layers, model_hparams, input_aggregation='concat', output_projection_overlap=True, reduction='sum', nh_attention=False) -> None: 
        super().__init__()
        
        #to do: implement grid window + output projections
        #self.grid_window = model_hparams['max_grid_window']
        self.grid_layers = grid_layers
        self.x_level_indices = []
        self.global_levels = global_levels
        
        self.mga_layers_levels = nn.ModuleDict()
        self.mgaca_layers_levels = nn.ModuleDict()

        self.reduction=reduction
        #self.grid_levels_processed = []
        n = 0
        for global_level in global_levels:
                        
            if n==0:
                grid_layers_keys = [str(global_level)]
            else:
                grid_layers_keys = [str(global_level+1), str(global_level)]
            
            grid_layers_ = dict(zip(grid_layers_keys, [grid_layers[key] for key in grid_layers_keys]))

            self.mga_layers_levels[str(global_level)] = nn.ModuleList(
                [multi_grid_attention(str(global_level), grid_layers_, model_hparams, input_aggregation=input_aggregation, output_projection_overlap=output_projection_overlap, nh_attention=nh_attention) for _ in range(model_hparams['n_processing_layers'])]
                )
            
            if reduction != 'sum' and n > 0:
                self.mgaca_layers_levels[str(global_level)] = multi_grid_channel_attention(2, model_hparams, chunks=4, output_reduction=True)

            n+=1

    def forward(self, x_levels, drop_mask_levels, indices_levels, batch_dict):
        
        n = 0
        for global_level in self.global_levels:
            
            mga_layers_level = self.mga_layers_levels[str(global_level)]

            if n==0:
                x = [x_levels[n]]
            else:
                x = [x, x_levels[n]]

            for mga_layer in mga_layers_level:
                x = mga_layer(x, drop_mask_levels[n], indices_levels, batch_dict)

            if n>0:
                if self.reduction == 'sum':
                    x = torch.stack(x, dim=-1).sum(dim=-1)
                else:
                    x = self.mgaca_layers_levels[str(global_level)](x)[0]
            else:
                x = x[0]
            
            n += 1
        
        return x



class cascading_layer(nn.Module):
    def __init__(self, global_levels, grid_layers, model_hparams, input_aggregation='concat', output_projection_overlap=False, nh_attention=False) -> None: 
        super().__init__()
        
        #to do: implement grid window + output projections
        #self.grid_window = model_hparams['max_grid_window']
        self.grid_layers = grid_layers
        self.x_level_indices = []
        self.global_levels = global_levels
        
        self.mga_layers_levels = nn.ModuleDict()
        global_levels_processed = dict(zip(global_levels, global_levels))
        
        #self.grid_levels_processed = []
        n = 0
        for global_level in global_levels:
            
            #self.grid_levels_processed.append(global_levels_processed)
            
            grid_layers_keys = [str(level) for level in global_levels[:(n+1)]]
            grid_layers_ = dict(zip(grid_layers_keys, [grid_layers[str(global_levels_processed[int(key)])] for key in grid_layers_keys]))


            self.mga_layers_levels[str(global_level)] = nn.ModuleList(
                [multi_grid_attention(str(global_level), grid_layers_, model_hparams, input_aggregation=input_aggregation, output_projection_overlap=output_projection_overlap, nh_attention=nh_attention) for _ in range(model_hparams['n_processing_layers'])]
                )
            
                        
            grid_layers_v = [int(grid_layers_keys[-1]) if int(level)>int(grid_layers_keys[-1]) else level for level in global_levels_processed.values()]
            global_levels_processed = dict(zip(global_levels_processed.keys(),grid_layers_v))

            n+=1

    def forward(self, x_levels, drop_mask_levels, indices_levels, batch_dict):
        
        n = 0
        for global_level in self.global_levels:
            
            mga_layers_level = self.mga_layers_levels[str(global_level)]

            if n==0:
                x_levels_output = [x_levels[n]]
            else:
                x_levels_output.append(x_levels[n])

            for mga_layer in mga_layers_level:
                x_levels_output = mga_layer(x_levels_output, drop_mask_levels[n], indices_levels, batch_dict)

            n += 1
        
        return x_levels_output




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

    dist_weights = normal_dist(dists, dists_0, sigma)
    
    weights = F.softmax(dist_weights, dim=-1)

    #return weights/(weights_norm+1e-10)
    return weights

def get_spatial_projection_weights_n_dist(d_lons, d_lats, dlon_0, sigma_lon, dlat_0, sigma_lat):

    lon_weights = normal_dist(d_lons, dlon_0, sigma_lon)
    lat_weights = normal_dist(d_lats, dlat_0, sigma_lat)
    
    weights = lon_weights * lat_weights

    weights = F.softmax(weights, dim=-2)

    #return weights/(weights_norm+1e-10)
    return weights

def get_spatial_projection_weights_vm_dist(phis, dists, phi_0, kappa_vm, dists_0, sigma):

    vm_weights = von_mises(phis, phi_0, kappa_vm)
    dist_weights = normal_dist(dists, dists_0, sigma)
    
    vm_weights[dist_weights[:,:,:,:,:,0]==1] = torch.exp(kappa_vm)

    weights = vm_weights * dist_weights

    weights = F.softmax(weights, dim=-2)

    #return weights/(weights_norm+1e-10)
    return weights


def von_mises(thetas, theta_offsets, kappa):

    if not torch.is_tensor(theta_offsets):
        theta_offsets = torch.tensor(theta_offsets)

    vm_norm = 1
    vm = vm_norm * torch.exp(kappa * torch.cos(thetas.unsqueeze(dim=-1) - theta_offsets.unsqueeze(dim=-2)).unsqueeze(dim=-1))
    return vm


def normal_dist(distances, distances_offsets, sigma):

    if not torch.is_tensor(distances_offsets):
        distances_offsets = torch.tensor(distances_offsets)

    if sigma.dim()==3:
        sigma = sigma.unsqueeze(dim=-1).unsqueeze(dim=-1)

    norm = 1
    nd = norm * torch.exp(-0.5 * ((distances.unsqueeze(dim=-1) - distances_offsets.unsqueeze(dim=-2)).unsqueeze(dim=-1) / sigma) ** 2)
    return nd

class angular_embedder(nn.Module):
    def __init__(self, n_bins, emb_dim) -> None: 
        super().__init__()
 
        self.thata_embedder = helpers.PositionEmbedder_phys(-torch.pi, torch.pi, n_bins, n_heads=emb_dim, special_token=True)

    def forward(self, thetas, dist_0_mask):
        return  self.thata_embedder(thetas, special_token_mask=dist_0_mask)


class decomp_layer_angular_embedding(nn.Module):
    def __init__(self, global_levels, grid_layers: dict, model_hparams) -> None: 
        super().__init__()
        

        n_bins = 4
        emb_dim = model_hparams['model_dim']
        n_bins_input = 4
        self.global_levels = global_levels
        self.grid_layers = grid_layers

        self.hier_angular_embedders = nn.ModuleDict()
        for global_level in self.global_levels:
            self.hier_angular_embedders[str(global_level)] = seq_grid_embedding(grid_layers[str(global_level)], n_bins, 1, emb_dim, softmax=True, constant_init=False)
            # with layernorm or softmax?

        # also init vm kappas?
        # only sum if valid

    # with channel attention
    def forward(self, x, indices_layers, drop_mask=None):
        
        x_levels=[]
        drop_masks_level = []
        for global_level in self.global_levels[::-1]:
            
            drop_masks_level.append(drop_mask)
            indices_grid_layer = indices_layers[global_level]
            b,n,e = x.shape

            if global_level < self.global_levels[0]:
                # map to midpoint (for rigid grids)
                x_sections = self.hier_angular_embedders[str(global_level)](x, indices_layers[global_level], drop_mask=drop_mask)
                x_sections = sequenize(x_sections, max_seq_level=1)

                if drop_mask is not None:
                    drop_mask = sequenize(drop_mask, max_seq_level=1)

                    x_sections[drop_mask]=0

                x_sections = x_sections.sum(dim=-2, keepdim=True)

                #if global_level==0:
                    #implement from input layer
                    #pass
                    
                #x_sections, _, _ = self.grid_layers[str(global_level)].get_sections(x, indices_grid_layer, section_level=1)
                x, _, _ = self.grid_layers[str(global_level)].get_sections(x, indices_grid_layer, section_level=1, return_indices=False)

                x_res = (x - x_sections)
                
                if drop_mask is not None:
                    x_res[drop_mask] = 0 
                    drop_mask = drop_mask.sum(dim=-1)==4

                x_levels.append(x_res.view(b,n,-1))

            else:
                x_levels.append((x).view(b,n,-1))

            x = x_sections.squeeze(dim=-2)

        if len(self.global_levels) == 0:
            x_levels[0] = x

        x_levels = x_levels[::-1]

        if drop_mask is not None:
            drop_masks_level = drop_masks_level[::-1]

            #assert int(drop_mask.sum())==0, "still nans in data, number of grid layers might not be sufficient"

        return x_levels, drop_masks_level



class decomp_layer_diff(nn.Module):
    def __init__(self, global_levels, grid_layers: dict) -> None: 
        super().__init__()

        self.grid_layers = grid_layers
        
        self.max_level = list(grid_layers.keys())[-1]
        self.global_levels = global_levels

    def forward(self, x, indices_layers, drop_mask=None):
        
        drop_masks_level = []
        x_levels=[]
        #from fine to coarse
        for global_level in self.global_levels[::-1]:
            
            drop_masks_level.append(drop_mask)

            b,n,e = x.shape

            if global_level < self.global_levels[0]:
                x_sections, _, _ = self.grid_layers[str(global_level)].get_sections(x, indices_layers[global_level], section_level=1, return_indices=False)

                weights = torch.ones(x_sections.shape[:-1])

                if drop_mask is not None:
                    weights[drop_mask.view(weights.shape)] = 0
                    drop_mask = sequenize(drop_mask, max_seq_level=1)
                    x_sections[drop_mask]=0

                weights = weights/(weights.sum(dim=-1, keepdim=True)+1e-10)
                x_sections_f = (x_sections*weights.unsqueeze(dim=-1)).sum(dim=-2, keepdim=True)
                
                x_res = (x_sections - x_sections_f)
                          
                if drop_mask is not None:
                    x_res[drop_mask] = 0 
                    drop_mask = drop_mask.sum(dim=-1)==4

                x_levels.append(x_res.view(b,n,-1))
            else:
                x_levels.append((x).view(b,n,-1))           

            x = x_sections_f.squeeze(dim=-2)


        if len(self.global_levels) == 0:
            x_levels[0] = x

        x_levels = x_levels[::-1]
        drop_masks_level = drop_masks_level[::-1]

        return x_levels, drop_masks_level


class processing_layer(nn.Module):
    #to be modifies
    def __init__(self, global_levels, grid_layers: dict, model_hparams, mode='nh_VM') -> None: 
        super().__init__()

        output_dim = len(model_hparams['variables_target']['cell'])*(1+model_hparams['var_model'])
        self.var_projection = model_hparams['var_model']

        self.global_levels = global_levels
        self.mode = mode

        self.processing_layers = nn.ModuleDict()

        for global_level in global_levels:

            if mode == 'nh_ca_VM':
                self.processing_layers[str(global_level)] = nh_vm_channel_attention(grid_layers[str(global_level)], model_hparams)
            elif mode == 'nh_ca':
                self.processing_layers[str(global_level)] = nh_channel_attention(grid_layers[str(global_level)], model_hparams)
            elif mode == 'ca':
                self.processing_layers[str(global_level)] = channel_attention(grid_layers[str(global_level)], model_hparams)
            elif mode == 'linear':
                self.processing_layers[str(global_level)] = nn.Linear(model_hparams['model_dim'], model_hparams['model_dim'], bias=False)

        self.grid_layers = grid_layers

            
    def forward(self, x_levels, indices_layers, batch_dict, output_coords=None):
        
        x_output = []


        for k, global_level in enumerate(self.global_levels):
            
            if self.mode == 'linear' or self.mode == 'ca':
                x = self.processing_layers[str(global_level)](x_levels[k])
            else:
                x = self.processing_layers[str(global_level)](x_levels[k], indices_layers[global_level], batch_dict)


            x_output.append(x)

        return x_output


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
    def __init__(self, max_seq_level, n_bins, emb_dim, softmax=True, constant_init=False) -> None: 
        super().__init__()
        #uses hierarical grid embeddings
        #vm to interpolate -> for nh attention

        self.max_seq_level = max_seq_level

        self.softmax = softmax

        self.angular_embedders= nn.ParameterList()
        for _ in range(max_seq_level):
            self.angular_embedders.append(helpers.PositionEmbedder_phys(-torch.pi, torch.pi, n_bins, n_heads=emb_dim, special_token=True, constant_init=constant_init))

    def forward(self, x_level, indices_layer, seq_level, add_to_x=True, drop_mask=None):
        b,n,f = x.shape
        
        seq_level = min([get_max_seq_level(x), self.seq_level])

        for i in range(seq_level):
            x, mask, rel_coords, indices_layer = self.grid_layer.get_sections(x, indices_layer, section_level=1, relative_coordinates=True, return_indices=True, coord_system="polar")
            
            embeddings = self.angular_embedders[i](rel_coords[1], special_token_mask=rel_coords[0]<1e-6)

            if self.softmax:
                #b_i, n_i, s_i  = torch.where(drop_mask.view(x.shape[:-1],1))
             #   embeddings = embeddings.masked_fill(drop_mask.view(x.shape[:-1],1), float("-inf"))
                #emb_isnan = embeddings.isnan()
                if drop_mask is not None:
                    embeddings[drop_mask.view(x.shape[:-1],1)] = -100
                embeddings = F.softmax(embeddings, dim=-2-i)

            if i > 0:
                scale = scale.view(x.shape) * embeddings#.unsqueeze(dim=-(1+i))
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
                #b_i, n_i, s_i  = torch.where(drop_mask.view(x.shape[:-1],1))
             #   embeddings = embeddings.masked_fill(drop_mask.view(x.shape[:-1],1), float("-inf"))
                #emb_isnan = embeddings.isnan()
                if drop_mask is not None:
                    embeddings[drop_mask.view(x.shape[:-1],1)] = -100
                embeddings = F.softmax(embeddings, dim=-2-i)

            if i > 0:
                scale = scale.view(x.shape) * embeddings#.unsqueeze(dim=-(1+i))
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
                if mode=='learned':
                    self.projection_layers[str(global_level)] = projection_layer_learned(grid_layers[str(global_level)], grid_layers["0"], model_hparams)
                elif mode=='learned_cont':
                    self.projection_layers[str(global_level)] = layer_layer_projection_nh(grid_layers[str(global_level)], grid_layers["0"], model_hparams)
                elif mode=='VM':
                    self.projection_layers[str(global_level)] = projection_layer_vm(grid_layers[str(global_level)], grid_layers["0"], model_hparams)
                elif mode=='n':
                    self.projection_layers[str(global_level)] = projection_layer_n(grid_layers[str(global_level)], grid_layers["0"], model_hparams, uniform_simga=False)
            
            self.lin_projection_layers[str(global_level)] = nn.Linear(model_hparams['model_dim'], output_dim, bias=False)


        self.grid_layers = grid_layers

            
    def forward(self, x_levels, indices_layers, batch_dict, coords_output=None):
        
        x_output_mean = []
        x_output_var = []

        x_level_out = x_levels[-1]

        coords_output = self.coordinates[:,self.mapping[indices_layers[0]]]
        n_c, b, n, n_nh = coords_output.shape

        for k, global_level in enumerate(self.global_levels):
            
            if global_level>0:
                if self.mode == 'learned':
                    x = self.projection_layers[str(global_level)](x_levels[k], x_level_out, indices_layers[global_level], indices_layers[0], batch_dict, coords_output=coords_output)
                elif self.mode == 'simple':
                    x = x_levels[k]
                else:
                    x = self.projection_layers[str(global_level)](x_levels[k], indices_layers[global_level], indices_layers[0], batch_dict, coords_output=coords_output)
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


class layer_layer_projection_nh(nn.Module):
    def __init__(self, grid_layer_in: grid_layer, grid_layer_out: grid_layer, model_hparams: dict) -> None: 
        super().__init__()

        pos_emb_calc = model_hparams['pos_emb_calc']

        if 'cartesian' in pos_emb_calc:
            self.coord_system = 'cartesian'
        else:
            self.coord_system = 'polar'

        self.grid_layer_in = grid_layer_in
        self.grid_layer_out = grid_layer_out

        self.seq_level = grid_layer_in.global_level - grid_layer_out.global_level
      
        self.projection_layer = projection_layer_learned_cont(model_hparams)

            
    def forward(self, x_level_in, indices_layers_in, indices_layers_out, batch_dict, coords_output=None):
         
        x_nh, mask, rel_coords_nh = self.grid_layer_in.get_nh(x_level_in, indices_layers_in, batch_dict, relative_coordinates=True, coord_system=self.coord_system)

        indices_layers_out_seq = sequenize(indices_layers_out, max_seq_level=self.seq_level)

        if coords_output is None:
            rel_coords_out = self.grid_layer_out.get_relative_coordinates_from_grid_indices(indices_layers_out_seq, coord_system=self.coord_system)
        else:
            n_c, b, n, nh = coords_output.shape
            coords_output = coords_output.view(n_c, b ,-1)
        
            coords_output = coords_output.view(n_c, b, -1, 4**self.seq_level)
            rel_coords_out = self.grid_layer_out.get_relative_coordinates_cross(indices_layers_out_seq, coords_output, coord_system=self.coord_system)

        x = self.projection_layer(x_nh, rel_coords_nh, rel_coords_out, mask)

        return x

class projection_layer_learned_cont(nn.Module):
    def __init__(self, model_hparams: dict) -> None: 
        super().__init__()
        #

        model_dim = model_hparams['model_dim']
        pos_emb_calc = model_hparams['pos_emb_calc']
        emb_table_bins = model_hparams['emb_table_bins']

        if 'cartesian' in pos_emb_calc:
            self.coord_system = 'cartesian'
        else:
            self.coord_system = 'polar'
      
        self.positon_embedder = position_embedder(0,0, emb_table_bins, model_dim, pos_emb_calc=pos_emb_calc)

            
    def forward(self, x, rel_coords_in, rel_coords_out, mask=None):
         
        pos_embeddings_in = self.positon_embedder(rel_coords_in[0], rel_coords_in[1])
        pos_embeddings_out = self.positon_embedder(rel_coords_out[0], rel_coords_out[1])

        b, n_in, n_seq, f = x.shape 
        n_out = pos_embeddings_out.shape[-2]
        
        if mask is not None:
            pos_embeddings_in[mask] = -1e30 if pos_embeddings_in.dtype == torch.float32 else -1e4

        pos_embeddings_out = pos_embeddings_out.view(b, n_in, -1, f)
        
        weights = pos_embeddings_in.unsqueeze(dim=-3) * pos_embeddings_out.unsqueeze(dim=-2)

        weights = F.softmax(weights, dim=-2)

        return (weights * x.unsqueeze(dim=-3)).sum(dim=-2)


class projection_layer_learned(nn.Module):
    def __init__(self, grid_layer_in: grid_layer, grid_layer_out: grid_layer, model_hparams: dict) -> None: 
        super().__init__()
        #

        model_dim = model_hparams['model_dim']

        self.grid_layer_in = grid_layer_in
        self.grid_layer_out = grid_layer_out

        self.seq_level_proj = 0
        bins=4
        self.seq_level = self.seq_level_proj + grid_layer_in.global_level - grid_layer_out.global_level
       # self.input_embedder = nh_pos_embedding(grid_layer_in, model_hparams['nh'], model_dim)
        self.output_embedder = seq_grid_embedding(grid_layer_out, bins, self.seq_level, model_dim, softmax=True, constant_init=False)

       # self.positon_embedder = position_embedder(0,0, 12, model_dim, pos_emb_calc='km_cartesian')

        #x, mask, rel_coords, indices_layer = self.grid_layer.get_sections(x, indices_layer, section_level=1, relative_coordinates=True, return_indices=True, coord_system="polar")
        #self.gamma = nn.Parameter(torch.ones(model_dim)*1e-6, requires_grad=True)

        self.input_layer_norm = nn.Identity() #nn.LayerNorm(model_dim)
        self.output_layer_norm = nn.Identity()
            
    def forward(self, x_level_in, x_level_out, indices_layers_in, indices_layers_out, batch_dict, coords_output=None):

        #x_nh, mask, _ = self.grid_layer_in.get_nh(x_level_in, indices_layers_in, batch_dict)
        b,n,f = x_level_in.shape
        pos_embeddings = self.output_embedder(x_level_out, indices_layers_out, add_to_x=False)
  
    
        pos_embeddings = pos_embeddings.view(b,-1, 4**(self.seq_level), f)
        x_level_in = x_level_in.view(b, -1, 1, f)
        
        x_out = (pos_embeddings * x_level_in)
        return x_out.view(b,-1,f)


class projection_layer_simple(nn.Module):
    def __init__(self, grid_layers: dict, model_hparams: dict, var_projection=False, output_dim=None) -> None: 
        super().__init__()

        model_dim = model_hparams['model_dim']
        self.var_projection = var_projection

        self.grid_layers = grid_layers

        if output_dim is None:
            output_dim = len(model_hparams['variables_target']['cell'])
                
        self.layers = nn.ModuleList()
        for _ in grid_layers.keys():
            self.layers.append(nn.Linear(model_dim, output_dim + self.var_projection*output_dim, bias=False))

            
    def forward(self, x_levels):
        
        x_output_mean = []
        x_output_var = []

        for k, x in enumerate(x_levels):
            
            x = self.layers[k](x)
           
            if self.var_projection:
                x, x_var = x.split(x.shape[-1] // 2, dim=-1)
                x_output_mean.append(x)
                x_output_var.append(nn.functional.softplus(x_var))
            else:
                x_output_mean.append(x)

        return x_output_mean, x_output_var


class projection_layer_vm(nn.Module):
    def __init__(self, grid_layer_in, grid_layer_out, model_hparams: dict, uniform_kappa=True, uniform_simga=False) -> None: 
        super().__init__()

        self.grid_layer_in = grid_layer_in
        self.grid_layer_out = grid_layer_out
        self.global_level_diff = self.grid_layer_in.global_level - self.grid_layer_out.global_level

        sigma_dim = 1 if uniform_simga else model_hparams['model_dim']
        self.simga_d = nn.Parameter(torch.ones(sigma_dim) * grid_layer_in.min_dist/2, requires_grad=True)
        
        kappa_dim = 1 if uniform_kappa else model_hparams['model_dim']
        self.kappa_vm = nn.Parameter(torch.ones(kappa_dim) * model_hparams["kappa_vm"], requires_grad=True)

    def forward(self, x_level_in, indices_layers_in, indices_layers_out, batch_dict, coords_output=None):
        
        if coords_output is None:
            output_coords = self.grid_layer_out.get_coordinates_from_grid_indices(indices_layers_out)

        output_coords = output_coords.view(2, output_coords.shape[1], -1, 4**self.global_level_diff)
      
        x = self.grid_layer_in.get_projection_cross_vm(x_level_in, indices_layers_in, batch_dict, output_coords, self.simga_d, self.kappa_vm) 
        
        x = x.view(x.shape[0], -1, x.shape[-1])

        return x

class projection_layer_n(nn.Module):
    def __init__(self, grid_layer_in, grid_layer_out, model_hparams: dict, uniform_simga=False) -> None: 
        super().__init__()

        self.grid_layer_in = grid_layer_in
        self.grid_layer_out = grid_layer_out
        self.global_level_diff = self.grid_layer_in.global_level - self.grid_layer_out.global_level

        sigma_dim = 1 if uniform_simga else model_hparams['model_dim']
        self.simga = nn.Parameter(torch.ones(sigma_dim) * grid_layer_in.min_dist*200, requires_grad=True)
        

    def forward(self, x_level_in, indices_layers_in, indices_layers_out, batch_dict, coords_output=None):
        
        if coords_output is None:
            output_coords = self.grid_layer_out.get_coordinates_from_grid_indices(indices_layers_out)

        output_coords = output_coords.view(2, output_coords.shape[1], -1, 4**self.global_level_diff)
      
        x = self.grid_layer_in.get_projection_cross_n(x_level_in, indices_layers_in, batch_dict, output_coords, self.simga, self.simga) 
        
        x = x.view(x.shape[0], -1, x.shape[-1])

        return x

class projection_layer_vm_learned(nn.Module):
    def __init__(self, grid_layer_in, grid_layer_out, model_hparams: dict) -> None: 
        super().__init__()

        self.grid_layer_in = grid_layer_in
        self.grid_layer_out = grid_layer_out
        self.global_level_diff = self.grid_layer_in.global_level - self.grid_layer_out.global_level

        # channel attention is good here!
        self.n_vm = 4
        self.n_d  = 6

        self.kappa_scan = nn.Parameter(torch.tensor(model_hparams["kappa_vm"], dtype=float), requires_grad=True)
    
        self.simga_scan = nn.Parameter(grid_layer_in.min_dist, requires_grad=True)
                       
        self.proj_amp = nn.Sequential(nn.Linear(self.n_d, 1, bias=False))
        self.proj_wl = nn.Sequential(nn.Linear(self.n_d, 1, bias=False))

        self.min_val = self.grid_layer_in.min_dist/10
        self.max_val = self.grid_layer_in.max_dist
                    
        self.dists_0 = nn.Parameter(torch.linspace(0, self.max_val, self.n_d).unsqueeze(dim=-1).repeat_interleave(self.n_vm, dim=-1), requires_grad=False)
        self.phi_0 = nn.Parameter(torch.linspace(-torch.pi, torch.pi, self.n_vm + 1)[:-1].unsqueeze(dim=0).repeat_interleave(self.n_d, dim=0), requires_grad=False)

        self.dist_weights_phi = nn.Parameter(torch.arange(self.n_d, 1., -1), requires_grad=True)

        self.sigmoid = nn.Sigmoid()


    def forward(self, x_level_in, x_level_out, indices_layers_in, indices_layers_out, batch_dict, coords_output=None):
        
        if coords_output is None:
            output_coords = self.grid_layer_out.get_coordinates_from_grid_indices(indices_layers_out)

        x = self.grid_layer_in.get_projection_nh(x_level_in, indices_layers_in, batch_dict, self.phi_0, self.dists_0, self.simga_scan, self.kappa_scan)

        b,n,nd,nvm,f = x.shape
        x = x.view(b*n,nd,nvm,f)

        x_offset, x_env = x.split([1,self.n_d-1], dim=1) #split in offset and rest
        x_offset = x[:,[0],:,:].mean(dim=[-2])

        direction = (F.softmax(self.dist_weights_phi, dim=0).view(1,self.n_d-1,1,1) * x_env).sum(dim=1)
        direction_weights = F.softmax(direction, dim=-2).transpose(-1,-2)
        direction = torch.matmul(direction_weights, torch.cos(self.phi_0[0].view(-1,1)))
        direction = torch.acos(direction) + torch.pi/2

        x = (x * direction_weights.view(b*n,1,nvm,f)).sum(dim=-2)

        wl = F.softmax(x, dim=-2) * self.dists_0[:,0].view(-1,1)
        wl = self.sigmoid(self.proj_wl(wl.transpose(-1,-2))) + self.min_val

        amp = self.proj_amp((x - x_offset).transpose(-1,-2))

        output_coords = output_coords.view(2, output_coords.shape[1], -1, 4**self.global_level_diff)

        distances, phis = get_relative_positions(output_coords[:,:,:,0], output_coords, polar=True)
        distances = distances.transpose(-1,-2)
        phis = phis.transpose(-1,-2)

        amp = amp.view(b,n,1,f)
        wl = wl.view(b,n,1,f)
        x_offset = x_offset.view(b,n,1,f)
        direction = direction.view(b,n,1,f)

        x = amp*torch.cos(2*torch.pi/wl*torch.cos(direction-phis)*distances) + x_offset

        x = x.view(b,-1,f)
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
        self.coord_system = "polar" if "polar" in  self.pos_emb_calc else "cartesian"

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

        projection_dict = {"kappa_vm": .1,
                           "n_theta": 6,
                           "n_dist": self.model_settings["nh"]}

        self.model_settings.update(projection_dict)

        grid_layers = nn.ModuleDict()
        self.global_levels = []

        for grid_level_idx in range(n_grid_levels):

            global_level = grid_level_idx
            grid_layers[str(global_level)] = grid_layer(global_level, mgrids[global_level]['adjc_lvl'], mgrids[global_level]['adjc_mask'], mgrids[global_level]['coords'], coord_system=self.coord_system, periodic_fov=self.periodic_fov)
            self.global_levels.append(global_level)

        self.global_levels.sort(reverse=True)
        
        input_mapping, input_in_range, input_coordinates, output_mapping, output_in_range, output_coordinates = self.get_grid_mappings(mgrids[0]['coords'],mgrids[0]['coords'])

        if 'input_learned' in self.model_settings.keys() and self.model_settings['input_learned']:
            self.input_layer = input_projection_layer(input_mapping['cell']['cell'], input_in_range['cell']['cell'], input_coordinates['cell'], grid_layers["0"], self.model_settings)
            self.input_learned = True
        else:
            self.input_layer = input_layer_simple(self.model_settings)
            self.input_learned = False


        #self.reduction_layer = multi_grid_channel_attention(len(self.global_levels), self.model_settings, chunks=8, output_reduction=True)


 #       self.ca_layer = multi_grid_channel_attention(1, self.model_settings, chunks=4, output_reduction=True)

        #self.cascading_layer_reduction = cascading_layer_reduction(self.global_levels, grid_layers, self.model_settings, reduction='ca')
        

       # self.ca_reduction_layer = multi_grid_channel_attention(len(self.global_levels),self.model_settings, chunks=4, output_reduction=True)

        # cross channel attention + add everything
        #decomp, single layer nh attention


        # decomp with angular or without
        # cascade seqences
        # reduce sum or channel attention or sum and channel attention?

        # decomp with angular or without
        # single layer channel attention?
        # nh processing -> of each layer or cascade?
        # seq processing + nh cascade?
        # 

        # model 1
        self.decomp_layers = nn.ModuleList()
        self.processing_layers = nn.ModuleList()
        self.cascading_layers = nn.ModuleList()

        self.initial_decomp_layer = decomp_layer_angular_embedding(self.global_levels, grid_layers, self.model_settings)

        self.fill_layer = cascading_layer(self.global_levels, grid_layers, self.model_settings, input_aggregation='concat', output_projection_overlap=False) # to fill values

        processing_mode = self.model_settings['processing_mode'] if 'processing_mode' in self.model_settings.keys() else 'ca'
        reduction_mode = self.model_settings['reduction_mode'] if 'reduction_mode' in self.model_settings.keys() else 'ca'
        cascading_nh_attention = self.model_settings['cascading_nh_attention'] if 'cascading_nh_attention' in self.model_settings.keys() else True

        for _ in range(self.model_settings['n_layers']):

            self.decomp_layers.append(decomp_layer_angular_embedding(self.global_levels, grid_layers, self.model_settings))

            self.processing_layers.append(processing_layer(self.global_levels, grid_layers, self.model_settings, mode=processing_mode))
            self.cascading_layers.append(cascading_layer_reduction(self.global_levels, grid_layers, self.model_settings, reduction=reduction_mode, nh_attention=cascading_nh_attention))

        self.multi_level_output = self.model_settings["n_grid_levels_out"]>1
        if self.multi_level_output:
            self.output_decomp_layer = decomp_layer_angular_embedding(self.global_levels, grid_layers, self.model_settings)
            self.output_projection_layer = cascading_layer(self.global_levels, grid_layers, self.model_settings, input_aggregation='concat', output_projection_overlap=False, nh_attention=True)
            self.output_layer = output_layer(output_mapping['cell']['cell'], output_coordinates['cell'], self.global_levels, grid_layers, self.model_settings, mode='simple')

            #alternative:
            #processing + projection layer
        else:
            self.output_layer = output_layer(output_mapping['cell']['cell'], output_coordinates['cell'], [self.global_levels[-1]], grid_layers, self.model_settings, mode='learned_cont')


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
            indices_layers = dict(zip(self.global_levels,[self.get_global_indices_local(indices_batch_dict['sample'], indices_batch_dict['sample_level'], global_level) for global_level in self.global_levels]))
        
        if self.input_learned:
            x, drop_mask = self.input_layer(x['cell'], indices_layers[0] ,drop_mask=drop_mask)
        else:
            x = self.input_layer(x['cell'])

        x_levels, drop_mask_levels = self.initial_decomp_layer(x, indices_layers, drop_mask=drop_mask)

        x_levels = self.fill_layer(x_levels, drop_mask_levels, indices_layers, indices_batch_dict)

        drop_mask_levels = [None for _ in range(len(x_levels))]
        x = torch.stack(x_levels, dim=-1).sum(dim=-1)

        for k in range(len(self.cascading_layers)):
            x_levels, _ = self.decomp_layers[k](x, indices_layers)
            x_levels = self.processing_layers[k](x_levels, indices_layers, indices_batch_dict)
            x = self.cascading_layers[k](x_levels, drop_mask_levels, indices_layers, indices_batch_dict)


        if self.multi_level_output:
            x_levels, _ = self.output_decomp_layer(x, indices_layers)
            x_levels = self.output_projection_layer(x_levels, drop_mask_levels, indices_layers, indices_batch_dict)
            x, x_var = self.output_layer(x_levels, indices_layers, indices_batch_dict)
        else:
            x, x_var = self.output_layer([x], indices_layers, indices_batch_dict)
            


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

 #  def precompute_distances(self):
 #       self.eval()

 #       with torch.no_grad():


    #currently cell only
    
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

            data_g = data_g[indices]
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