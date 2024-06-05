import json,os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr


from ..utils.io import load_ckpt, load_model
import climatereconstructionai.model.transformer_helpers as helpers
from climatereconstructionai.utils.grid_utils import get_distance_angle, get_coords_as_tensor, get_mapping_to_icon_grid, get_nh_variable_mapping_icon
from .. import transformer_training as trainer
from ..utils.normalizer import grid_normalizer
from ..utils.optimization import grid_dict_to_var_dict

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




class nha_layer(nn.Module):
    def __init__(self, input_dim, model_dim, ff_dim, n_heads=4, dropout=0, input_mlp=False, output_dim=None, activation=nn.SiLU(), q_res=True, pos_emb_type='bias', pos_embedder=None, pos_emb_dim=None, kv_dropout=0, qkv_bias=True, v_proj=True) -> None: 
        super().__init__()

        self.model_dim = model_dim
        self.n_heads = n_heads
        self.pos_emb_type = pos_emb_type
        self.kv_dropout = kv_dropout
        self.qkv_bias = qkv_bias

        if input_mlp:
            self.input_mlp = nn.Sequential(
                nn.Linear(input_dim, model_dim, bias=False),
                nn.LayerNorm(model_dim)
            )
            input_dim = model_dim
        else:
            self.input_mlp = nn.Identity()
        
        if output_dim is not None:
            self.output_layer = nn.Sequential(
                    nn.Linear(model_dim, ff_dim, bias=False),
                    activation,
                    nn.Linear(ff_dim, output_dim, bias=False)
                )
        else:
            self.output_layer = nn.Identity()

        self.pos_embedder = pos_embedder

        if self.pos_emb_type=='bias':
            self.emb_proj_bias = nn.Linear(pos_emb_dim, n_heads, bias=False)

        elif self.pos_emb_type=='context':
            self.emb_proj_q = nn.Linear(pos_emb_dim, model_dim // n_heads, bias=False)
            self.emb_proj_k = nn.Linear(pos_emb_dim, model_dim // n_heads, bias=False)
            self.emb_proj_v = nn.Linear(pos_emb_dim, model_dim // n_heads, bias=False)

        self.MHA = helpers.MultiHeadAttentionBlock(
            model_dim, model_dim, n_heads, input_dim=input_dim, qkv_proj=True, qkv_bias=qkv_bias, v_proj=v_proj
            )           

        self.mlp_layer = nn.Sequential(
            nn.Linear(model_dim, ff_dim, bias=False),
            activation,
            nn.Linear(ff_dim, model_dim, bias=False)
        )
             
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(model_dim) if model_dim > 1 else nn.Identity()
        self.norm2 = nn.LayerNorm(model_dim)

        self.q_res = q_res

    def forward(self, x: torch.tensor, xq=None, xv=None, mask=None, pos=None):    
            
        x = self.input_mlp(x)

        b, n, nh, e = x.shape
        x = x.reshape(b*n,nh,e)
        q = k = v = x

        if xq is not None:
            xq = self.input_mlp(xq)
            b, nq = xq.shape[:2]
            q = xq.reshape(b*nq,-1,xq.shape[-1])

        if xv is not None:
            xv = self.input_mlp(xv)
            b, nv = xv.shape[:2]
            v = xv.reshape(b*nv,-1,xv.shape[-1])

        if self.kv_dropout > 0:
            pos1 = pos[0].view(b*n, pos[0].shape[-2], nh)
            pos2 = pos[1].view(b*n, pos[1].shape[-2], nh)           

            indices_keep = (torch.randperm((nh)*(x.shape[0]), device=x.device) % (nh-1)).view(x.shape[0], nh)[:, :int(math.ceil((1- self.kv_dropout)*nh))]
       
            k = torch.gather(k, dim=1, index=indices_keep.unsqueeze(dim=-1).repeat(1,1,k.shape[-1]))
            v = torch.gather(v, dim=1, index=indices_keep.unsqueeze(dim=-1).repeat(1,1,k.shape[-1]))
            pos1 = torch.gather(pos1, dim=-1, index=indices_keep.view(b*n,1,-1).repeat(1, pos1.shape[1],1))
            pos2 = torch.gather(pos2, dim=-1, index=indices_keep.view(b*n,1,-1).repeat(1, pos2.shape[1],1))

            pos = (pos1.view(b, n, pos1.shape[-2],-1), pos2.view(b, n, pos1.shape[-2],-1))

            if mask is not None:
                mask = mask.view(b*n, mask.shape[-2], nh)
                mask = torch.gather(mask, dim=-1, index=indices_keep.view(b*n,1,-1))
                mask = mask.view(b, n, mask.shape[-2],-1)

        if self.qkv_bias:

            if self.pos_emb_type =='context' and pos is not None:
            # aq = self.emb_proj_q(pos[0], pos[1], self.pos_embedder)
                pos_embedding = self.pos_embedder(pos[0], pos[1])

                ak = self.emb_proj_k(pos_embedding)
                av = self.emb_proj_v(pos_embedding)

                att_out, att = self.MHA(q=q, k=k, v=v, aq=None, ak=ak, av=av, return_debug=True, mask=mask) 

            elif self.pos_emb_type =='bias' and pos is not None:
                pos_embedding = self.pos_embedder(pos[0], pos[1])
                bias = self.emb_proj_bias(pos_embedding)

                att_out, att = self.MHA(q=q, k=k, v=v, bias=bias, return_debug=True, mask=mask)    

            else:
                att_out, att = self.MHA(q=q, k=k, v=v, return_debug=True, mask=mask) 

        else:
            pos_embedding = self.pos_embedder(pos[0], pos[1])
            bias = self.emb_proj_bias(pos_embedding)
            att_out, att = self.MHA(v=v, bias=bias, return_debug=True, mask=mask) 

        if self.q_res:
            x = q + self.dropout1(att_out)
        else:
            x = self.dropout1(att_out)

        x = self.norm1(x)

        x = self.norm2(x + self.dropout2(self.mlp_layer(x)))

        x = x.view(b,n,-1,e)

        return self.output_layer(x)

class n_nha_layers(nn.Module):
    def __init__(self, n_layers, input_dim, model_dim, ff_dim, n_heads=4, dropout=0, input_mlp=False, output_dim=None, activation=nn.SiLU(), pos_emb_type='bias', pos_embedder=None, pos_emb_dim=None, kv_dropout=0) -> None: 
        super().__init__()

        self.layer_list = nn.ModuleList()
        for k in range(n_layers):
            if k == 0:
                dim_in = input_dim
            else:
                dim_in = model_dim

            self.layer_list.append(nha_layer(dim_in, 
                                        model_dim, 
                                        ff_dim, 
                                        n_heads=n_heads, 
                                        dropout=dropout, 
                                        input_mlp=False,
                                        output_dim=output_dim, 
                                        activation=activation, 
                                        pos_emb_type=pos_emb_type, 
                                        pos_embedder=pos_embedder, 
                                        pos_emb_dim=pos_emb_dim,
                                        kv_dropout=kv_dropout))

    def forward(self, x: torch.tensor, xq=None, mask=None, pos=None):
        for layer in self.layer_list:
            x = layer(x, xq=xq, mask=mask, pos=pos)
        return x

    

class coarsen_layer(nn.Module):
    def __init__(self, n_reduce, model_dim, ff_dim, n_heads=4, dropout=0, activation=nn.SiLU(), pos_emb_type='bias', pos_embedder=None, pos_emb_dim=None, q_res=False) -> None: 
        super().__init__()

        self.n_reduce = n_reduce
        
        self.nha_layer = nha_layer(
                input_dim = model_dim,
                model_dim = model_dim,
                ff_dim = ff_dim,
                n_heads = n_heads,
                input_mlp = False,
                q_res = q_res,
                dropout=dropout,
                pos_emb_type=pos_emb_type,
                pos_embedder=pos_embedder,
                pos_emb_dim=pos_emb_dim,
                kv_dropout=0)
        
        
    def forward(self, x, pos=None):
       
        x = x.reshape(x.shape[0],-1,self.n_reduce,x.shape[-1])

        x = self.nha_layer(x, pos=pos)

        x = x.mean(dim=-2)

        return x



class input_layer(nn.Module):
    def __init__(self, grid_level_0, input_mapping, input_in_range, input_coordinates, input_dim, model_dim, ff_dim, seq_level=1, n_heads=4, dropout=0, pos_emb_type='bias', pos_embedder=None, pos_emb_dim=None, polar=True, force_nha=False, kv_dropout=0, input_mlp=True) -> None: 
        super().__init__()

        self.register_buffer("input_mapping", input_mapping, persistent=False)
        self.register_buffer("grid_out_of_range_mask", ~input_in_range, persistent=False)
        self.register_buffer("input_coordinates", input_coordinates, persistent=False)
        self.grid_level_0 = grid_level_0

        self.seq_level = seq_level
        self.pos_embedder = pos_embedder
        self.pos_calculation = "polar" if polar else "cartesian"
        
       # if input_mlp:
        self.input_mlp = nn.Sequential(
                        nn.Linear(input_dim, model_dim, bias=True),
                        nn.SiLU())
      #  else:
      #      self.input_mlp= nn.Identity()

        input_dim = model_dim if input_mlp else input_dim

        #if input_mapping.shape[-1]>1 or force_nha:
        self.nha_layer = nha_layer(
                    input_dim = model_dim,
                    model_dim = model_dim,
                    ff_dim = ff_dim,
                    n_heads = n_heads,
                    input_mlp = False,
                    dropout=dropout,
                    q_res=False,
                    pos_emb_type=pos_emb_type,
                    pos_embedder=pos_embedder,
                    pos_emb_dim=pos_emb_dim,
                    kv_dropout=kv_dropout,
                    qkv_bias=False)

    def get_relative_positions(self, grid_level_indices):
        
        
        indices = self.input_mapping[grid_level_indices]

        coords1 = self.grid_level_0.get_coordinates_from_grid_indices(grid_level_indices).unsqueeze(dim=-1)

        coords2 = self.input_coordinates[:, indices]

        pos1_grid, pos2_grid = get_distance_angle(coords1[0].unsqueeze(dim=-1), coords1[1].unsqueeze(dim=-1), coords2[0].unsqueeze(dim=-3), coords2[1].unsqueeze(dim=-3), base=self.pos_calculation)
        #pos1_grid, pos2_grid = get_distance_angle(coords1[0], coords1[1], coords2[0], coords2[1], base=self.pos_calculation)
        b,n,nh1,nh2,ng = pos1_grid.shape

        pos1_grid = pos1_grid.view(b,n,nh1,-1)
        pos2_grid = pos2_grid.view(b,n,nh1,-1)

        return pos1_grid.float(), pos2_grid.float()


    def forward(self, x, grid_level_indices, drop_mask=None):

        b,n,nh,f = x.shape

        x = self.input_mlp(x)

        if not isinstance(self.nha_layer, nn.Identity):
            
            grid_out_of_range_mask = self.grid_out_of_range_mask[grid_level_indices]

            grid_level_indices = sequenize(grid_level_indices, self.seq_level)

            pos = self.get_relative_positions(grid_level_indices)
            
            mask = torch.logical_or(grid_out_of_range_mask, drop_mask.view(b,n,1).repeat(1,1,nh))

            mask = sequenize(mask, self.seq_level)

            mask = mask.unsqueeze(dim=2).repeat(1,1,mask.shape[2],1,1)
            mask = mask.view(mask.shape[0], mask.shape[1], mask.shape[2], -1)         
        
            # tale nearest for q
           # xq = x[:,:,[0],:]
           # xq = sequenize(xq, self.seq_level)
           # xq = xq.reshape(b, xq.shape[1],-1, x.shape[-1])

            x = sequenize(x, self.seq_level)

            x = x.reshape(b, x.shape[1],-1, x.shape[-1])

            # xk = self.proj_k(self.pos_embedder(pos_source[0], pos_source[1]))
            # xk = xk.view(x.shape)

            #xq = self.proj_q(self.pos_embedder(pos_grid[0], pos_grid[1]))
            x = self.nha_layer(x, pos=pos, mask=mask)

        x = x.view(b, n, -1)
    
        return x

      

class position_embedder(nn.Module):
    def __init__(self, min_dist, max_dist, embed_dim, n_out, pos_emb_calc="polar") -> None: 
        super().__init__()
        self.pos_emb_calc = pos_emb_calc

        self.operation = None
        self.transform = None
        self.proj_layer = None
        self.cartesian = False

        if "descrete" in pos_emb_calc and "polar" in pos_emb_calc:
            self.pos1_emb = helpers.PositionEmbedder_phys_log(min_dist, max_dist, embed_dim, n_heads=n_out)
            self.pos2_emb = helpers.PositionEmbedder_phys(-torch.pi, torch.pi, embed_dim, n_heads=n_out)

        if "semi" in pos_emb_calc and "polar" in pos_emb_calc:
            self.pos1_emb = nn.Linear(1, n_out)
            self.pos2_emb = helpers.PositionEmbedder_phys(-torch.pi, torch.pi, embed_dim, n_heads=n_out)

        if "cartesian" in pos_emb_calc:
            self.proj_layer = nn.Sequential(nn.Linear(2, embed_dim, bias=True),
                                        nn.SiLU(),
                                        nn.Linear(embed_dim, n_out, bias=True))
            
            self.cartesian = True


        if "learned" in pos_emb_calc and "polar" in pos_emb_calc:
            self.proj_layer = nn.Sequential(nn.Linear(2*n_out, n_out, bias=True),
                                        nn.SiLU(),
                                        nn.Linear(n_out, n_out, bias=True))
        
        if 'inverse' in pos_emb_calc:
            self.transform = helpers.conv_coordinates_inv

        if 'log' in pos_emb_calc:
            self.transform = helpers.conv_coordinates_log
                    
        if 'sine' in pos_emb_calc:
            self.operation = 'sine'

        elif 'sum' in pos_emb_calc:
            self.operation = 'sum'

        elif 'product' in pos_emb_calc:
            self.operation = 'product'


    def forward(self, pos1, pos2):
        if self.cartesian:
            if self.transform is not None:
                pos1 = self.transform(pos1)
                pos2 = self.transform(pos2)

            return self.proj_layer(torch.stack((pos1, pos2), dim=-1))    

        else:
            if self.transform is not None:
                pos1 = self.transform(pos1)
            
            if isinstance(self.pos1_emb, nn.Linear):
                pos1 = pos1.unsqueeze(dim=-1)

            pos1 = self.pos1_emb(pos1)
            pos2 = self.pos2_emb(pos2)

            if self.proj_layer is not None:
                return self.proj_layer(torch.concat((pos1, pos2), dim=-1))
            
            if self.operation == 'sine':
                return pos1 * torch.sin(pos2)
            
            elif self.operation == 'sum':
                return pos1 + pos2
            
            elif self.operation == 'product':
                return pos1 * pos2


class grid_layer(nn.Module):
    def __init__(self, global_level, adjc, coordinates) -> None: 
        super().__init__()

        self.global_level = global_level
        self.register_buffer("coordinates", coordinates, persistent=False)
        self.register_buffer("adjc", adjc, persistent=False)

    def get_nh(self, x, local_indices, sample_dict):
        indices_nh, mask = get_nh_indices(self.adjc, local_cell_indices=local_indices, global_level=int(self.global_level))
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
        return x, coords



class processing_layers(nn.Module):
    def __init__(self, grid_layers: dict, model_hparams: dict, pos_embedder) -> None: 
        super().__init__()

        model_dim = model_hparams['model_dim']
        n_heads = model_hparams['n_heads']
        dropout = model_hparams['dropout']
        self.max_seq_level = model_hparams['max_seq_level']
        n_layers = model_hparams['n_processing_layers']
        kv_dropout = model_hparams['kv_dropout']
        pos_emb_type = model_hparams['pos_emb_type']

        self.updated_lf_att = model_hparams['updated_lf_att']

        pos_embedder_handle = pos_embedder['pos_embedder_handle']
        pos_emb_dim = pos_embedder['pos_emb_dim']
        self.polar = pos_embedder['polar']

        self.grid_layers = grid_layers
        
        self.max_level = list(grid_layers.keys())[-1]

        self.layers = nn.ModuleDict()
        self.global_levels = []

        for global_level in grid_layers.keys(): 
            
            self.global_levels.append(global_level)
            if global_level != self.max_level:
                cross_layers = nn.ModuleList([nha_layer(input_dim= model_dim,
                            model_dim = model_dim,
                            ff_dim = model_dim,
                            n_heads =  n_heads,
                            input_mlp = False,
                            dropout=dropout,
                            pos_emb_type=pos_emb_type,
                            pos_embedder=pos_embedder_handle,
                            pos_emb_dim=pos_emb_dim,
                            activation=nn.SiLU(),
                            kv_dropout=kv_dropout) for _ in range(n_layers)])
            else:
                cross_layers = nn.ModuleList([None for _ in range(n_layers)])
                
            seq_layers = nn.ModuleList([nha_layer(input_dim= model_dim,
                        model_dim = model_dim,
                        ff_dim = model_dim,
                        n_heads =  n_heads,
                        input_mlp = False,
                        dropout=dropout,
                        pos_emb_type=pos_emb_type,
                        pos_embedder=pos_embedder_handle,
                        pos_emb_dim=pos_emb_dim,
                        activation=nn.SiLU(),
                        kv_dropout=kv_dropout) for _ in range(n_layers)])
            
            nh_layers = nn.ModuleList([nha_layer(input_dim= model_dim,
                        model_dim = model_dim,
                        ff_dim = model_dim,
                        n_heads =  n_heads,
                        input_mlp = False,
                        dropout=dropout,
                        pos_emb_type=pos_emb_type,
                        pos_embedder=pos_embedder_handle,
                        pos_emb_dim=pos_emb_dim,
                        activation=nn.SiLU(),
                        kv_dropout=0) for _ in range(n_layers)])
                
            self.layers[str(global_level)] = nn.ModuleDict(zip(['cross_layers', 'seq_layers', 'nh_layers'],
                                                            [cross_layers, seq_layers, nh_layers]))

            
    def forward(self, x_levels, indices_layers, sample_dict):
        
        if self.updated_lf_att:
            x_processed = x_levels
        else:
            x_processed = dict(zip(x_levels.keys(), [x.clone() for x in x_levels.values()]))

        for global_level in self.global_levels[::-1]:
            layers = self.layers[str(global_level)]

            x = x_levels[global_level]

            b, n, f = x.shape               
            
            for layer_idx in range(len(layers['seq_layers'])):
                
                if layers['cross_layers'][layer_idx] is not None:
                    global_level_lf = global_level + 1

                    x_lf = x_levels[global_level_lf]

                    x_lf_nh, mask_lf, coords_lf_nh = self.grid_layers[global_level_lf].get_nh(x_lf, indices_layers[global_level_lf], sample_dict)

                    x, coords_sections = self.grid_layers[global_level].get_sections(x, indices_layers[global_level], section_level=1)

                    relative_positions = get_relative_positions(coords_sections, coords_lf_nh, polar=self.polar)

                    x = layers['cross_layers'][layer_idx](x_lf_nh, xq=x, mask=mask_lf.unsqueeze(dim=-2), pos=relative_positions)

                    x = x.view(b, n, -1)


                x, coords_sections = self.grid_layers[global_level].get_sections(x, indices_layers[global_level], section_level=self.max_seq_level)

                relative_positions = get_relative_positions(coords_sections, coords_sections, polar=self.polar)
                x = layers['seq_layers'][layer_idx](x, pos=relative_positions)
                x = x.view(b, n, -1)

                x_nh, mask, coords_nh = self.grid_layers[global_level].get_nh(x, indices_layers[global_level], sample_dict)
                coords = self.grid_layers[global_level].get_coordinates_from_grid_indices(indices_layers[global_level]).unsqueeze(dim=-1)

                relative_positions = get_relative_positions(coords, coords_nh, polar=self.polar)

                x = layers['seq_layers'][layer_idx](x_nh, xq=x, mask=mask.unsqueeze(dim=-2), pos=relative_positions)

                x = x.view(b, n, -1)

                x_processed[global_level] = x

        return x_levels



class decomp_layer(nn.Module):
    def __init__(self, grid_layers: dict, model_hparams: dict, pos_embedder, residual_decomp=True) -> None: 
        super().__init__()

        model_dim = model_hparams['model_dim']
        n_heads = model_hparams['n_heads']
        dropout = model_hparams['dropout']
        pos_emb_type = model_hparams['pos_emb_type']

        pos_embedder_handle = pos_embedder['pos_embedder_handle']
        pos_emb_dim = pos_embedder['pos_emb_dim']
        self.polar = pos_embedder['polar']

        self.residual_decomp = residual_decomp

        self.grid_layers = grid_layers
        
        self.max_level = list(grid_layers.keys())[-1]

        self.layers = nn.ModuleDict()
        self.global_levels = []
        for global_level in grid_layers.keys(): 
            if global_level != self.max_level:

                self.layers[str(global_level)] = nha_layer(
                                input_dim = model_dim,
                                model_dim = model_dim,
                                ff_dim = model_dim,
                                n_heads = n_heads,
                                input_mlp = False,
                                q_res = False,
                                dropout=dropout,
                                pos_emb_type=pos_emb_type,
                                pos_embedder=pos_embedder_handle,
                                pos_emb_dim=pos_emb_dim,
                                kv_dropout=0,
                                v_proj=False)
                
                self.global_levels.append(global_level)
            
    def forward(self, x, indices_layers):
        
        x_levels={}
        for global_level in self.global_levels:
            
            b,n,e = x.shape
            layer = self.layers[str(global_level)]

            x_sections, coords_sections = self.grid_layers[global_level].get_sections(x, indices_layers[global_level], section_level=1)

            relative_positions = get_relative_positions(coords_sections, coords_sections, polar=self.polar)
            x_sections_f = layer(x_sections, pos=relative_positions).mean(dim=-2, keepdim=True)

            if self.residual_decomp:
                x_levels[global_level] = (x_sections - x_sections_f).view(b,n,-1)
            else:
                x_levels[global_level] = x
            
            x = x_sections_f.squeeze(dim=-2)

        x_levels[global_level + 1] = x

        return x_levels



class projection_layer(nn.Module):
    def __init__(self, grid_layers: dict, model_hparams: dict, pos_embedder, output_dim, output_mappings=None, input_mappings=None) -> None: 
        super().__init__()

        self.output_mappings = output_mappings
        self.input_mappings = input_mappings

        model_dim = model_hparams['model_dim']
        n_heads = model_hparams['n_heads']
        dropout = model_hparams['dropout']

        pos_embedder_handle = pos_embedder['pos_embedder_handle']
        pos_emb_dim = pos_embedder['pos_emb_dim']
        self.polar = pos_embedder['polar']

        self.grid_layers = grid_layers
        
        self.max_level = list(grid_layers.keys())[-1]
        
        self.layers = nn.ModuleDict()
        self.global_levels = []
        for global_level in grid_layers.keys(): 

            self.layers[str(global_level)] = nha_layer(
                            input_dim = model_dim,
                            model_dim = model_dim,
                            output_dim = output_dim,
                            ff_dim = model_dim,
                            n_heads = n_heads,
                            input_mlp = False,
                            q_res = False, 
                            dropout=dropout,
                            pos_emb_type='bias',
                            qkv_bias=False,
                            pos_embedder=pos_embedder_handle,
                            pos_emb_dim=pos_emb_dim,
                            kv_dropout=0)
                
            self.global_levels.append(global_level)
            
    def forward(self, x_levels, indices_layers, sample_dict):
        
        if self.output_mappings is None:
            output_coords = self.grid_layers[0].get_coordinates_from_grid_indices(indices_layers[0])

        x_output = None
        for global_level in self.global_levels:

            x_nh, mask, coords_nh = self.grid_layers[global_level].get_nh(x_levels[global_level], indices_layers[global_level], sample_dict)

            output_coords = output_coords.reshape(output_coords.shape[0],output_coords.shape[1], -1, 4**global_level)

            relative_positions = get_relative_positions(output_coords, coords_nh, polar=self.polar)

            x = self.layers[str(global_level)](x_nh, pos=relative_positions, mask=mask.unsqueeze(dim=-2))

            x = x.view(x.shape[0], -1, x.shape[-1])
            if x_output is None:
                x_output = x
            else:
                x_output += x
 
        return x_output

def get_relative_positions(coords1, coords2, polar=False):
    
    if coords2.dim() > coords1.dim():
        coords1 = coords1.unsqueeze(dim=-1)

    if coords1.dim() > coords2.dim():
        coords2 = coords2.unsqueeze(dim=-2)

    if coords1.dim() == coords2.dim():
        coords1 = coords1.unsqueeze(dim=-1)
        coords2 = coords2.unsqueeze(dim=-2)

    distances, phis = get_distance_angle(coords1[0], coords1[1], coords2[0], coords2[1], base="polar" if polar else "cartesian")

    return distances.float(), phis.float()


class ICON_Transformer(nn.Module):
    def __init__(self, model_settings) -> None: 
        super().__init__()

        self.model_settings = load_settings(model_settings, id='model')

        self.check_model_dir()

        self.pos_emb_calc = self.model_settings["pos_emb_calc"]
        self.polar = True if "polar" in  self.pos_emb_calc else False

        self.grid = xr.open_dataset(self.model_settings['processing_grid'])
        self.register_buffer('global_indices', torch.arange(len(self.grid.clon)).unsqueeze(dim=0), persistent=False)

        eoc = torch.tensor(self.grid.edge_of_cell.values - 1)
        self.register_buffer('eoc', eoc, persistent=False)

        acoe = torch.tensor(self.grid.adjacent_cell_of_edge.values - 1)
        self.register_buffer('acoe', acoe, persistent=False)

        cell_coords_global = get_coords_as_tensor(self.grid, lon='clon', lat='clat').double()
        self.register_buffer('cell_coords_global', cell_coords_global, persistent=False)  

        self.input_data  = self.model_settings['variables_source']
        self.output_data = self.model_settings['variables_target']
        self.model_dim = self.model_settings['model_dim']

        n_grid_levels = self.model_settings['n_grid_levels']
        grid_levels_steps = 1

        pos_embedder_handle = self.init_position_embedder(self.model_settings["pos_emb_dim"], min_coarsen_level=0, max_coarsen_level=n_grid_levels, embed_dim=self.model_settings["emb_table_bins"])

        pos_embedder = {}

        pos_embedder['pos_emb_dim'] = self.model_settings["pos_emb_dim"]
        pos_embedder['polar'] = self.polar
        pos_embedder['pos_embedder_handle'] = pos_embedder_handle
    
        grid_layers = {}
        self.global_levels = []
        for grid_level_idx in range(n_grid_levels):

            global_level = grid_level_idx*grid_levels_steps
            self.global_levels.append(global_level)

            adjc = self.get_adjacent_global_cell_indices(global_level) 
            coordinates = self.get_coordinates_level(global_level) 

            grid_layers[global_level] = grid_layer(global_level, adjc, coordinates)


        input_mapping, input_in_range, input_coordinates = self.get_input_grid_mapping()
        #output_mapping, _, output_coordinates = self.get_output_grid_mapping()

        #tbd: use input projection layer as input layer with input_mapping and output_layer with output_mapping
        self.input_layers = self.init_input_layers(grid_layers[0], self.model_dim, input_mapping, input_in_range, input_coordinates, pos_embedder)

        self.decomp_layer = decomp_layer(grid_layers, self.model_settings, pos_embedder, residual_decomp=self.model_settings['residual_decomp'])

        self.processing_projection_layers = nn.ModuleList()
        for k in range(self.model_settings['n_proccesing_decomp_layers']):
            proc_layer = processing_layers(grid_layers, self.model_settings, pos_embedder)
            
            if k < self.model_settings['n_proccesing_decomp_layers']-1:
                output_dim = None
            else:
                output_dim = len(self.model_settings['variables_target']['cell'])

            if k == 0 or k == self.model_settings['n_proccesing_decomp_layers']-1:
                proj_layer = projection_layer(grid_layers, self.model_settings, pos_embedder, output_dim=output_dim)

            self.processing_projection_layers.append(nn.ModuleDict(zip(['processing', 'projection'], [proc_layer, proj_layer])))

        
        out_dim_input = self.model_dim
        self.input_projection = nn.Sequential(nn.Linear(self.model_dim * len(self.input_data), out_dim_input, bias=False))
  

        strict = self.model_settings['load_strict'] if 'load_strict' in self.model_settings.keys() else True

        trained_iterations = None
        if "pretrained_path" in self.model_settings.keys():
            trained_iterations = self.check_pretrained(log_dir_check=self.model_settings['pretrained_path'], strict=strict)

        if "pretrained_pos_embeddings_path" in self.model_settings.keys():
            self.check_pretrained(log_dir_check=self.model_settings['pretrained_pos_embeddings_path'], strict=False, match_list='pos_embedder')

        if "pretrained_model_wo_input" in self.model_settings.keys():
            self.check_pretrained(log_dir_check=self.model_settings['pretrained_model_wo_input'], strict=False, match_list='pos_embedder', not_match='input')

        self.trained_iterations = trained_iterations


    def forward(self, x, indices_batch_dict=None, debug=False):
        # if global_indices are provided, batches in x are treated as independent
        debug_list = []

        if indices_batch_dict is None:
            indices_batch_dict = {'global_cell': self.global_indices,
                                  'local_cell': self.global_indices,
                       'sample': None,
                       'sample_level': None}
        else:
            indices_layers = dict(zip(self.global_levels,[self.get_global_indices_local(indices_batch_dict['sample'], indices_batch_dict['sample_level'], global_level) for global_level in self.global_levels]))
        
        input_data = []
        for key, values in x.items():
            
            input = self.input_layers[key](values, indices_batch_dict["local_cell"], drop_mask=indices_batch_dict['drop_mask'])
            input_data.append(input) 
        
        x = torch.concat(input_data, dim=-1)

        x = self.input_projection(x)

        for layers in self.processing_projection_layers:

            x_levels = self.decomp_layer(x, indices_layers)
            x_levels = layers['processing'](x_levels, indices_layers, indices_batch_dict)
            x = layers['projection'](x_levels, indices_layers, indices_batch_dict)
        
        if debug:
            return {'cell': x.unsqueeze(dim=-2)}, debug_list
        else:
            return {'cell': x.unsqueeze(dim=-2)}

        
    
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

        output = grid_dict_to_var_dict(outputs_all, self.model_settings["variables_target"])
        
        return output


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


    def get_input_grid_mapping(self):
        
        input_coordinates = {}
        for grid_type in self.input_data.keys():
            input_coordinates[grid_type] = get_coords_as_tensor(xr.open_dataset(self.model_settings['input_grid']),grid_type=grid_type)
           #global_indices = self.coarsen_indices(self.global_level_start)[0]
           # input_coordinates[grid_type] = input_coordinates[grid_type][:,global_indices[0,:,0]]
       

        input_mapping, in_range = get_nh_variable_mapping_icon(self.model_settings['processing_grid'], ['cell'], 
                                    self.model_settings['input_grid'], self.input_data, 
                                    search_raadius=self.model_settings['search_raadius'], 
                                    max_nh=self.model_settings['nh_input'], 
                                    level_start=self.model_settings['level_start_input'], 
                                    lowest_level=0)

        
        return input_mapping, in_range, input_coordinates

    def get_output_grid_mapping(self):
        
        output_mapping, in_range = get_nh_variable_mapping_icon(self.model_settings['processing_grid'], ['cell'], 
                                    self.model_settings['processing_grid'], self.output_data, 
                                    search_raadius=self.model_settings['search_raadius'], max_nh=self.model_settings['nh_input'], level_start=self.model_settings['level_start_input'])

        output_coordinates = {}
        for grid_type in self.output_data.keys():
            output_coordinates[grid_type] = get_coords_as_tensor(xr.open_dataset(self.model_settings['processing_grid']),grid_type=grid_type)

        return output_mapping, in_range, output_coordinates


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
    
    def get_adjacent_global_cell_indices(self, global_level):
        adjc_indices = self.acoe.T[self.eoc.T].reshape(-1,4**global_level,6)
        self_indices = self.global_indices.view(-1,4**global_level)[:,0]

        adjc_indices = adjc_indices.view(self_indices.shape[0],-1) // 4**global_level
        self_indices = self_indices // 4**global_level

        adjc_unique = (adjc_indices).long().unique(dim=-1)
        
        is_self = adjc_unique - self_indices.view(-1,1) == 0

        adjc = adjc_unique[~is_self]

        adjc = adjc.reshape(self_indices.shape[0], -1)

        return adjc
    
    def get_coordinates_level(self, global_level):
        indices = self.global_indices.reshape(-1,4**int(global_level))[:,0]
        coords = self.cell_coords_global[:,indices]
        return coords

    def get_relative_positions(self, cell_indices1, cell_indices2):
      
        coords1 = self.cell_coords_global[:,cell_indices1]
        coords2 = self.cell_coords_global[:,cell_indices2]
  
        if coords2.dim() > coords1.dim():
            coords1 = coords1.unsqueeze(dim=-1)

        if coords1.dim() > coords2.dim():
            coords2 = coords2.unsqueeze(dim=-2)

        if coords1.dim() == coords2.dim():
            coords1 = coords1.unsqueeze(dim=-1)
            coords2 = coords2.unsqueeze(dim=-2)

        distances, phis = get_distance_angle(coords1[0], coords1[1], coords2[0], coords2[1], base="polar" if self.polar else "cartesian")

        return distances.float(), phis.float()


    def init_position_embedder(self, n_out, min_coarsen_level=0, max_coarsen_level=0, embed_dim=64):
        # tbd: sample points for redcution of memory
        
    
        _, indices_global , indices_global_nh ,_ = self.coarsen_indices(min_coarsen_level)
        pos = self.get_relative_positions(indices_global, 
                                    indices_global_nh)
        
        # quantile very sensitive -> quantile embedding table? or ln fcn? or linear?
        min_dist = pos[0].quantile(0.01)

        _, indices_global , indices_global_nh ,_ = self.coarsen_indices(max_coarsen_level)
        pos = self.get_relative_positions(indices_global, 
                                    indices_global_nh)
        
        max_dist = pos[0].quantile(0.99)
          
        return  position_embedder(min_dist, max_dist, embed_dim=embed_dim, n_out=n_out, pos_emb_calc=self.pos_emb_calc)
    

    def init_input_layers(self, grid_layer_0, model_dim_var, input_mapping, input_in_range, input_coordinates, pos_embedder):

        n_heads = self.model_settings['n_heads']

        input_layers = nn.ModuleDict()
        for key in input_mapping["cell"].keys():
            
            n_input = len(self.input_data[key])

            layer = input_layer(
                    grid_layer_0,
                    input_mapping["cell"][key],
                    input_in_range["cell"][key],
                    input_coordinates[key],
                    seq_level=self.model_settings['input_seq_level'],
                    input_dim = n_input,
                    model_dim = model_dim_var,
                    dropout=self.model_settings['dropout'],
                    ff_dim = model_dim_var,
                    n_heads = n_heads,
                    pos_embedder=pos_embedder['pos_embedder_handle'],
                    pos_emb_type = "bias",
                    pos_emb_dim=pos_embedder['pos_emb_dim'],
                    polar=pos_embedder['polar'],
                    kv_dropout=self.model_settings['kv_dropout_input'])
            
            input_layers[key] = layer

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