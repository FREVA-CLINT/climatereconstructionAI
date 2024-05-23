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
    def __init__(self, input_dim, model_dim, ff_dim, n_heads=4, dropout=0, input_mlp=False, output_dim=None, activation=nn.SiLU(), q_res=True, pos_emb_type='bias', pos_embedder=None, pos_emb_dim=None, kv_dropout=0, qkv_bias=True) -> None: 
        super().__init__()

        self.model_dim = model_dim
        self.n_heads = n_heads
        self.pos_emb_type = pos_emb_type
        self.kv_dropout = kv_dropout
        self.qkv_bias= qkv_bias
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
    #        self.emb_proj_q = nn.Linear(pos_emb_dim, model_dim // n_heads, bias=False)
            self.emb_proj_k = nn.Linear(pos_emb_dim, model_dim // n_heads, bias=False)
            self.emb_proj_v = nn.Linear(pos_emb_dim, model_dim // n_heads, bias=False)

        self.MHA = helpers.MultiHeadAttentionBlock(
            model_dim, model_dim, n_heads, input_dim=input_dim, qkv_proj=True, qkv_bias=qkv_bias
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
            pos1 = pos[0].view(b*n,nh,-1)
            pos2 = pos[1].view(b*n,nh,-1)

            indices_keep = (torch.randperm((nh)*(x.shape[0]), device=x.device) % (nh-1)).view(x.shape[0], nh)[:, :int(1- self.kv_dropout*nh)]
       
            k = torch.gather(k, dim=1, index=indices_keep.unsqueeze(dim=-1).repeat(1,1,k.shape[-1]))
            v = torch.gather(v, dim=1, index=indices_keep.unsqueeze(dim=-1).repeat(1,1,k.shape[-1]))
            pos1 = torch.gather(pos1, dim=-1, index=indices_keep.view(b*n,1,-1).repeat(1, pos1.shape[1],1))
            pos2 = torch.gather(pos2, dim=-1, index=indices_keep.view(b*n,1,-1).repeat(1, pos2.shape[1],1))
            pos = (pos1.view(b,n,nh,-1), pos2.view(b,n,nh,-1))

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
    def __init__(self, n_layers, input_dim, model_dim, ff_dim, n_heads=4, dropout=0, input_mlp=False, output_dim=None, activation=nn.SiLU(), pos_emb_type='bias', pos_embedder=None, pos_emb_dim=None) -> None: 
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
                                        pos_emb_dim=pos_emb_dim))

    def forward(self, x: torch.tensor, xq=None, mask=None, pos=None):
        for layer in self.layer_list:
            x = layer(x, xq=xq, mask=mask, pos=pos)
        return x

    
class nha_reduction_layer(nn.Module):
    def __init__(self, input_dim, model_dim, ff_dim, nh_reduction, input_mlp=True, n_heads=4, dropout=0, activation=nn.SiLU(), pos_emb_type='bias', pos_embedder=None, pos_emb_dim=None) -> None: 
        super().__init__()

        self.nha_layer = nha_layer(
                input_dim = input_dim,
                model_dim = model_dim,
                ff_dim = ff_dim,
                n_heads = n_heads,
                input_mlp = input_mlp,
                dropout=dropout,
                pos_emb_type=pos_emb_type,
                pos_embedder=pos_embedder,
                pos_emb_dim=pos_emb_dim)
        
        
        self.reduction_mlp = reduction_mlp(model_dim, nh_reduction=nh_reduction)
        
    def forward(self, x, pos=None):

        x = self.nha_layer(x, pos=pos)
        
        x = self.reduction_mlp(x)

        return x


class coarsen_layer(nn.Module):
    def __init__(self, input_dim, model_dim, ff_dim, n_heads=4, dropout=0, activation=nn.SiLU(), pos_emb_type='bias', pos_embedder=None, pos_emb_dim=None) -> None: 
        super().__init__()

        self.nha_reduction_layer = nha_reduction_layer(
                input_dim = input_dim,
                model_dim = model_dim,
                ff_dim = ff_dim,
                n_heads = n_heads,
                nh_reduction = 4,
                input_mlp = True,
                dropout=dropout,
                pos_emb_type=pos_emb_type,
                pos_embedder=pos_embedder,
                pos_emb_dim=pos_emb_dim)
        
    def forward(self, x, pos=None):
        
       
        x = x.reshape(x.shape[0],-1,4,x.shape[-1])
        x = self.nha_reduction_layer(x, pos=pos)

        return x
    

class processing_layers(nn.Module):
    def __init__(self, global_level, adjc, coordinates, n_layers, input_dim, model_dim, ff_dim, n_heads=4, dropout=0, activation=nn.SiLU(), pos_emb_type='bias', pos_embedder=None, pos_emb_dim=None, seq_level=1, nh_att=True, seq_att=True, polar=True) -> None: 
        super().__init__()
        self.nh_att = nh_att
        self.seq_att = seq_att
        self.global_level = global_level
        self.pos_calculation = "polar" if polar else "cartesian"

        self.register_buffer("coordinates", coordinates)
        self.register_buffer("adjc", adjc)

        self.input_mlp = nn.Sequential(
                nn.Linear(input_dim, model_dim, bias=False)
            )

        if nh_att:
            self.nh_layers = nn.ModuleList([nha_layer(input_dim= model_dim,
                        model_dim = model_dim,
                        ff_dim = ff_dim,
                        n_heads =  n_heads,
                        input_mlp = False,
                        dropout=dropout,
                        pos_emb_type=pos_emb_type,
                        pos_embedder=pos_embedder,
                        pos_emb_dim=pos_emb_dim,
                        activation=activation) for k in range(n_layers)])

        if seq_att:
            self.seq_layers = nn.ModuleList([nha_layer(input_dim= model_dim,
                        model_dim = model_dim,
                        ff_dim = ff_dim,
                        n_heads =  n_heads,
                        input_mlp = False,
                        dropout=dropout,
                        pos_emb_type=pos_emb_type,
                        pos_embedder=pos_embedder,
                        pos_emb_dim=pos_emb_dim,
                        activation=activation) for k in range(n_layers)])
            
        self.seq_level = seq_level

    def get_relative_positions(self, indices1, indices2):
        coords1 = self.coordinates[:,indices1]
        coords2 = self.coordinates[:,indices2]
  
        if coords2.dim() > coords1.dim():
            coords1 = coords1.unsqueeze(dim=-1)

        if coords1.dim() > coords2.dim():
            coords2 = coords2.unsqueeze(dim=-2)

        if coords1.dim() == coords2.dim():
            coords1 = coords1.unsqueeze(dim=-1)
            coords2 = coords2.unsqueeze(dim=-2)

        pos1, pos2 = get_distance_angle(coords1[0], coords1[1], coords2[0], coords2[1], base=self.pos_calculation)

        return pos1.float(), pos2.float()

        
    def forward(self, x: torch.tensor, indices_global_level, sample_dict):
        
        x = self.input_mlp(x)

        local_indices = indices_global_level // 4**int(self.global_level)
        if self.nh_att:
            indices_nh, mask = get_nh_indices(self.adjc, local_cell_indices=local_indices, global_level=int(self.global_level))

            pos_nh = self.get_relative_positions(local_indices, 
                                                indices_nh.squeeze())
        
        indices_sequence = sequenize(local_indices, self.seq_level)
        pos_seq = self.get_relative_positions(indices_sequence, 
                                            indices_sequence)
        
        b, n, e = x.shape
        
        for k in range(len(self.nh_layers)):

            if self.nh_att:
                x_nh = gather_nh_data(x, indices_nh, sample_dict['sample'], sample_dict['sample_level'], int(self.global_level))
                x = self.nh_layers[k](x_nh, xq=x, mask=mask.unsqueeze(dim=-2), pos=pos_nh)
                x = x.view(b, n, -1)
            
            if self.seq_att:
                x = sequenize(x, self.seq_level)
                x = self.seq_layers[k](x, pos=pos_seq)
                x = x.view(b, n, -1)

        return x   


class input_layer(nn.Module):
    def __init__(self, input_mapping, input_coordinates, input_dim, model_dim, ff_dim, seq_level=1, n_heads=4, dropout=0, pos_emb_type='bias', pos_embedder=None, pos_emb_dim=None, polar=True, force_nha=False, kv_dropout=0, input_mlp=True) -> None: 
        super().__init__()

        self.register_buffer("input_mapping", input_mapping)
        self.register_buffer("input_coordinates", input_coordinates)
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

        if input_mapping.shape[-1]>1 or force_nha:
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
        
        else:
            self.nha_layer = nn.Identity()
        '''
        if pos_emb_dim != input_dim:
            self.proj_q = nn.Linear(pos_emb_dim, model_dim, bias=False)
            self.proj_k = nn.Linear(pos_emb_dim, model_dim, bias=False)
        else:
            self.proj_q = nn.Identity()
            self.proj_k = nn.Identity()

        self.pos_embedder = pos_embedder
        '''
        

    def get_relative_positions(self, grid_level_indices, grid_level_coords):
        
        
        indices = self.input_mapping[grid_level_indices]

        coords1 = grid_level_coords[:, grid_level_indices].unsqueeze(dim=-1)

        coords2 = self.input_coordinates[:, indices]

        # use mid point of sequence for "absolute-relative" coordinates
        #coords_ref = coords1[:,:,:,[0]]
        coords2 = coords2.view(coords2.shape[0], coords2.shape[1],coords2.shape[2],1, -1)

        pos1_grid, pos2_grid = get_distance_angle(coords1[0], coords1[1], coords2[0], coords2[1], base=self.pos_calculation)

        return pos1_grid.float(), pos2_grid.float()


    def forward(self, x, grid_level_indices, grid_level_coords):
        b,n,nh,f = x.shape

        x = self.input_mlp(x)

        if not isinstance(self.nha_layer, nn.Identity):

            grid_level_indices = sequenize(grid_level_indices, self.seq_level)

            pos = self.get_relative_positions(grid_level_indices,
                                                grid_level_coords)
            
        

            # tale nearest for q
           # xq = x[:,:,[0],:]
           # xq = sequenize(xq, self.seq_level)
           # xq = xq.reshape(b, xq.shape[1],-1, x.shape[-1])

            x = sequenize(x, self.seq_level)
            x = x.reshape(b, x.shape[1],-1, x.shape[-1])

            # xk = self.proj_k(self.pos_embedder(pos_source[0], pos_source[1]))
            # xk = xk.view(x.shape)

            #xq = self.proj_q(self.pos_embedder(pos_grid[0], pos_grid[1]))
            x = self.nha_layer(x, pos=pos)

        x = x.view(b, n, -1)
    
        return x


class output_layer(nn.Module):
    # change to dynamic output sequences
    def __init__(self, global_level, adjc, coordinates, output_mapping_0, output_coordinates, input_dim, model_dim, ff_dim, n_heads=4, dropout=0, output_dim=None, pos_emb_type='bias', pos_embedder=None, pos_emb_dim=None, seq_level=1, polar=True, force_nha=False) -> None: 
        super().__init__()    

        self.n_output = output_mapping_0.shape[-1]
        output_mapping = output_mapping_0.reshape(-1, 4**int(global_level), output_mapping_0.shape[-1])
        output_mapping = output_mapping.reshape(output_mapping.shape[0], -1)

        self.register_buffer("output_mapping", output_mapping)

        self.seq_level = seq_level
        self.global_level = global_level
        self.output_dim = model_dim

        if output_mapping.shape[-1]>1 or force_nha:
            self.layer = refinement_layer(
                    global_level,
                    adjc,
                    coordinates_refined=output_coordinates,
                    coordinates=coordinates,
                    input_dim=input_dim,
                    model_dim = model_dim,
                    output_dim = model_dim,
                    pos_emb_type = pos_emb_type,
                    ff_dim = ff_dim,
                    n_heads = n_heads,
                    pos_embedder= pos_embedder,
                    pos_emb_dim=pos_emb_dim,
                    seq_level=seq_level,
                    polar=polar)
        else:
            self.layer = nn.Identity()

        self.output_mlp = nn.Sequential(nn.Linear(model_dim, output_dim, bias=False))

    def forward(self, x, local_indices, sample_dict):
        
        if not isinstance(self.layer, nn.Identity):
            local_indices = local_indices // 4**int(self.global_level)

            local_indices_refined = self.output_mapping[local_indices].view(local_indices.shape[0],-1)
            
            x = self.layer(x, local_indices, local_indices_refined, sample_dict, reshape=False)
            
        x = x.view(x.shape[0], -1, self.n_output, self.output_dim)

        return self.output_mlp(x)

class skip_layer(nn.Module):
    def __init__(self, input_dim, model_dim, ff_dim, n_heads=4, dropout=0, pos_emb_type='bias', pos_embedder=None, pos_emb_dim=None, seq_level=1) -> None: 
        super().__init__()
        
        self.seq_level = seq_level

        self.nha_layer = nha_layer(
                    input_dim = input_dim,
                    model_dim = model_dim,
                    ff_dim = ff_dim,
                    n_heads = n_heads,
                    input_mlp = False,
                    dropout=dropout,
                    pos_emb_type=pos_emb_type,
                    pos_embedder=pos_embedder,
                    pos_emb_dim=pos_emb_dim)
        

    def forward(self, x, x_skip, reshape=True, mask=None, pos=None):

        x = sequenize(x, self.seq_level)
        x_skip = sequenize(x_skip, self.seq_level)

        x = self.nha_layer(x_skip, xq=x, mask=mask, pos=pos)

        if reshape:
            x = x.view(x.shape[0],-1, x.shape[-1])


        return x

class refinement_layer(nn.Module):
    def __init__(self, global_level, adjc, coordinates_refined, coordinates, input_dim, model_dim, ff_dim, n_heads=4, dropout=0, output_dim=None, pos_emb_type='bias', pos_embedder=None, pos_emb_dim=None, seq_level=1, polar=True) -> None: 
        super().__init__()

        self.global_level = global_level
        self.register_buffer("coordinates_refined", coordinates_refined)
        self.register_buffer("coordinates", coordinates)
        self.register_buffer("adjc", adjc)

        self.pos_calculation = "polar" if polar else "cartesian"

        self.seq_level = seq_level

        self.nha_layer = nha_layer(
                    input_dim = input_dim,
                    model_dim = model_dim,
                    ff_dim = ff_dim,
                    n_heads = n_heads,
                    output_dim=output_dim,
                    input_mlp = True,
                    q_res=False,
                    dropout=dropout,
                    pos_emb_type=pos_emb_type,
                    pos_embedder=pos_embedder,
                    pos_emb_dim=pos_emb_dim,
                    qkv_bias=False)
        


    def get_relative_positions(self, indices_refined, indices_ref):
        coords1 = self.coordinates_refined[:,indices_refined]
        coords2 = self.coordinates[:,indices_ref]
  
        if coords2.dim() > coords1.dim():
            coords1 = coords1.unsqueeze(dim=-1)

        if coords1.dim() > coords2.dim():
            coords2 = coords2.unsqueeze(dim=-2)

        if coords1.dim() == coords2.dim():
            coords1 = coords1.unsqueeze(dim=-1)
            coords2 = coords2.unsqueeze(dim=-2)

        pos1, pos2 = get_distance_angle(coords1[0], coords1[1], coords2[0], coords2[1], base=self.pos_calculation)

        return pos1.float(), pos2.float()


    def forward(self, x, local_indices, local_indices_refined, sample_dict, reshape=True):
        
        b, n, f = x.shape
        n_refine = local_indices_refined.shape[1]// local_indices.shape[1]

        indices_nh, mask = get_nh_indices(self.adjc, local_cell_indices=local_indices, global_level=int(self.global_level))

        indices_nh = torch.concat((local_indices.view(b, n, 1), indices_nh), dim=-1)

        pos_nh = self.get_relative_positions(local_indices_refined.view(local_indices_refined.shape[0], -1, n_refine), 
                                            indices_nh.squeeze())
        

        #x_refined = x.unsqueeze(dim=-2).repeat_interleave(n_refine, dim=-2)

        x_nh = gather_nh_data(x, indices_nh, sample_dict['sample'], sample_dict['sample_level'], int(self.global_level))

        x = self.nha_layer(x_nh, pos=pos_nh)

        if reshape:
            x = x.view(x.shape[0],-1, x.shape[-1])

        
        return x
    


class pos_refinement_layer(nn.Module):
    def __init__(self, global_level, adjc, coordinates_refined, coordinates, input_dim, model_dim, ff_dim, n_heads=4, dropout=0, output_dim=None, pos_embedder=None, pos_emb_dim=None, seq_level=1, polar=True) -> None: 
        super().__init__()

        self.global_level = global_level
        self.register_buffer("coordinates_refined", coordinates_refined)
        self.register_buffer("coordinates", coordinates)
        self.register_buffer("adjc", adjc)

        self.pos_calculation = "polar" if polar else "cartesian"
        self.seq_level = seq_level
        self.pos_embedder = pos_embedder

        if pos_emb_dim != input_dim:
            self.proj_q = nn.Linear(pos_emb_dim, input_dim, bias=False)
            self.proj_k = nn.Linear(pos_emb_dim, input_dim, bias=False)
        else:
            self.proj_q = nn.Identity()
            self.proj_k = nn.Identity()

        self.nha_layer = nha_layer(
                    input_dim = input_dim,
                    model_dim = model_dim,
                    ff_dim = ff_dim,
                    n_heads = n_heads,
                    output_dim = output_dim,
                    input_mlp = False,
                    q_res=False,
                    dropout=dropout,
                    pos_emb_type=None,
                    pos_embedder=None,
                    pos_emb_dim=pos_emb_dim)


    def get_relative_positions(self, indices_refined, indices_nh):
        coords1 = self.coordinates_refined[:,indices_refined]
        coords2 = self.coordinates[:,indices_nh]
  
        coords_ref = coords2[:,:,:,[0]]
        coords2 = coords2.view(coords2.shape[0], coords2.shape[1],coords2.shape[2],-1)

        pos1_refined, pos2_refined = get_distance_angle(coords1[0], coords1[1], coords_ref[0], coords_ref[1], base=self.pos_calculation)
        pos1_nh, pos2_nh = get_distance_angle(coords2[0], coords2[1], coords_ref[0], coords_ref[1], base=self.pos_calculation)

        return (pos1_nh.float(), pos2_nh.float()), (pos1_refined.float(), pos2_refined.float())


    def forward(self, x, local_indices, local_indices_refined, sample_dict, reshape=True):
        
        b, n, f = x.shape
        n_refined = local_indices_refined.shape[1]

        n_refine = n_refined// n

        indices_nh, mask = get_nh_indices(self.adjc, local_cell_indices=local_indices, global_level=int(self.global_level))

        indices_nh = torch.concat((local_indices.view(b, n, 1), indices_nh), dim=-1)

        pos_nh, pos_refined = self.get_relative_positions(local_indices_refined.view(b, -1, n_refine), 
                                            indices_nh)

        x_nh = gather_nh_data(x, indices_nh, sample_dict['sample'], sample_dict['sample_level'], int(self.global_level))
        
        xk = self.proj_k(self.pos_embedder(pos_nh[0], pos_nh[1]))
        xk = xk.view(x_nh.shape)

        xq = self.proj_q(self.pos_embedder(pos_refined[0], pos_refined[1]))
        x = self.nha_layer(xk, xq=xq, xv=x_nh)

        if reshape:
            x = x.view(b, n_refined, -1)

        return x


class reduction_mlp(nn.Module):
    def __init__(self, model_dim, nh_reduction=1, output_dim=None, activation=nn.SiLU(), dropout=0) -> None: 
        super().__init__()

        output_dim = model_dim if output_dim is None else output_dim

        self.layer = nn.Sequential(
            nn.Linear(model_dim*nh_reduction, model_dim*nh_reduction, bias=False),
            nn.Dropout(dropout),
            activation,
            nn.Linear(model_dim*nh_reduction, output_dim, bias=False)
        )

    def forward(self, x):
        return self.layer(x.view(x.shape[0], x.shape[1], -1))

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


class ICON_Transformer(nn.Module):
    def __init__(self, model_settings) -> None: 
        super().__init__()

        self.model_settings = load_settings(model_settings, id='model')

        self.check_model_dir()

        self.max_seq_level = self.model_settings["max_seq_level"]
        self.dropout = self.model_settings["dropout"]
        self.pos_emb_calc = self.model_settings["pos_emb_calc"]
        self.polar = True if "polar" in  self.pos_emb_calc else False

        self.grid = xr.open_dataset(self.model_settings['processing_grid'])
        self.register_buffer('global_indices', torch.arange(len(self.grid.clon)).unsqueeze(dim=0))

        eoc = torch.tensor(self.grid.edge_of_cell.values - 1)
        self.register_buffer('eoc', eoc)

        acoe = torch.tensor(self.grid.adjacent_cell_of_edge.values - 1)
        self.register_buffer('acoe', acoe)

        self.pretrain_bias = self.model_settings['pretrain'] if 'pretrain' in self.model_settings.keys() else False
        self.pretrain_droprate = self.model_settings['pretrain_droprate'] if 'pretrain' in self.model_settings.keys() else False

        self.global_level_start = self.model_settings['global_level_start']
        self.global_level_end = self.model_settings['global_level_end']
        self.global_level_output_start = self.model_settings['global_level_output_start']

        global_indices_start = self.coarsen_indices(self.global_level_start)[0][0,:,0]
        self.register_buffer('global_indices_start', global_indices_start)

        cell_coords_global = get_coords_as_tensor(self.grid, lon='clon', lat='clat').double()
        self.register_buffer('cell_coords_global', cell_coords_global)   

        self.register_buffer('cell_coords_input', cell_coords_global[:, global_indices_start])    

        self.input_data  = self.model_settings['variables_source']
        self.output_data = self.model_settings['variables_target']

        n_input_grids = len(self.input_data)
        self.pos_emb_type = self.model_settings["pos_embedding_type"]
        self.pos_emb_type_IO = self.model_settings["pos_embedding_type_IO"]

        self.n_decoder_layers = len(self.model_settings["encoder_dims"]) - self.global_level_end
        self.n_encoder_layers = len(self.model_settings["encoder_dims"]) - self.global_level_start

        self.use_skip_layers = self.model_settings['use_skip_layers']
        self.use_skip_channels = self.model_settings['use_skip_channels']

        self.global_pos_embedder = self.init_position_embedder(self.model_settings["emb_table_dim"], min_coarsen_level=0, max_coarsen_level=len(self.model_settings["encoder_dims"]), embed_dim=self.model_settings["emb_table_bins"])
        
        self.global_pos_embedder_refine = self.init_position_embedder(self.model_settings["emb_table_dim"], min_coarsen_level=0, max_coarsen_level=len(self.model_settings["encoder_dims"]), embed_dim=self.model_settings["emb_table_bins"])

        input_mapping, input_coordinates = self.get_input_grid_mapping()
        output_mapping, output_coordinates = self.get_output_grid_mapping()

        self.input_layers = self.init_input_layers(input_mapping, input_coordinates)
        
        out_dim_input = self.model_settings["encoder_dims"][self.global_level_start]

        self.input_projection = nn.Sequential(nn.Linear(self.model_settings["encoder_dims"][0] * n_input_grids, out_dim_input, bias=False))
  
        self.coarsen_layers = nn.ModuleDict()
        self.processing_layers_enc = nn.ModuleDict()

        self.refinement_layers = nn.ModuleDict()
        self.processing_layers_dec = nn.ModuleDict()
        self.skip_layers = nn.ModuleDict()

        self.decoder_dims_level = {}
        self.encoder_dims_level = {}

        if not self.pretrain_bias:     
        #self.mid_layers = nn.ModuleList()        

            for k in range(1 + self.global_level_start, len(self.model_settings["encoder_dims"])):
                
                global_level = str(k - 1)

                dim_in  = self.model_settings["encoder_dims"][k-1]
                dim_out = self.model_settings["encoder_dims"][k]

                self.encoder_dims_level[global_level] = {'in': dim_in, 'out': dim_out}           
            
                adjc = self.get_adjacent_global_cell_indices(int(global_level))    
                coordinates = self.get_coordinates_level(global_level)

                self.processing_layers_enc[global_level] = processing_layers(
                                            global_level,
                                            adjc,
                                            coordinates,
                                            n_layers = self.model_settings['n_processing_layers'],
                                            input_dim = dim_in,
                                            model_dim = dim_in,
                                            ff_dim = dim_in,
                                            n_heads =  self.model_settings["n_heads"][k],
                                            dropout=self.dropout,
                                            pos_emb_type= self.pos_emb_type,
                                            pos_embedder=self.global_pos_embedder,
                                            pos_emb_dim=self.model_settings["emb_table_dim"],
                                            seq_level=self.max_seq_level,
                                            polar = self.polar)   
                
                self.coarsen_layers[global_level] = coarsen_layer(
                                                        input_dim= dim_in,
                                                        model_dim = dim_out,
                                                        ff_dim = dim_out,
                                                        n_heads = self.model_settings["n_heads"][k],
                                                        pos_emb_type= self.pos_emb_type,
                                                        dropout=self.dropout,
                                                        pos_embedder=self.global_pos_embedder,
                                                        pos_emb_dim=self.model_settings["emb_table_dim"])

            for k in range(self.n_decoder_layers):

                global_level = str(len(self.model_settings["encoder_dims"]) - 1 - k)

                dim_in  = self.model_settings["encoder_dims"][::-1][k]
                dim_out = self.model_settings["encoder_dims"][::-1][k+1] if k < self.n_decoder_layers - 1 else dim_in

                if self.use_skip_channels:
                    if global_level in self.encoder_dims_level.keys():
                        dim_in += self.encoder_dims_level[global_level]["in"]

                self.decoder_dims_level[global_level] = dim_in

                adjc = self.get_adjacent_global_cell_indices(int(global_level))
                coordinates = self.get_coordinates_level(global_level)

                self.processing_layers_dec[global_level] = processing_layers(
                                        global_level,
                                        adjc,
                                        coordinates,
                                        n_layers = self.model_settings['n_processing_layers'],
                                        input_dim = dim_in,
                                        model_dim = dim_in,
                                        ff_dim = dim_out,
                                        n_heads = self.model_settings["n_heads"][k],
                                        dropout=self.dropout,
                                        pos_emb_type= self.pos_emb_type,
                                        pos_embedder=self.global_pos_embedder,
                                        pos_emb_dim=self.model_settings["emb_table_dim"],
                                        seq_level=self.max_seq_level,
                                        polar = self.polar)
                            
                    
                if k < self.n_decoder_layers - 1:
                    self.refinement_layers[global_level] = refinement_layer(
                            global_level,
                            adjc,
                            coordinates_refined = self.get_coordinates_level(int(global_level)-1),
                            coordinates = coordinates,
                            input_dim = dim_in,
                            model_dim = dim_out,
                            ff_dim = dim_out,
                            dropout=self.dropout,
                            n_heads = self.model_settings["n_heads"][k],
                            pos_emb_type= self.pos_emb_type_IO,
                            pos_embedder=self.global_pos_embedder_refine,
                            pos_emb_dim=self.model_settings["emb_table_dim"],
                            seq_level=self.max_seq_level,
                            polar = self.polar)
                    
                    global_level = str(int(global_level)-1)
                    if self.use_skip_layers and global_level in self.encoder_dims_level.keys():
                        self.skip_layers[global_level] = skip_layer(
                                                input_dim = dim_out,
                                                model_dim = dim_out,
                                                ff_dim = dim_out,
                                                n_heads = self.model_settings["n_heads"][k],
                                                dropout= self.dropout,
                                                pos_emb_type= self.pos_emb_type,
                                                pos_embedder=self.global_pos_embedder,
                                                pos_emb_dim=self.model_settings["emb_table_dim"],
                                                seq_level=self.max_seq_level)
        else:    
            self.decoder_dims_level['0'] = self.model_settings["encoder_dims"][self.global_level_start] 

        self.output_layers = self.init_output_layers(output_mapping, output_coordinates)

        if "pretrained_path" in self.model_settings.keys():
            self.check_pretrained(log_dir_check=self.model_settings['pretrained_path'])

    def forward(self, x, indices_batch_dict=None, debug=False):
        # if global_indices are provided, batches in x are treated as independent
        debug_list = []

        if indices_batch_dict is None:
            indices_batch_dict = {'global_cell': None,
                       'sample': None,
                       'sample_level': None}

        input_data = []
        for key, values in x.items():
            
            input = self.input_layers[key](values, indices_batch_dict["local_cell"], self.cell_coords_input)
            input_data.append(input) 
        
        x = torch.concat(input_data, dim=-1)

        x = self.input_projection(x)

        x_skip = {}
        indices_global_level = indices_batch_dict['global_cell']
        for global_level, coarsen_layer in self.coarsen_layers.items():
            

            x = self.processing_layers_enc[global_level](x, indices_global_level, indices_batch_dict)

            if self.use_skip_channels or self.use_skip_layers:
                x_skip[global_level] = x
            

            indices_global_level = indices_global_level.reshape(indices_global_level.shape[0],-1,4)

            pos = self.get_relative_positions(indices_global_level,
                                            indices_global_level)
            
            x = coarsen_layer(x, pos=pos)

            indices_global_level = indices_global_level[:,:,0]

            if debug:
                debug_list.append({'pos':pos,
                                   'indices':indices_global_level,
                                   'input': x,
                                   'output':x,
                                    'layer': f'coarsen_enc_{global_level}'})

        #continue here
        
        x_output = {}
        if self.pretrain_bias:
            x_output['0'] = x

        for global_level, processing_layers in self.processing_layers_dec.items():
            
            if self.use_skip_channels and global_level in x_skip.keys():
                x = torch.concat((x, x_skip[global_level]), dim=-1)

            indices_global_level = self.get_global_indices_global(indices_batch_dict['sample'], indices_batch_dict['sample_level'], int(global_level))

            x = processing_layers(x, indices_global_level, indices_batch_dict)

            if int(global_level) <= self.global_level_output_start:
                x_output[global_level] = x

            # refinement layers
            if global_level in self.refinement_layers.keys():
                #x_nh, mask, indices_global, indices_global_nh = self.get_nh(x, indices, int(global_level))
             #   x_nh = gather_nh_data(x, local_indices_nh, indices_batch_dict['sample'], indices_batch_dict['sample_level'],  int(global_level))

              #  indices_global_refined = self.get_global_indices_global(indices_batch_dict['sample'], indices_batch_dict['sample_level'], int(global_level)-1)

                
               # pos = self.get_relative_positions(indices_global_refined.reshape(-1, indices_global_nh.shape[1], 4),
               #                                 indices_global_nh.squeeze())
  
                local_indices_level = indices_global_level // 4**(int(global_level))

                indices_global_refined = self.get_global_indices_global(indices_batch_dict['sample'], indices_batch_dict['sample_level'], int(global_level) -1)
                local_indices_refined = indices_global_refined // 4**(int(global_level)-1)

                x = self.refinement_layers[global_level](x, local_indices_level, local_indices_refined, indices_batch_dict)
                
                global_level = str(int(global_level) - 1)
                if self.use_skip_layers and global_level in x_skip.keys():
                    indices_sequence = sequenize(indices_global_refined, self.max_seq_level)
                    pos_seq = self.get_relative_positions(indices_sequence, 
                                                    indices_sequence)
                    
                    if self.model_settings['reverse_skip']:
                        x = self.skip_layers[global_level](x_skip[global_level], x , pos=pos_seq)
                    else:
                        x = self.skip_layers[global_level](x, x_skip[global_level], pos=pos_seq)
        
        k = 0
        for global_level in self.output_layers.keys():
            if k==0:
                output_data = {}

            for key in self.output_data.keys():
            
                indices_global_level = self.get_global_indices_global(indices_batch_dict['sample'], indices_batch_dict['sample_level'], int(global_level))

                if key in output_data.keys():
                    output_data[key] += self.output_layers[global_level][key](x_output[global_level], indices_global_level, indices_batch_dict)
                else:
                    output_data[key] = self.output_layers[global_level][key](x_output[global_level], indices_global_level, indices_batch_dict)

            k+=1

        if debug:
            return output_data, debug_list
        else:
            return output_data
    

    def get_input_grid_mapping(self):
        
        input_coordinates = {}
        for grid_type in self.input_data.keys():
            input_coordinates[grid_type] = get_coords_as_tensor(xr.open_dataset(self.model_settings['input_grid']),grid_type=grid_type)
           #global_indices = self.coarsen_indices(self.global_level_start)[0]
           # input_coordinates[grid_type] = input_coordinates[grid_type][:,global_indices[0,:,0]]
       

        input_mapping = get_nh_variable_mapping_icon(self.model_settings['processing_grid'], ['cell'], 
                                    self.model_settings['input_grid'], self.input_data, 
                                    search_raadius=self.model_settings['search_raadius'], 
                                    max_nh=self.model_settings['nh_input'], 
                                    level_start=self.model_settings['level_start_input'], 
                                    lowest_level=self.model_settings['global_level_start'])

        
        return input_mapping, input_coordinates

    def get_output_grid_mapping(self):
        
        output_mapping = get_nh_variable_mapping_icon(self.model_settings['processing_grid'], ['cell'], 
                                    self.model_settings['processing_grid'], self.output_data, 
                                    search_raadius=self.model_settings['search_raadius'], max_nh=self.model_settings['nh_input'], level_start=self.model_settings['level_start_input'])

        output_coordinates = {}
        for grid_type in self.output_data.keys():
            output_coordinates[grid_type] = get_coords_as_tensor(xr.open_dataset(self.model_settings['processing_grid']),grid_type=grid_type)

        return output_mapping, output_coordinates


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
    

    def init_input_layers(self, input_mapping, input_coordinates):

        model_dim = ff_dim =  self.model_settings['encoder_dims'][0]
        n_heads = self.model_settings['n_heads'][0]

        if 'share_emb_w_input' in self.model_settings.keys() and not self.model_settings['share_emb_w_input']:
            pos_embedder = self.init_position_embedder(model_dim, min_coarsen_level=0, max_coarsen_level=self.global_level_start+1, embed_dim=self.model_settings["emb_table_bins"])
            emb_dim = model_dim
        else:
            pos_embedder = self.global_pos_embedder_refine
            emb_dim = self.model_settings["emb_table_dim"]

        input_layers = nn.ModuleDict()
        for key in input_mapping["cell"].keys():
            
            n_input = len(self.input_data[key])

            layer = input_layer(
                    input_mapping["cell"][key],
                    input_coordinates[key],
                    seq_level=self.model_settings['input_seq_level'],
                    input_dim = n_input,
                    model_dim = model_dim,
                    dropout=self.dropout,
                    ff_dim = ff_dim,
                    n_heads = n_heads,
                    pos_embedder=pos_embedder,
                    pos_emb_type = self.pos_emb_type_IO,
                    pos_emb_dim=emb_dim,
                    polar=self.polar,
                    input_mlp = False if self.pretrain_bias else True,
                    force_nha = self.pretrain_bias,
                    kv_dropout=0 if not self.pretrain_bias else self.pretrain_droprate)
            
            input_layers[key] = layer

        return input_layers
    
 
    def init_output_layers(self, output_mapping, output_coordinates):
        
        output_layers = nn.ModuleDict()

        for global_level, input_dim in self.decoder_dims_level.items():
            
            if int(global_level) <= self.global_level_output_start:

                input_dim = self.decoder_dims_level[global_level]
            
                if 'share_emb_w_output' in self.model_settings.keys() and not self.model_settings['share_emb_w_output']:
                    pos_embedder = self.init_position_embedder(input_dim, min_coarsen_level=int(global_level), max_coarsen_level=int(global_level), embed_dim=self.model_settings["emb_table_bins"])
                    emb_dim = input_dim
                else:
                    pos_embedder = self.global_pos_embedder_refine
                    emb_dim = self.model_settings["emb_table_dim"]

                output_layers_level = nn.ModuleDict()
                for key in self.output_data.keys():    
                    
                    output_dim = len(self.output_data[key])
                    n_heads = self.model_settings['n_heads'][0]

                    adjc = self.get_adjacent_global_cell_indices(int(global_level))
                    coordinates = self.get_coordinates_level(int(global_level))  
                    #output_mapping = 
                    
                    layer = output_layer(
                        global_level,
                        adjc,
                        coordinates,
                        output_mapping["cell"][key],
                        output_coordinates[key],
                        input_dim = input_dim,
                        model_dim = input_dim,
                        output_dim = output_dim,
                        dropout=self.dropout,
                        ff_dim = input_dim,
                        n_heads = n_heads,
                        pos_emb_type = self.pos_emb_type_IO,
                        pos_embedder = pos_embedder,
                        pos_emb_dim = emb_dim,
                        seq_level = self.max_seq_level,
                        polar= self.polar)
                
                    output_layers_level[key] = layer

        output_layers[global_level] = output_layers_level

        return output_layers
    
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

        if pretrain_subdir is not None:
            self.check_pretrained(os.path.join(self.model_dir, pretrain_subdir, 'logs'))

        train_settings = self.train_settings

        train_settings["variables_source"] = self.model_settings["variables_source"]
        train_settings["variables_target"] = self.model_settings["variables_target"]
        train_settings['model_dir'] = self.model_dir

     
        trainer.train(self, train_settings, self.model_settings)



    def check_pretrained(self, log_dir_check='', strict=False):

        if len(log_dir_check)>0:
            ckpt_dir = os.path.join(log_dir_check, 'ckpts')
            weights_path = os.path.join(ckpt_dir, 'best.pth')
            if not os.path.isfile(weights_path):
                weights_paths = [f for f in os.listdir(ckpt_dir) if 'pth' in f]
                weights_paths.sort(key=getint)
                if len(weights_path)>0:
                    weights_path = os.path.join(ckpt_dir, weights_paths[-1])
            
            if os.path.isfile(weights_path):
                self.load_pretrained(weights_path, strict=strict)

    def load_pretrained(self, ckpt_path:str, strict=False):
        device = 'cpu' if 'device' not in self.model_settings.keys() else self.model_settings['device']
        ckpt_dict = torch.load(ckpt_path, map_location=torch.device(device))
        load_model(ckpt_dict, self, strict=False)



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