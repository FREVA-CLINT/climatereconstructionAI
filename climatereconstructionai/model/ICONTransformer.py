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
    def __init__(self, input_dim, model_dim, ff_dim, n_heads=4, dropout=0, input_mlp=False, output_mlp=False, output_dim=None, activation=nn.SiLU(), q_res=True, pos_emb_type='bias', pos_embedding_table=None, pos_emb_dim=None) -> None: 
        super().__init__()

        self.model_dim = model_dim
        self.n_heads = n_heads
        self.pos_emb_type = pos_emb_type

        if input_mlp:
            self.input_mlp = nn.Sequential(
                nn.Linear(input_dim, model_dim, bias=False)
            )
        else:
            self.input_mlp = nn.Identity()
            input_dim=model_dim
        
        if output_mlp:
            self.output_mlp = nn.Sequential(
                nn.Linear(model_dim, output_dim, bias=False)
            )
        else:
            self.output_mlp = nn.Identity()

        self.pos_embedding_table = pos_embedding_table

        if self.pos_emb_type=='bias':
            self.emb_proj_bias = position_embedding_proj(pos_emb_dim, n_heads)
        elif self.pos_emb_type=='context':
            self.emb_proj_q = position_embedding_proj(pos_emb_dim, model_dim // n_heads)
            self.emb_proj_k = position_embedding_proj(pos_emb_dim, model_dim // n_heads)
            self.emb_proj_v = position_embedding_proj(pos_emb_dim, model_dim // n_heads)

        self.MHA = helpers.MultiHeadAttentionBlock(
            model_dim, model_dim, n_heads, qkv_proj=True
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
            

        if self.pos_emb_type =='context' and pos is not None:
            aq = self.emb_proj_k(pos[0], pos[1], self.pos_embedding_table)
            ak = self.emb_proj_k(pos[0], pos[1], self.pos_embedding_table)
            av = self.emb_proj_v(pos[0], pos[1], self.pos_embedding_table)

            att_out, att = self.MHA(q, k, v, aq=aq, ak=ak, av=av, return_debug=True, mask=mask) 

        elif self.pos_emb_type =='bias' and pos is not None:
            bias = self.emb_proj_bias(pos[0], pos[1], self.pos_embedding_table)
            att_out, att = self.MHA(q, k, v, bias=bias, return_debug=True, mask=mask)    

        else:
            att_out, att = self.MHA(q, k, v, return_debug=True, mask=mask) 

        if self.q_res:
            x = q + self.dropout1(att_out)
        else:
            x = self.dropout1(att_out)

        x = self.norm1(x)

        x = self.norm2(x + self.dropout2(self.mlp_layer(x)))

        x = x.view(b,n,-1,e)

        return self.output_mlp(x)

class n_nha_layers(nn.Module):
    def __init__(self, n_layers, input_dim, model_dim, ff_dim, n_heads=4, dropout=0, input_mlp=False, output_mlp=False, output_dim=None, activation=nn.SiLU(), pos_emb_type='bias', pos_embedding_table=None, pos_emb_dim=None) -> None: 
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
                                        input_mlp=input_mlp, 
                                        output_mlp=output_mlp, 
                                        output_dim=output_dim, 
                                        activation=activation, 
                                        pos_emb_type=pos_emb_type, 
                                        pos_embedding_table=pos_embedding_table, 
                                        pos_emb_dim=pos_emb_dim))

    def forward(self, x: torch.tensor, xq=None, mask=None, pos=None):
        for layer in self.layer_list:
            x = layer(x, xq=xq, mask=mask, pos=pos)
        return x

    
class nha_reduction_layer(nn.Module):
    def __init__(self, input_dim, model_dim, ff_dim, nh_reduction, input_mlp=True, n_heads=4, dropout=0, activation=nn.SiLU(), pos_emb_type='bias', pos_embedding_table=None, pos_emb_dim=None) -> None: 
        super().__init__()

        self.nha_layer = nha_layer(
                input_dim = input_dim,
                model_dim = model_dim,
                ff_dim = ff_dim,
                n_heads = n_heads,
                input_mlp = input_mlp,
                dropout=dropout,
                pos_emb_type=pos_emb_type,
                pos_embedding_table=pos_embedding_table,
                pos_emb_dim=pos_emb_dim)
        
        
        self.reduction_mlp = reduction_mlp(model_dim, nh_reduction=nh_reduction)
        
    def forward(self, x, pos=None):

        x = self.nha_layer(x, pos=pos)
        
        x = self.reduction_mlp(x)

        return x


class coarsen_layer(nn.Module):
    def __init__(self, input_dim, model_dim, ff_dim, n_heads=4, dropout=0, activation=nn.SiLU(), pos_emb_type='bias', pos_embedding_table=None, pos_emb_dim=None) -> None: 
        super().__init__()

        self.nha_reduction_layer = nha_reduction_layer(
                input_dim = input_dim,
                model_dim = model_dim,
                ff_dim = ff_dim,
                n_heads = n_heads,
                nh_reduction = 4,
                input_mlp = False,
                dropout=dropout,
                pos_emb_type=pos_emb_type,
                pos_embedding_table=pos_embedding_table,
                pos_emb_dim=pos_emb_dim)
        
    def forward(self, x, pos=None):

        x = x.reshape(x.shape[0],-1,4,x.shape[-1])
        x = self.nha_reduction_layer(x, pos=pos)

        return x
    

class processing_layer(nn.Module):
    def __init__(self, input_dim, model_dim, ff_dim, n_heads=4, dropout=0, activation=nn.SiLU(),input_mlp=False, pos_emb_type='bias', pos_embedding_table=None, pos_emb_dim=None, seq_level=1, nh_att=True, seq_att=True) -> None: 
        super().__init__()
        self.nh_att = nh_att
        self.seq_att = seq_att

        if nh_att:
            self.nh_layer = nha_layer(input_dim=input_dim,
                        model_dim = model_dim,
                        ff_dim = ff_dim,
                        n_heads =  n_heads,
                        input_mlp = input_mlp,
                        dropout=dropout,
                        pos_emb_type=pos_emb_type,
                        pos_embedding_table=pos_embedding_table,
                        pos_emb_dim=pos_emb_dim,
                        activation=activation)

        if seq_att:
            self.seq_layer = nha_layer(input_dim=input_dim,
                        model_dim = model_dim,
                        ff_dim = ff_dim,
                        n_heads =  n_heads,
                        input_mlp = False,
                        dropout=dropout,
                        pos_emb_type=pos_emb_type,
                        pos_embedding_table=pos_embedding_table,
                        pos_emb_dim=pos_emb_dim,
                        activation=activation)    
            
        self.seq_level = seq_level
        
    def forward(self, x: torch.tensor, pos_seq: torch.tensor, x_nh=None, mask=None, pos_nh=None):
        b, n, e = x.shape

        if x_nh is not None and self.nh_att:
            x = self.nh_layer(x_nh, xq=x, mask=mask.unsqueeze(dim=-2), pos=pos_nh)
            x = x.view(b, n, -1)
        
        if self.seq_att:
            x = sequenize(x, self.seq_level)
            x = self.seq_layer(x, pos=pos_seq)
            x = x.view(b, n, -1)

        return x   

class input_layer(nn.Module):
    def __init__(self, input_mapping, input_coordinates, input_dim, model_dim, ff_dim, seq_level=1, n_heads=4, dropout=0, output_mlp=False, pos_emb_type='bias', pos_embedding_table=None, pos_emb_dim=None) -> None: 
        super().__init__()

        self.register_buffer("input_mapping", input_mapping)
        self.register_buffer("input_coordinates", input_coordinates)
        self.seq_level = seq_level

        self.nha_layer = nha_layer(
                    input_dim = model_dim,
                    model_dim = model_dim,
                    ff_dim = ff_dim,
                    n_heads = n_heads,
                    output_mlp = False,
                    input_mlp = False,
                    dropout=dropout,
                    q_res=False,
                    pos_emb_type=pos_emb_type,
                    pos_embedding_table=None,
                    pos_emb_dim=pos_emb_dim)


        self.pos_embedding_table = pos_embedding_table

        self.input_mlp = nn.Sequential(
                        nn.Linear(input_dim, model_dim, bias=False),
                        nn.SiLU()
                    )
        
        self.emb_proj = position_embedding_proj(model_dim, model_dim)


    def get_relative_positions(self, grid_level_indices, grid_level_coords):
        
        indices = self.input_mapping[grid_level_indices]

        coords1 = grid_level_coords[:, grid_level_indices]

        coords2 = self.input_coordinates[:, indices]

        # use mid point of sequence for "absolute-relative" coordinates
        coords_ref = coords1[:,:,:,[0]]
        coords2 = coords2.view(coords2.shape[0], coords2.shape[1],coords2.shape[2],-1)

        distances_grid, phis_grid = get_distance_angle(coords1[0], coords1[1], coords_ref[0], coords_ref[1])
        distances_source, phis_source = get_distance_angle(coords2[0], coords2[1], coords_ref[0], coords_ref[1])

        return (distances_source.float(), phis_source.float()), (distances_grid.float(), phis_grid.float())
    

    def forward(self, x, grid_level_indices, grid_level_coords, reshape=True, mask=None):
        b,n,nh,f = x.shape

        grid_level_indices = sequenize(grid_level_indices, self.seq_level)

        pos_source, pos_grid = self.get_relative_positions(grid_level_indices,
                                                            grid_level_coords)
        
        x = sequenize(x, self.seq_level)
        x = x.reshape(b,x.shape[1],-1,f)

        x = self.input_mlp(x)

        xk = self.emb_proj(pos_source[0], pos_source[1], self.pos_embedding_table)
        xk = xk.view(x.shape)

        xq = self.emb_proj(pos_grid[0], pos_grid[1], self.pos_embedding_table)
        x = self.nha_layer(xk, xq=xq, xv=x)

        x = x.view(b, n, -1)

        return x


class output_layer(nn.Module):
    # change to dynamic output sequences
    def __init__(self, output_mapping, output_coordinates, input_dim, model_dim, ff_dim, n_heads=4, dropout=0, output_dim=None, output_mlp=False, pos_emb_type='bias', pos_embedding_table=None, pos_emb_dim=None) -> None: 
        super().__init__()    

        self.register_buffer("output_mapping", output_mapping)
        self.register_buffer("output_coordinates", output_coordinates)

        self.layer = refinement_layer(
                input_dim=input_dim,
                model_dim = model_dim,
                output_dim = output_dim,
                ff_dim = ff_dim,
                n_heads = n_heads,
                output_mlp=True,
                pos_emb_type=pos_emb_type,
                pos_embedding_table= pos_embedding_table,
                pos_emb_dim=pos_emb_dim)


    def get_relative_positions(self, grid_sample_indices, grid_sample_indices_nh, grid_coords):
        
        indices = self.output_mapping[grid_sample_indices]
        coords1 = self.output_coordinates[:, indices]

        coords2 = grid_coords[:,grid_sample_indices_nh]

        while coords1.dim()<=4:
            coords1 = coords1.unsqueeze(dim=-1)
        if coords2.dim()==4:
            coords2 = coords2.unsqueeze(dim=-2)  

        distances, phis = get_distance_angle(coords1[0], coords1[1], coords2[0], coords2[1])

        return distances.float(), phis.float()
    
    def forward(self, x, x_nh, grid_sample_indices, grid_sample_indices_nh, grid_coords, mask=None):
        
        pos_output = self.get_relative_positions(grid_sample_indices,
                                            grid_sample_indices_nh,
                                            grid_coords)
        
        x = self.layer(x, x_nh, mask=mask, pos=pos_output, reshape=False)

        return x

class skip_layer(nn.Module):
    def __init__(self, input_dim, model_dim, ff_dim, n_heads=4, dropout=0, output_dim=None, output_mlp=False, pos_emb_type='bias', pos_embedding_table=None, pos_emb_dim=None, seq_level=1) -> None: 
        super().__init__()
        
        self.seq_level = seq_level

        self.nha_layer = nha_layer(
                    input_dim = input_dim,
                    model_dim = model_dim,
                    ff_dim = ff_dim,
                    n_heads = n_heads,
                    output_mlp = False,
                    input_mlp = False,
                    dropout=dropout,
                    pos_emb_type=pos_emb_type,
                    pos_embedding_table=pos_embedding_table,
                    pos_emb_dim=pos_emb_dim)
        

    def forward(self, x, x_skip, reshape=True, mask=None, pos=None):

        #add position embeddings of refined layer level(x_nh) != level(xq)
        x = sequenize(x, self.seq_level)
        x_skip = sequenize(x_skip, self.seq_level)

        x = self.nha_layer(x_skip, xq=x, mask=mask, pos=pos)

        if reshape:
            x = x.view(x.shape[0],-1, x.shape[-1])


        return x

class refinement_layer(nn.Module):
    def __init__(self, input_dim, model_dim, ff_dim, n_heads=4, dropout=0, output_dim=None, output_mlp=False, pos_emb_type='bias', pos_embedding_table=None, pos_emb_dim=None) -> None: 
        super().__init__()

        self.nha_layer = nha_layer(
                    input_dim = input_dim,
                    model_dim = model_dim,
                    ff_dim = ff_dim,
                    n_heads = n_heads,
                    output_mlp = False,
                    input_mlp = False,
                    dropout=dropout,
                    pos_emb_type=pos_emb_type,
                    pos_embedding_table=pos_embedding_table,
                    pos_emb_dim=pos_emb_dim)
        
        if output_mlp:
            output_dim = model_dim if output_dim is None else output_dim 
            self.mlp_layer = nn.Sequential(
                nn.Linear(model_dim, ff_dim, bias=True),
                nn.SiLU(),
                nn.Linear(ff_dim, output_dim, bias=True)
            )
        else:
            self.mlp_layer = nn.Identity()


    def forward(self, x, x_nh, reshape=True, mask=None, pos=None):
        
        n_refine = pos[0].shape[-2]

        x = x.unsqueeze(dim=-2).repeat_interleave(n_refine, dim=-2)

        if mask is not None:
            mask = mask.unsqueeze(dim=-2).repeat_interleave(n_refine, dim=-2)

        x = self.nha_layer(x_nh, xq=x, mask=mask, pos=pos)

        x = self.mlp_layer(x)

        if reshape:
            x = x.view(x.shape[0],-1, x.shape[-1])


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

class embedding_table(nn.Module):
    def __init__(self, min_dist, max_dist, embed_dim, n_out) -> None: 
        super().__init__()

        #self.dist_emb = helpers.PositionEmbedder_phys(min_dist, max_dist, embed_dim, n_heads=n_out)
        self.dist_emb = nn.Linear(1, n_out)
        self.phi_emb = helpers.PositionEmbedder_phys(-torch.pi, torch.pi, embed_dim, n_heads=n_out)

    def forward(self, distances, phis):
        
        return self.dist_emb(distances.unsqueeze(dim=-1)), self.phi_emb(phis)[0]


class position_embedding_proj(nn.Module):
    def __init__(self, embed_dim, model_dim, operation='product') -> None: 
        super().__init__()

        self.operation = operation
        
        self.proj_layer = nn.Linear(embed_dim, model_dim, bias=False)


    def forward(self, distances, phis, emb_table):
        distance_emb, phi_emb = emb_table(distances, phis)

        if self.operation == 'product':
            return self.proj_layer(distance_emb*phi_emb)
        else:
            return self.proj_layer(distance_emb+phi_emb)


class position_embedder(nn.Module):
    def __init__(self, embed_dim, n_out) -> None: 
        super().__init__()

        self.dist_emb = nn.Linear(1, n_out, bias=False)
        self.phi_emb = helpers.PositionEmbedder_phys(-2*torch.pi, 2*torch.pi, embed_dim, n_heads=n_out)
        self.emb_mlp = nn.Sequential(nn.Linear(n_out*2, n_out, bias=False))


    def forward(self, distances, phis):
        
        p_dist = self.dist_emb(distances.float().unsqueeze(dim=-1))
        p_phi = self.phi_emb(phis)[0]

        return self.emb_mlp(torch.concat((p_dist,p_phi), dim=-1))



class ICON_Transformer(nn.Module):
    def __init__(self, model_settings) -> None: 
        super().__init__()

        self.model_settings = load_settings(model_settings, id='model')

        self.check_model_dir()

        self.grid = xr.open_dataset(self.model_settings['processing_grid'])
        self.register_buffer('global_indices', torch.arange(len(self.grid.clon)).unsqueeze(dim=0))

        eoc = torch.tensor(self.grid.edge_of_cell.values - 1)
        self.register_buffer('eoc', eoc)

        acoe = torch.tensor(self.grid.adjacent_cell_of_edge.values - 1)
        self.register_buffer('acoe', acoe)

        self.global_level_start = self.model_settings['global_level_start']

        global_indices_start = self.coarsen_indices(self.global_level_start)[0][0,:,0]
        self.register_buffer('global_indices_start', global_indices_start)

        cell_coords_global = get_coords_as_tensor(self.grid, lon='clon', lat='clat').double()
        self.register_buffer('cell_coords_global', cell_coords_global)        

        self.register_buffer('cell_coords_input', cell_coords_global[:, global_indices_start])    

        self.input_data  = self.model_settings['variables_source']
        self.output_data = self.model_settings['variables_target']

        n_input_grids = len(self.input_data)
        self.pos_emb_type = self.model_settings["pos_embedding_type"]

        self.n_decoder_layers = len(self.model_settings["encoder_dims"])
        self.n_encoder_layers = len(self.model_settings["encoder_dims"]) - self.global_level_start

        self.global_emb_table = self.init_pos_embedding_table(self.model_settings["emb_table_dim"], embed_dim=self.model_settings["emb_table_bins"])
        
        input_mapping, input_coordinates = self.get_input_grid_mapping()
        output_mapping, output_coordinates = self.get_output_grid_mapping()

        self.input_layers = self.init_input_layers(input_mapping, input_coordinates)
        self.output_layers = self.init_output_layers(output_mapping, output_coordinates)

        self.input_projection = nn.Sequential(nn.Linear(self.model_settings["encoder_dims"][0] * n_input_grids,
                                          self.model_settings["encoder_dims"][self.global_level_start], bias=True))

        self.coarsen_layers = nn.ModuleDict()
        self.processing_layers_enc = nn.ModuleDict()

        self.refinement_layers = nn.ModuleDict()
        self.processing_layers_dec = nn.ModuleDict()
        self.skip_layers = nn.ModuleDict()

        self.max_seq_level = self.model_settings["max_seq_level"]
        #self.mid_layers = nn.ModuleList()        
        self.encoder_dims_level = {}

        for k in range(1 + self.global_level_start, len(self.model_settings["encoder_dims"])):
            
            global_level = str(k - 1)

            dim_in  = self.model_settings["encoder_dims"][k-1]
            dim_out = self.model_settings["encoder_dims"][k]

            self.encoder_dims_level[global_level] = {'in': dim_in, 'out': dim_out} 
            
            processing_layers = nn.ModuleList()
            for l in range(self.model_settings["n_processing_layers"]):
                if l==0:
                    input_dim = dim_in 
                    input_mlp=True
                else:
                    input_dim = dim_out
                    input_mlp=False

                processing_layers.append(processing_layer(
                                        input_dim = input_dim,
                                        model_dim = dim_out,
                                        ff_dim = dim_out,
                                        input_mlp=input_mlp,
                                        n_heads =  self.model_settings["n_heads"][k],
                                        dropout=0,
                                        pos_emb_type= self.pos_emb_type,
                                        pos_embedding_table=self.global_emb_table,
                                        pos_emb_dim=self.model_settings["emb_table_dim"],
                                        seq_level=self.max_seq_level))
        
            self.processing_layers_enc[global_level] = processing_layers    

            
            self.coarsen_layers[global_level] = coarsen_layer(
                                                    input_dim= dim_out,
                                                    model_dim = dim_out,
                                                    ff_dim = dim_out,
                                                    n_heads = self.model_settings["n_heads"][k],
                                                    pos_emb_type= self.pos_emb_type,
                                                    pos_embedding_table=self.global_emb_table,
                                                    pos_emb_dim=self.model_settings["emb_table_dim"])


        self.use_skip_layers = self.model_settings['use_skip_layers']
        self.use_skip_channels = self.model_settings['use_skip_channels']

        for k in range(self.n_decoder_layers):

            global_level = str(self.n_decoder_layers - 1 - k)

            dim_in  = self.model_settings["encoder_dims"][::-1][k]
            dim_out = self.model_settings["encoder_dims"][::-1][k+1] if k < self.n_decoder_layers - 1 else dim_in

            if self.use_skip_channels:
                if global_level in self.encoder_dims_level.keys() and k>0:
                    dim_in += self.encoder_dims_level[global_level]["out"]
            
            processing_layers = nn.ModuleList()
            for l in range(self.model_settings["n_processing_layers"]):
                if l==0:
                    input_dim = dim_in 
                    input_mlp=True
                else:

                    input_dim = dim_out
                    input_mlp=False
                processing_layers.append(processing_layer(
                                        input_dim = input_dim,
                                        model_dim = dim_out,
                                        ff_dim = dim_out,
                                        input_mlp = input_mlp,
                                        n_heads = self.model_settings["n_heads"][k],
                                        dropout=0,
                                        pos_emb_type= self.pos_emb_type,
                                        pos_embedding_table=self.global_emb_table,
                                        pos_emb_dim=self.model_settings["emb_table_dim"],
                                        seq_level=self.max_seq_level))
                
            self.processing_layers_dec[global_level] = processing_layers               
                
            if k < self.n_decoder_layers - 1:
                self.refinement_layers[global_level] = refinement_layer(
                        input_dim = dim_out,
                        model_dim = dim_out,
                        ff_dim = dim_out,
                        n_heads = self.model_settings["n_heads"][k],
                        pos_emb_type= self.pos_emb_type,
                        pos_embedding_table=self.global_emb_table,
                        pos_emb_dim=self.model_settings["emb_table_dim"])
                
                global_level = str(int(global_level)-1)
                if self.use_skip_layers and global_level in self.encoder_dims_level.keys():
                    self.skip_layers[global_level] = skip_layer(
                                            input_dim = dim_out,
                                            model_dim = dim_out,
                                            ff_dim = dim_out,
                                            n_heads = self.model_settings["n_heads"][k],
                                            dropout=0,
                                            pos_emb_type= self.pos_emb_type,
                                            pos_embedding_table=self.global_emb_table,
                                            pos_emb_dim=self.model_settings["emb_table_dim"],
                                            seq_level=self.max_seq_level)
        


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
            
            if self.use_skip_channels or self.use_skip_layers:
                x_skip[global_level] = x

            
            local_indices_nh, mask = self.get_nh_indices(int(global_level), global_cell_indices=indices_global_level)

            indices_global_nh = self.gather_nh_data(indices_global_level, local_indices_nh, indices_batch_dict['sample'], indices_batch_dict['sample_level'], int(global_level))
            
            pos_nh = self.get_relative_positions(indices_global_level, 
                                            indices_global_nh.squeeze())
            
            indices_sequence = sequenize(indices_global_level, self.max_seq_level)
            pos_seq = self.get_relative_positions(indices_sequence, 
                                            indices_sequence)
            
            for l, layer in enumerate(self.processing_layers_enc[global_level]):
                x_nh = self.gather_nh_data(x, local_indices_nh, indices_batch_dict['sample'], indices_batch_dict['sample_level'],  int(global_level))
        
                x = layer(x, pos_seq, x_nh=x_nh, mask=mask.unsqueeze(dim=-2), pos_nh=pos_nh)



            if debug:
                debug_list.append({'pos':pos_nh,
                                   'indices':indices_global_level,
                                   'indices_att':indices_global_nh,
                                   'input': x_nh,
                                    'output':x,
                                    'layer': f'processing_enc_{global_level}'})

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
        for global_level, processing_layers in self.processing_layers_dec.items():
            
            if self.use_skip_channels and global_level in x_skip.keys():
                x = torch.concat((x, x_skip[global_level]), dim=-1)
            
            indices_global_level = self.get_global_indices_global(indices_batch_dict['sample'], indices_batch_dict['sample_level'], int(global_level))

            local_indices_nh, mask = self.get_nh_indices(int(global_level), global_cell_indices=indices_global_level)

            indices_global_nh = self.gather_nh_data(indices_global_level, local_indices_nh, indices_batch_dict['sample'], indices_batch_dict['sample_level'], int(global_level))
            
            pos_nh = self.get_relative_positions(indices_global_level, 
                                            indices_global_nh.squeeze())
            
            indices_sequence = sequenize(indices_global_level, self.max_seq_level)
            pos_seq = self.get_relative_positions(indices_sequence, 
                                            indices_sequence)
            
            for l, layer in enumerate(processing_layers):
                x_nh = self.gather_nh_data(x, local_indices_nh, indices_batch_dict['sample'], indices_batch_dict['sample_level'],  int(global_level))
        
                x = layer(x, pos_seq, x_nh=x_nh, mask=mask.unsqueeze(dim=-2), pos_nh=pos_nh)

             
            # refinement layers
            if global_level in self.refinement_layers.keys():
                #x_nh, mask, indices_global, indices_global_nh = self.get_nh(x, indices, int(global_level))
                x_nh = self.gather_nh_data(x, local_indices_nh, indices_batch_dict['sample'], indices_batch_dict['sample_level'],  int(global_level))

                indices_global_refined = self.get_global_indices_global(indices_batch_dict['sample'], indices_batch_dict['sample_level'], int(global_level)-1)

                
                pos = self.get_relative_positions(indices_global_refined.reshape(-1, indices_global_nh.shape[1], 4),
                                                indices_global_nh.squeeze())

                x = self.refinement_layers[global_level](x, x_nh, mask=mask, pos=pos)
                
                global_level = str(int(global_level) -1)
                if self.use_skip_layers and global_level in x_skip.keys():
                    indices_sequence = sequenize(indices_global_refined, self.max_seq_level)
                    pos_seq = self.get_relative_positions(indices_sequence, 
                                                    indices_sequence)
                    
                    x = self.skip_layers[global_level](x, x_skip[global_level], pos=pos_seq)

            if debug:
                debug_list.append({'pos':pos,
                                   'indices':indices_global_refined,
                                   'indices_att':indices_global_nh,
                                   'input': x_nh,
                                   'output':x,
                                    'layer': f'refine_dec_{global_level}'})
                

        x_nh = self.gather_nh_data(x, local_indices_nh, indices_batch_dict['sample'], indices_batch_dict['sample_level'],  0)

        output_data = {}
        for key in self.output_data.keys():         
            output_data[key] = self.output_layers[key](x, x_nh, indices_global_level, indices_global_nh.squeeze(), self.cell_coords_global, mask=mask)

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

    def get_nh_indices(self, global_level, global_cell_indices=None, local_cell_indices=None):
 
        adjc_global = self.get_adjacent_global_cell_indices(global_level)

        if global_cell_indices is not None:
            local_cell_indices =  global_cell_indices // 4**global_level

        local_cell_indices_nh, mask = helpers.get_nh_of_batch_indices(local_cell_indices, adjc_global)

        return local_cell_indices_nh, mask

    def gather_nh_data(self, x, local_cell_indices_nh, batch_sample_indices, sampled_level, global_level):
        # x in batches sampled from local_cell_indices_nh
        if x.dim()<3:
            x = x.unsqueeze(dim=-1)

        b,n,e = x.shape
        nh = local_cell_indices_nh.shape[-1]

        local_cell_indices_nh_batch = local_cell_indices_nh - (batch_sample_indices*4**(sampled_level - global_level)).view(-1,1,1)

        return torch.gather(x.reshape(b,-1,e),1, index=local_cell_indices_nh_batch.reshape(b,-1,1).repeat(1,1,e)).reshape(b,n,nh,e)


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

        distances, phis = get_distance_angle(coords1[0], coords1[1], coords2[0], coords2[1])

        return distances.float(), phis.float()


    def init_pos_embedding_table(self, n_out, embed_dim=64):
        # tbd: sample points for redcution of memory
        
    
        _, indices_global , indices_global_nh ,_ = self.coarsen_indices(self.n_decoder_layers)
        pos = self.get_relative_positions(indices_global, 
                                    indices_global_nh)
        
        # quantile very sensitive -> quantile embedding table? or ln fcn? or linear?
        min_dist = 1e-3
        max_dist = pos[0].quantile(0.98)
          
        emb_table = embedding_table(min_dist, max_dist, embed_dim=embed_dim, n_out=n_out)

        return emb_table
    

    def init_input_layers(self, input_mapping, input_coordinates):

        input_layers = nn.ModuleDict()
        for key in input_mapping["cell"].keys():
            
            n_input = len(self.input_data[key])

            model_dim = ff_dim =  self.model_settings['encoder_dims'][0]
            n_heads = self.model_settings['n_heads'][0]

            emb_table = self.init_pos_embedding_table(model_dim, embed_dim=self.model_settings["emb_table_bins"])

            layer = input_layer(
                    input_mapping["cell"][key],
                    input_coordinates[key],
                    seq_level=self.model_settings['input_seq_level'],
                    input_dim = n_input,
                    model_dim = model_dim,
                    ff_dim = ff_dim,
                    n_heads = n_heads,
                    pos_emb_type=self.pos_emb_type,
                    pos_embedding_table=emb_table,
                    pos_emb_dim=self.model_settings["emb_table_dim"])
            
            input_layers[key] = layer

        return input_layers
    
 
    def init_output_layers(self, output_mapping, output_coordinates):
        
        output_layers = nn.ModuleDict()

        for key in self.output_data.keys():
            
            input_dim = self.model_settings['encoder_dims'][0]
            model_dim = ff_dim = input_dim

            output_dim = len(self.output_data[key])
            n_heads = self.model_settings['n_heads'][0]            
            
            layer = output_layer(
                output_mapping["cell"][key],
                output_coordinates[key],
                input_dim=input_dim,
                model_dim = model_dim,
                output_dim = output_dim,
                ff_dim = ff_dim,
                n_heads = n_heads,
                output_mlp=True,
                pos_emb_type=self.pos_emb_type,
                pos_embedding_table=self.global_emb_table,
                pos_emb_dim=self.model_settings["emb_table_dim"])
        
            output_layers[key] = layer

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

def sequenize(tensor, max_seq_level):
    
    seq_len = tensor.shape[1]
    max_seq_level_seq = int(math.log(seq_len)/math.log(4))
    seq_level = min([max_seq_level_seq, max_seq_level])
    
    if tensor.dim()==3:
        tensor = tensor.reshape(tensor.shape[0], -1, 4**(seq_level), tensor.shape[-1])
    elif tensor.dim()==2:
        tensor = tensor.reshape(tensor.shape[0], -1, 4**(seq_level))
    elif tensor.dim()==4:
        tensor = tensor.reshape(tensor.shape[0], -1, 4**(seq_level), tensor.shape[-2], tensor.shape[-1])

    return tensor
    
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