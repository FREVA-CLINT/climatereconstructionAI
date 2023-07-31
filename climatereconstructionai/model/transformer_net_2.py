import torch
import torch.nn as nn
import torch.functional as F

import climatereconstructionai.model.transformer_helpers as helpers
from .. import config as cfg
from ..utils import grid_utils as gu

radius_earth = 6371

class Input_Net(nn.Module):
    def __init__(self, nh=10, input_dropout=0):
        super().__init__()

        self.input_dropout = nn.Dropout(input_dropout)

        self.interpolator = helpers.interpolator(cfg.device)

        self.nn_layer = helpers.nn_layer(nh)

    def forward(self, x, coord_dict):
        
        x = self.input_dropout(x)

        b,t,e = x.shape
        
        x_nearest, indices_dist, indices_lon, indices_lat = self.nn_layer(x, coord_dict['rel']['target-source'][0], coord_dict['rel']['target-source'][1])

        x_inter = self.interpolator(x, coord_dict['abs']['source'], coord_dict['abs']['target'])

        return x_nearest, x_inter, indices_dist, indices_lon, indices_lat


class CRTransNetBlock(nn.Module):
    def __init__(self, input_dim, model_dim, ff_dim, RPE_phys, n_heads=10, dropout=0.1, is_final_layer=False, logit_scale=True):
        super().__init__()

        self.is_final_layer = is_final_layer

        self.q = nn.Linear(input_dim, model_dim)
        self.k = nn.Linear(input_dim, model_dim)
        self.v = nn.Linear(input_dim, model_dim)


        self.att_layer = helpers.MultiHeadAttentionBlock(
            model_dim, input_dim, RPE_phys, n_heads, logit_scale=logit_scale
            )
                
        self.mlp_layer = nn.Sequential(
            nn.Linear(input_dim, ff_dim),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(ff_dim, input_dim),
            nn.ReLU(inplace=True)
        )

        if is_final_layer:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
        else:
            self.norm1 = nn.LayerNorm(input_dim)
            self.norm2 = nn.LayerNorm(input_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, rel_coords, return_debug=False):
        
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        if return_debug:
            att_out, att, rel_emb, b_emb = self.att_layer(q, k, v, rel_coords, return_debug=return_debug)
        else:
            att_out = self.att_layer(q, k, v, rel_coords, return_debug=return_debug)[0]

        x = x + self.dropout(att_out)
        x = self.norm1(x)

       # if self.is_final_layer:
        x = x + self.mlp_layer(x)  
        #else:
        #    x = x + self.dropout(self.mlp_layer(x))
        #x = self.norm2(x)

        if return_debug:
            return x, att, rel_emb, b_emb
        else:
            return x

class CRTransNet(nn.Module):
    def __init__(self, model_settings) -> None: 
        super().__init__()
        
        dropout = model_settings['dropout']
        n_heads = model_settings['grid']['n_heads']

        nh = model_settings['local']['nh']
        self.pass_source = model_settings['pass_source']

        if model_settings['embeddings']['rel']['polar']:
            emb_class = helpers.RelativePositionEmbedder_polar
        else:
            emb_class = helpers.RelativePositionEmbedder_cart

        model_settings['embeddings']['rel']['emb_dim'] = n_heads
        self.RPE_phys = emb_class(model_settings['embeddings']['rel'])

        self.input_net = Input_Net(
           nh=nh)

        
        self.LocalBlocks = nn.ModuleList()

        for _ in range(model_settings['local']['n_layers']):
            LocalBlock = CRTransNetBlock(
                input_dim=1,
                model_dim=model_settings['local']['model_dim'],
                RPE_phys=self.RPE_phys,
                ff_dim=model_settings['local']['ff_dim'],
                n_heads=n_heads,
                dropout=dropout,
                is_final_layer=True,
                logit_scale=model_settings['logit_scale'])
            
            self.LocalBlocks.append(LocalBlock)

        
        if self.pass_source:
            input_dim_grid=nh
        else:
            input_dim_grid = 1

        self.GridBlocks = nn.ModuleList()
        for _ in range(model_settings['grid']['n_layers']):
            GridBlock = CRTransNetBlock(
                    input_dim=input_dim_grid,
                    model_dim=model_settings['grid']['model_dim'],
                    RPE_phys=self.RPE_phys,
                    ff_dim=model_settings['grid']['ff_dim'],
                    n_heads=n_heads,
                    dropout=dropout,
                    is_final_layer=False,
                    logit_scale=model_settings['logit_scale'])
            self.GridBlocks.append(GridBlock)
        

        self.mlp_out = nn.Sequential(nn.Linear(nh,1),nn.ReLU(inplace=True))
        

    def forward(self, x, coord_dict, return_debug=False):
        b,s,e = x.shape
        
        x, x_inter, indices_dist, _ ,_ = self.input_net(x, coord_dict)

        d_lon = coord_dict['abs']['source'][0].view(-1)[indices_dist].unsqueeze(dim=-1)
        d_lat = coord_dict['abs']['source'][1].view(-1)[indices_dist].unsqueeze(dim=-1)

        d_lon = d_lon - d_lon.transpose(1,-1)
        d_lat = d_lat - d_lat.transpose(1,-1)

        b,t,nh,e = x.shape

        x = x.view(-1,nh,e)
        d_lon = d_lon.repeat(b,1,1)
        d_lat = d_lat.repeat(b,1,1)


        for block in self.LocalBlocks:
            if return_debug:
                x, att, rel_emb, b_emb =  block(x, [d_lon, d_lat], return_debug=return_debug)
            else:
                x =  block(x, [d_lon, d_lat], return_debug=return_debug)

        x = x.view(b,t,nh)

        if not self.pass_source:
            x = x_inter + self.mlp_out(x)
        
        d_lon = coord_dict['rel']['target'][0]
        d_lat = coord_dict['rel']['target'][1]

        for block in self.GridBlocks:
            if return_debug:
                x, att, rel_emb, b_emb =  block(x, [d_lon, d_lat], return_debug=return_debug)
            else:
                x =  block(x, [d_lon, d_lat], return_debug=return_debug)

        if self.pass_source:
            x = x_inter + self.mlp_out(x)

        if return_debug:
            debug_dict = {'RPE_emb_table': {'lat':self.RPE_phys.emb_table_lon.embeddings_table,'lon':self.RPE_phys.emb_table_lat.embeddings_table},
                    'att_mixed': att,
                    'rel_emb':rel_emb,
                    'b_emb': b_emb
                    }
            return x, debug_dict

        else:
            return x
    
