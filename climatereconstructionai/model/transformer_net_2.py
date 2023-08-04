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

        self.interpolator = helpers.interpolator_iwd(nh)

        self.nn_layer = helpers.nn_layer(nh)

    def forward(self, x, coord_dict):
        
        x = self.input_dropout(x)

       
        x_nearest, indices_dist, indices_lon, indices_lat = self.nn_layer(x, coord_dict['rel']['target-source'][0], coord_dict['rel']['target-source'][1])

        x_inter = self.interpolator(x, coord_dict['rel']['target-source'])

        return x_nearest, x_inter, indices_dist, indices_lon, indices_lat


class feature_net(nn.Module):
    def __init__(self, in_feat, out_feat, k_size, dropout=0):
        super().__init__()

        self.in_feat=in_feat
        self.out_feat=out_feat
        self.norm_in = nn.LayerNorm(k_size)
        self.norm_out = nn.LayerNorm(out_feat)

        self.conv1 = nn.Sequential(nn.Conv1d(in_feat, out_feat, kernel_size=k_size, bias=False))
        
        self.dropout = nn.Dropout(dropout) if dropout >0 else nn.Identity()
        
    def forward(self, x):
        bs,nh,e = x.shape

        x = x - x.transpose(-1,-2)
        x = self.norm_in(x)
        x = x.reshape(bs*nh,1,nh)

        x = self.conv1(x)

        x = x.reshape(bs,nh,self.out_feat)
        x = self.norm_out(x)

        return x

class CRTransNetBlock(nn.Module):
    def __init__(self, input_dim, model_dim, ff_dim, RPE_phys, output_dim=None, n_heads=10, dropout=0.1, is_final_layer=False, logit_scale=True):
        super().__init__()

        self.is_final_layer = is_final_layer

        if output_dim is None:
            output_dim = input_dim
       
        self.input_dim = input_dim

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
            nn.Linear(ff_dim, output_dim),
            nn.ReLU(inplace=True)
        )

        if input_dim>1:
            self.norm1 = nn.LayerNorm(output_dim)
            self.dropout1 = nn.Dropout(dropout)
        else:
            self.norm2 = nn.Identity()
            self.dropout1 = nn.Identity()

        self.norm1 = nn.Identity()
        if output_dim>1:
            self.norm2 = nn.LayerNorm(output_dim)
            self.dropout2 = nn.Dropout(dropout)
        else:
            self.norm2 = nn.Identity()
            self.dropout2 = nn.Identity()

        self.diff = True if input_dim==output_dim else False
            

    def forward(self, x, rel_coords, xv=None, return_debug=False):
        
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        if return_debug:
            att_out, att, rel_emb = self.att_layer(q, k, v, rel_coords, return_debug=return_debug)
        else:
            att_out = self.att_layer(q, k, v, rel_coords, return_debug=return_debug)[0]

        x = x + self.dropout1(att_out)
        x = self.norm1(x)

        if self.diff:
            x = x + self.dropout2(self.mlp_layer(x))
        else:
            x = self.dropout2(self.mlp_layer(x))

        x = self.norm2(x)

        if return_debug:
            return x, att, rel_emb
        else:
            return x

class CRTransNet(nn.Module):
    def __init__(self, model_settings) -> None: 
        super().__init__()
        
        dropout = model_settings['dropout']
        n_heads = model_settings['grid']['n_heads']

        nh = model_settings['local']['nh']
        self.pass_source = model_settings['pass_source']

        if model_settings['embeddings']['rel']['use_mlp']:
            emb_class = helpers.RelativePositionEmbedder_mlp
        else:
            emb_class = helpers.RelativePositionEmbedder_par

        model_settings['embeddings']['rel']['emb_dim'] = n_heads
        self.RPE_phys = emb_class(model_settings['embeddings']['rel'], device=cfg.device)

        self.input_net = Input_Net(
           nh=nh, input_dropout=model_settings['input_dropout'])

        self.feat_net = feature_net(1, out_feat=model_settings['local']['model_dim'], k_size=nh)




        self.LocalBlocks = nn.ModuleList()
        for k in range(model_settings['local']['n_layers']):
            if k==model_settings['local']['n_layers']-1:
                output_dim = 1
            else:
                output_dim = model_settings['local']['model_dim']
            LocalBlock = CRTransNetBlock(
                input_dim=model_settings['local']['model_dim'],
                output_dim=output_dim,
                model_dim=model_settings['local']['model_dim'],
                RPE_phys=self.RPE_phys,
                ff_dim=model_settings['local']['ff_dim'],
                n_heads=n_heads,
                dropout=dropout,
                logit_scale=model_settings['logit_scale'])
            
            self.LocalBlocks.append(LocalBlock)

        if not model_settings['share_rel_emb']:
            self.RPE_phys_hr = emb_class(model_settings['embeddings']['rel'], device=cfg.device)
        else:
            self.RPE_phys_hr = self.RPE_phys

        if model_settings['grid']['n_layers']==0:
            self.out_mlp = nn.Sequential(nn.Linear(nh,1),nn.ReLU(inplace=True))
        else:
            self.out_mlp = nn.Identity()

        self.GridBlocks = nn.ModuleList()
        for k in range(model_settings['grid']['n_layers']):
            if k==model_settings['grid']['n_layers']-1:
                output_dim = 1
            else:
                output_dim = nh
            GridBlock = CRTransNetBlock(
                    input_dim=nh,
                    output_dim=output_dim,
                    model_dim=model_settings['grid']['model_dim'],
                    RPE_phys=self.RPE_phys_hr,
                    ff_dim=model_settings['grid']['ff_dim'],
                    n_heads=n_heads,
                    dropout=dropout,
                    logit_scale=model_settings['logit_scale'])
            self.GridBlocks.append(GridBlock)

        self.GridBlocks_out = nn.ModuleList()
        for _ in range(model_settings['grid_out']['n_layers']):
            GridBlock_out = CRTransNetBlock(
                    input_dim=1,
                    model_dim=model_settings['grid']['model_dim'],
                    RPE_phys=self.RPE_phys_hr,
                    ff_dim=model_settings['grid']['ff_dim'],
                    n_heads=n_heads,
                    dropout=dropout,
                    logit_scale=model_settings['logit_scale'])
            self.GridBlocks_out.append(GridBlock_out)       

    def forward(self, x, coord_dict, return_debug=False):
        b,s,e = x.shape
        
        x, x_inter, indices_dist, _ ,_ = self.input_net(x, coord_dict)


        d_lon = coord_dict['abs']['source'][0].view(-1)[indices_dist].unsqueeze(dim=-1)
        d_lat = coord_dict['abs']['source'][1].view(-1)[indices_dist].unsqueeze(dim=-1)

        d_lon = d_lon - d_lon.transpose(1,-1)
        d_lat = d_lat - d_lat.transpose(1,-1)

        b,t,nh,e = x.shape
        
        x = x.reshape(b*t,nh,e)
        d_lon = d_lon.repeat(b, 1, 1)
        d_lat = d_lat.repeat(b, 1, 1)

        x = self.feat_net(x)

        rel_embs = []
        atts = []

        for block in self.LocalBlocks:
            if return_debug:
                x, att, rel_emb =  block(x, [d_lon, d_lat], return_debug=return_debug)
                atts.append(att)
                rel_embs.append(rel_emb)
            else:
                x =  block(x, [d_lon, d_lat], return_debug=return_debug)

        x = x.reshape(b, t, nh)
        
        x = self.out_mlp(x)
        
        d_lon = coord_dict['rel']['target'][0]
        d_lat = coord_dict['rel']['target'][1]

        for block in self.GridBlocks:
            if return_debug:
                x, att, rel_emb =  block(x, [d_lon, d_lat], return_debug=return_debug)
                atts.append(att)
                rel_embs.append(rel_emb)
            else:
                x =  block(x, [d_lon, d_lat], return_debug=return_debug)
        
        x = x_inter + x

        for block in self.GridBlocks_out:
            if return_debug:
                x, att, rel_emb =  block(x, [d_lon, d_lat], return_debug=return_debug)
                atts.append(att)
                rel_embs.append(rel_emb)

            else:
                x =  block(x, [d_lon, d_lat], return_debug=return_debug)

        if return_debug:
            debug_dict = {'atts': atts, 'rel_embs':rel_embs, 'x_inter':x_inter}
            return x, debug_dict
        else:
            return x
    
