import torch
import torch.nn as nn
import torch.functional as F

#import deepcopy

import climatereconstructionai.model.transformer_helpers as helpers
from .. import config as cfg
from ..utils import grid_utils as gu

radius_earth = 6371

class Input_Net(nn.Module):
    def __init__(self, nh=10, input_dropout=0):
        super().__init__()

        self.input_dropout = nn.Dropout(input_dropout)
        self.interpolator = helpers.interpolator_iwd(nh)


    def forward(self, x, coord_dict):
        
        x = self.input_dropout(x)

        x_inter = self.interpolator(x, coord_dict['rel']['target-source'])

        return x_inter


class feature_net_conv(nn.Module):
    def __init__(self, in_feat, out_feat, dropout=0):
        super().__init__()

        self.norm_in = nn.BatchNorm2d(out_feat)
        
        self.dropout = nn.Dropout(dropout) if dropout >0 else nn.Identity()


        self.conv = nn.Sequential(nn.Conv2d(in_feat, out_feat, kernel_size=1, bias=False),
                                   self.norm_in,
                                   nn.LeakyReLU(negative_slope=0.2))

        
    def forward(self, x, coords_abs):
        
        b, n, nh, f = x.shape


        c1 = coords_abs[0].view(1,n,nh,1)
        c2 = coords_abs[0].view(1,n,nh,1)

        x_v = x[:,:,[0]].repeat(1,1,nh,1)
        x_c1 = c1[:,:,[0]].repeat(b,1,nh,1)
        x_c2 = c2[:,:,[0]].repeat(b,1,nh,1)

        x_vd = x - x_v
        x_cd1 = c1 - x_c1
        x_cd2 = c2 - x_c2


        x = torch.cat((x_vd,x_cd1,x_cd2,x_v),dim=-1)

        x = x.permute(0,-1,1,2).contiguous()

        x = self.conv(x)

        x = x.max(dim=-1, keepdim=True)[0]

        return x.transpose(-1,1)


class nn_Block(nn.Module):
    def __init__(self, nh, RPE, output_dim=1, dropout=0, n_heads=4, n_feat_net=1, n_msa=2) -> None: 
        super().__init__()
        
        self.nn_layer = helpers.nn_layer(nh)

        self.feat_net_blocks = nn.ModuleList()
        for k in range(n_feat_net):
            in_feat = 4 if k==0 else nh**(k+1)
            self.feat_net_blocks.append(feature_net_conv(in_feat, nh**(k+2), dropout=0))

        self.MSA_Blocks = nn.ModuleList()
        for k in range(n_msa):
            if k==n_msa-1:
                self.MSA_Blocks.append(MSABlock(nh**(n_feat_net+1), nh**(n_feat_net+1), nh**(n_feat_net+1), RPE, output_dim=output_dim, dropout=dropout, n_heads=n_heads))
            else:
                self.MSA_Blocks.append(MSABlock(nh**(n_feat_net+1), nh**(n_feat_net+1), nh**(n_feat_net+1), RPE, dropout=dropout, n_heads=n_heads))


    def forward(self, x, coords_rel_nn, coords_abs_nn, coords_rel_grid, return_debug=False):
        b, n, e = x.shape
        x, indices,_,_ = self.nn_layer(x, coords_rel_nn[0], coords_rel_nn[1])

        d_lon = coords_abs_nn[0].view(-1)[indices].unsqueeze(dim=-1)
        d_lat = coords_abs_nn[1].view(-1)[indices].unsqueeze(dim=-1)

        for feat_net_block in self.feat_net_blocks:
            x = feat_net_block(x, [d_lon, d_lat])

        #res here possible
        x  = x.squeeze()

        for MSA_Block in self.MSA_Blocks:
            if return_debug:
                x, att, rel_emb = MSA_Block(x, [coords_rel_grid[0], coords_rel_grid[1]], return_debug=return_debug)     
            else:
                x = MSA_Block(x, [coords_rel_grid[0], coords_rel_grid[1]])

        if return_debug:
            return x, att, rel_emb
        else:
            return x
        
        
    

class MSABlock(nn.Module):
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
            nn.Linear(ff_dim, output_dim)
        )

        if input_dim>1:
            self.norm1 = nn.LayerNorm(input_dim)
            self.dropout1 = nn.Dropout(dropout)
        else:
            self.norm1 = nn.Identity()
            self.dropout1 = nn.Identity()

        
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
        nh_s = model_settings['local']['nh_source']
        n_t = model_settings['local']['nh_target']
        n_tb = model_settings['grid']['n_layers']
        n_local = model_settings['local']['n_layers']

        pass_local_features = model_settings['pass_local_features']
       
        if model_settings['embeddings']['rel']['use_mlp']:
            emb_class = helpers.RelativePositionEmbedder_mlp
        else:
            emb_class = helpers.RelativePositionEmbedder_par

        model_settings['embeddings']['rel']['emb_dim'] = n_heads
        RPE_phys = emb_class(model_settings['embeddings']['rel'], device=cfg.device)

        self.input_net = Input_Net(
           nh=nh_s, input_dropout=model_settings['input_dropout'])
        
        if pass_local_features:
            out_source_blocks = nh_s**(n_local+1)
        else:
            out_source_blocks = 1

        self.nn_Block_source = nn_Block(nh_s, RPE_phys, output_dim=out_source_blocks, dropout=dropout, n_heads=n_heads, n_feat_net=1, n_msa=2)
        
        self.nn_Blocks_target = nn.ModuleList()

        for _ in range(n_tb):
            self.nn_Blocks_target.append(nn_Block(n_t, RPE_phys, dropout=dropout, n_heads=n_heads, n_feat_net=1, n_msa=2))
        
        self.mlp_out = nn.Sequential(nn.Linear(out_source_blocks,1, bias=True), nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x, coord_dict, return_debug=False):
        b,s,e = x.shape
               
        x_interpolated = self.input_net(x, coord_dict)

        atts = []
        rel_embs = []
        if return_debug:
            x, att, rel_emb = self.nn_Block_source(x, coord_dict['rel']['target-source'], coord_dict['abs']['source'], coord_dict['rel']['target'],return_debug)
            atts.append(att)
            rel_embs.append(rel_emb)
            #x = x+out
        else:
            x = self.nn_Block_source(x, coord_dict['rel']['target-source'], coord_dict['abs']['source'], coord_dict['rel']['target'])

        #x = x + x_interpolated

        for Block_target in self.nn_Blocks_target:
            if return_debug:
                out, att, rel_emb = Block_target(x, coord_dict['rel']['target'], coord_dict['abs']['target'], coord_dict['rel']['target'],return_debug)
                atts.append(att)
                rel_embs.append(rel_emb)
                x = x+out
            else:
                x = x + Block_target(x, coord_dict['rel']['target'], coord_dict['abs']['target'], coord_dict['rel']['target'])

        x = self.mlp_out(x)


        if return_debug:
            debug_dict = {'atts': atts, 'rel_embs':rel_embs, 'x_inter':x_interpolated}
            #debug_dict = {'x_inter':x_interpolated}
            return x, debug_dict
        else:
            return x
    
