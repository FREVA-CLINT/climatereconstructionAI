import torch
import torch.nn as nn
import torch.functional as F

import climatereconstructionai.model.transformer_helpers as helpers
import climatereconstructionai.model.transformer_model as tm
from .. import config as cfg
from ..utils import grid_utils as gu


radius_earth = 6371

class Input_Net(nn.Module):
    def __init__(self, emb_settings, model_dim, emb_dim_target, dropout=0):
        super().__init__()
        if emb_settings['polar']:
            emb_class = helpers.RelativePositionEmbedder_polar
        else:
            emb_class = helpers.RelativePositionEmbedder_cart
        self.APE_phys = emb_class(emb_settings)

        self.source_inp_layer = nn.Linear(emb_settings['emb_dim'], model_dim)
        self.target_inp_layer = nn.Linear(emb_settings['emb_dim'], emb_dim_target)

        self.t_proj = helpers.nearest_proj_layer(5)

    def forward(self, x, coord_dict):
        
        b,t,e = x.shape

        ape_emb_s ,b_idx_as =  self.APE_phys(coord_dict['abs']['source'][0], coord_dict['abs']['source'][1])
        xs = x + ape_emb_s

        xt = self.t_proj(x, torch.sqrt(coord_dict['rel']['target-source'][0]**2+coord_dict['rel']['target-source'][1]**2))
        ape_emb_t, b_idx_at =  self.APE_phys(coord_dict['abs']['target'][0], coord_dict['abs']['target'][1])
        xt = xt + ape_emb_t
        #xt = torch.zeros((b,len(coord_dict['abs']['target'][0]),1), device='cpu') + ape_emb_t.permute(1,2,0)

        xs = self.source_inp_layer(xs)
        xt = self.target_inp_layer(xt)

        return xs, xt, b_idx_as, b_idx_at

class feature_net2(nn.Module):
    def __init__(self, in_feat, out_feat, dropout=0):
        super().__init__()
      
        self.dropout = nn.Dropout(dropout) if dropout >0 else nn.Identity()

        self.conv = nn.Sequential(nn.Conv2d(in_feat, out_feat, kernel_size=1, bias=True),
                                   nn.LeakyReLU(negative_slope=0.2))
        

    def forward(self, x):
        
        b, n, nh, f = x.shape

        x_v = x[:,:,[0]].repeat(1,1,nh,1)
        x = torch.cat(((x - x_v), x),dim=-1)
         
        x = x.permute(0,-1,1,2).contiguous()

        x = self.conv(x)

        x = x.max(dim=-1, keepdim=True)[0]

        return x.transpose(-1,1)
    
class feature_net(nn.Module):
    def __init__(self, in_feat, out_feat, dropout=0):
        super().__init__()
      
        self.dropout = nn.Dropout(dropout) if dropout >0 else nn.Identity()

        self.conv = nn.Sequential(nn.Conv2d(in_feat, out_feat, kernel_size=1, bias=True),
                                   nn.LeakyReLU(negative_slope=0.2))
        

    def forward(self, x):
        
        b, n, nh, f = x.shape

        x_v = x[:,:,[0]].repeat(1,1,nh,1)
        x = torch.cat(((x - x_v), x_v),dim=-1)
         
        x = x.permute(0,-1,1,2).contiguous()

        x = self.conv(x)

        x = x.max(dim=-1, keepdim=True)[0]

        return x.transpose(-1,1)


class nn_feat_Block(nn.Module):
    def __init__(self, nh, model_dim, rpe_emb=None, dropout=0) -> None: 
        super().__init__()
        
        self.nn_layer = helpers.nn_layer(nh)
        self.feat_net_block = feature_net(2, model_dim, dropout=dropout)
        self.rpe_emb=rpe_emb

    def forward(self, x, coords_rel_nn):
        b, n, e = x.shape
        x, indices,_,_ = self.nn_layer(x, coords_rel_nn[0], coords_rel_nn[1])
        x = self.feat_net_block(x)
        x = x.squeeze()
        return x

class EncoderBlock(nn.Module):
    def __init__(self, model_dim, feat_nh, ff_dim, output_dim=None, n_heads=10, dropout=0.1, logit_scale=True):
        super().__init__()

        if output_dim is None:
            output_dim = model_dim
       
        self.input_dim = model_dim

        self.feature_net = nn_feat_Block(feat_nh, model_dim)

        self.self_att_layer = helpers.MultiHeadAttentionBlock(
            model_dim, model_dim, n_heads, logit_scale=logit_scale, qkv_proj=True
            )
                
        self.mlp_layer = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            nn.Dropout(dropout),
            nn.LeakyReLU(inplace=True),
            nn.Linear(ff_dim, output_dim),
            nn.LeakyReLU(inplace=True)
        )
       
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(output_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
            

    def forward(self, x, rel_coords, pos=None, return_debug=False):
        
        x = self.feature_net(x, rel_coords)
        x = self.norm1(x)

        if pos is not None:
            x = x+pos
        q = k = x

        if return_debug:
            att_out, att, rel_emb = self.self_att_layer(q, k, x, return_debug=True)
        else:
            att_out = self.self_att_layer(q, k, x)

        x = x + self.dropout1(att_out)
        x = self.norm2(x)
    
        x = x + self.dropout2(self.mlp_layer(x))

        if return_debug:
            return x, att, rel_emb
        else:
            return x


class DecoderBlock(nn.Module):
    def __init__(self, model_dim, ff_dim, output_dim=None, nh=-1, n_heads=10, dropout=0.1, logit_scale=True):
        super().__init__()

        if output_dim is None:
            output_dim = model_dim

        #self.nn_layer = helpers.nn_layer(nh)
        
        self.input_dim = model_dim
       
        self.self_att_layer = helpers.MultiHeadAttentionBlock(
            model_dim, model_dim, n_heads, logit_scale=logit_scale, qkv_proj=True
            )

        self.cross_att_layer = helpers.MultiHeadAttentionBlock(
            model_dim, model_dim, n_heads, logit_scale=logit_scale, qkv_proj=True
            )
        
        self.mlp_layer = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            nn.Dropout(dropout),
            nn.LeakyReLU(inplace=True),
            nn.Linear(ff_dim, output_dim),
            nn.LeakyReLU(inplace=True)
        )
       
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.norm_target = nn.LayerNorm(model_dim)
        self.norm3 = nn.LayerNorm(output_dim) if output_dim>1 else nn.Identity()

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
            

    def forward(self, target, x, pos_source, return_debug=False):       
        
        target = self.norm1(target)

        q = k = v = target

        if return_debug:
            att_out, att, rel_emb = self.self_att_layer(q, k, v, return_debug=return_debug)
        else:
            att_out = self.self_att_layer(q, k, v)

        target = target + self.dropout1(att_out)

        target2 = self.norm2(target)

        q = target2
        k = x + self.norm_target(pos_source)
        v = x

        if return_debug:
            att_out, att, rel_emb = self.cross_att_layer(q, k, v, return_debug=True)
        else:
            att_out = self.cross_att_layer(q, k, v)

        target = target + self.dropout1(att_out)
        target = self.norm3(target)

        target = target + self.dropout3(self.mlp_layer(target))

        if return_debug:
            return target, att, rel_emb
        else:
            return target


class Encoder(nn.Module):

    def __init__(self, n_layers, model_dim, feat_nh, ff_dim, n_heads=10, dropout=0.1, logit_scale=True):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(model_dim,
                                                feat_nh,
                                                ff_dim,
                                                n_heads=n_heads,
                                                dropout=dropout,
                                                logit_scale=logit_scale) for _ in range(n_layers)])

    def forward(self, x, rel_coords, pos=None, return_debug=False):
        
        atts = []
        for layer in self.layers:
            if not return_debug:
                output = layer(x, rel_coords, pos)
            else:
                output, att, _ = layer(x, rel_coords, pos, return_debug=return_debug)
                atts.append(att)

        if return_debug:
            return output, atts
        else:
            return output


class Decoder(nn.Module):

    def __init__(self, n_layers, model_dim, ff_dim, n_heads=10, dropout=0.1, logit_scale=True):
        super().__init__()
        self.layers = nn.ModuleList([DecoderBlock(model_dim,
                                                ff_dim,
                                                n_heads=n_heads,
                                                dropout=dropout,
                                                logit_scale=logit_scale) for _ in range(n_layers)])
        
    def forward(self, target, x, pos_source, return_debug=False):

        atts = []
        for layer in self.layers:
            if not return_debug:
                target = layer(target, x, pos_source)
            else:
                target, att, _ = layer(target, x, pos_source, return_debug=return_debug)
                atts.append(att)
            
        if return_debug:
            return target, atts
        else:
            return target
        

class SpatialTransNet(tm.transformer_model):
    def __init__(self, model_settings) -> None: 
        super().__init__(model_settings)
        
        model_settings = self.model_settings
        dropout = model_settings['dropout']
        model_dim = model_settings['model_dim']
        n_heads = model_settings['n_heads']
        logit_scale = model_settings['logit_scale']

        self.add_interpolation = model_settings['add_interpolation']
        self.pos_every = model_settings['encoder']['pos_every']


        if self.add_interpolation:
            self.interpolator = helpers.interpolator_iwd(model_settings['input_nh'], local_lambda=model_settings['local_lambda'])

        self.linear_pos_embedder = helpers.LinearPositionEmbedder_mlp(model_dim, model_settings['emb_hidden_dim'])

        self.Encoder = Encoder(
            model_settings['encoder']['n_layers'],
            model_settings['model_dim'],
            model_settings['feat_nh'],
            model_settings['ff_dim'],
            n_heads=n_heads,
            dropout=dropout,
            logit_scale=logit_scale
        )

        self.Decoder = Decoder(
            model_settings['decoder']['n_layers'],
            model_settings['model_dim'],
            model_settings['ff_dim'],
            n_heads=n_heads,
            dropout=dropout,
            logit_scale=logit_scale
        )

        self.mlp_out = nn.Sequential(nn.Linear(model_dim,1),nn.LeakyReLU(negative_slope=0.2,inplace=True))
        

    def forward(self, x, coord_dict, return_debug=False):

        rel_coords_source = coord_dict['rel']['source']
        rel_coords_target_source = coord_dict['rel']['target-source']
        coords_source = coord_dict['abs']['source']
        coords_target = coord_dict['abs']['target']

        pos_source = self.linear_pos_embedder(coords_source[0],coords_source[1])

        pos_target = self.linear_pos_embedder(coords_target[0],coords_target[1])

        x2 = x
        if self.pos_every:
            x2 = self.Encoder(x2, rel_coords_source, pos=pos_source, return_debug=return_debug)
        else:
            x2 = self.Encoder(x2 + pos_source, rel_coords_source, return_debug=return_debug)

        if return_debug:
            atts = x2[1]
            x2 = x2[0]

        pos_target = pos_target.unsqueeze(dim=0).repeat(x2.shape[0],1,1)
        out = self.Decoder(pos_target, x2, pos_source, return_debug=return_debug)

        if return_debug:
            atts = atts + out[1]
            out = out[0]
            out_dec = out

        out = self.mlp_out(out)

        if self.add_interpolation:
            x_inter = self.interpolator(x, rel_coords_target_source)
            out = out + x_inter
        else:
            x_inter = out

        if return_debug:

            debug_dict = {'x_inter': x_inter,
                          'atts':atts,
                          'pos_source': pos_source,
                          'pos_target': pos_target,
                          'out_dec': out_dec}
            
            return out, debug_dict
        else:
            return out
    
