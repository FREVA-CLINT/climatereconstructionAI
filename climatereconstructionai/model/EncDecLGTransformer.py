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


#def gather_rel(mat, idx, device='cpu'):
#    b,s,t = mat.shape

#    shift = (torch.arange(idx.shape[0],device=device)*s).view(b,1,1)

 #   idx_shifted = idx + shift

 #   idx_shifted = idx_shifted.transpose(-2,-1).reshape(b*t,s)
 #   mat_bs = mat.view(b*s,e)
 #   x_bs = mat_bs[c_ix_shifted].view(b,t,s)

class nh_Block_g(nn.Module):
    def __init__(self, nh, model_dim, ff_dim, PE=None, out_dim=1, input_dim=1, dropout=0, n_heads=4) -> None: 
        super().__init__()
        
        self.nn_layer = helpers.nn_layer(nh, cart=True, both_dims=True)

        self.PE = PE

        self.nh = nh
        self.md = model_dim

        self.local_att = helpers.MultiHeadAttentionBlock(
            model_dim, model_dim, n_heads, logit_scale=True, qkv_proj=True
            )
        
        self.mlp_layer_nh = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            nn.Dropout(dropout),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            nn.Linear(ff_dim, 1),
            nn.LeakyReLU(inplace=True, negative_slope=0.2)
        )

        self.mlp_layer_unfold = nn.Sequential(
            nn.Linear(model_dim*nh, model_dim*nh),
            nn.Dropout(dropout),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            nn.Linear(model_dim*nh, out_dim),
            nn.LeakyReLU(inplace=True, negative_slope=0.2)
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.norm3 = nn.LayerNorm(model_dim*nh)

    def forward(self, x, coords_rel, return_debug=False):
        b,t,e = x.shape
        
        x = self.norm1(x)
        
        #get nearest neighbours
        x, indices = self.nn_layer(x, coords_rel[0], coords_rel[1])

        c1 = torch.gather(coords_rel[0], dim=1, index=indices).transpose(0,1)
        c2 = torch.gather(coords_rel[1], dim=1, index=indices).transpose(0,1)

        c1 = c1.repeat(b,1)
        c2 = c2.repeat(b,1)

        x = x.view(b*t,self.nh,e)

        # absolute coordinates with respect to target coordinates
        rel_p_bias = self.PE(c1,c2)

        q = k = v = x

        if return_debug:
            att_out, att, _ = self.local_att(q, k, v, rel_pos_bias=rel_p_bias, return_debug=return_debug)
        else:
            att_out = self.local_att(q, k, v, rel_pos_bias=rel_p_bias)

        x = x + self.dropout1(att_out)
        x = self.norm2(x)
    
        x = x + self.dropout2(self.mlp_layer_nh(x))

        x = x.view(b,t,self.nh,self.md)
        x = x.view(b,t,self.nh*self.md)

        x = self.norm3(x)
        x = self.dropout3(self.mlp_layer_unfold(x))

        if return_debug:
            return x, att, rel_p_bias, [c1,c2]
        else:
            return x



class nh_Block(nn.Module):
    def __init__(self, nh, model_dim, ff_dim, PE=None, out_dim=1, input_dim=1, dropout=0, n_heads=4) -> None: 
        super().__init__()
        
        self.nn_layer = helpers.nn_layer(nh, cart=True)

        self.PE = PE

        self.nh = nh
        self.md = model_dim

        self.local_att = helpers.MultiHeadAttentionBlock(
            model_dim, model_dim, n_heads, logit_scale=True, qkv_proj=False
            )
        
        self.mlp_layer_nh = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            nn.Dropout(dropout),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            nn.Linear(ff_dim, 1),
            nn.LeakyReLU(inplace=True, negative_slope=0.2)
        )

        self.mlp_layer_unfold = nn.Sequential(
            nn.Linear(model_dim*nh, model_dim*nh),
            nn.Dropout(dropout),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            nn.Linear(model_dim*nh, model_dim*nh),
            nn.LeakyReLU(inplace=True, negative_slope=0.2)
        )

        self.pe_dropout = nn.Dropout(dropout) 
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.norm3 = nn.LayerNorm(model_dim*nh)

        self.q_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.v_proj = nn.Linear(input_dim, model_dim, bias=False)


    def forward(self, x, coords_rel, return_debug=False):
        e = x.shape[-1]
        b,s,t = coords_rel[0].shape

        #get nearest neighbours
        x, indices = self.nn_layer(x, coords_rel[0], coords_rel[1])

        c1 = torch.gather(coords_rel[0], dim=1, index=indices)
        c2 = torch.gather(coords_rel[1], dim=1, index=indices)

        v = x.view(b*t,self.nh,e)
        v = self.v_proj(v)

        # absolute coordinates with respect to target coordinates
        pe = self.pe_dropout(self.PE(c1,c2))

        x = x + pe.transpose(2,1)
        x = self.norm1(x)
        x = x.view(b*t,self.nh,self.md)

        q = self.q_proj(x)
        k = self.k_proj(x)

        if return_debug:
            att_out, att, _ = self.local_att(q, k, v, return_debug=return_debug)
        else:
            att_out = self.local_att(q, k, v)

        x = x + self.dropout1(att_out)
        x = self.norm2(x)
    
        x = x + self.dropout2(self.mlp_layer_nh(x))

        x = x.view(b,t,self.nh,self.md)
        x = x.view(b,t,self.nh*self.md)

        x = self.norm3(x)
        x = self.dropout3(self.mlp_layer_unfold(x))

        if return_debug:
            return x, att, pe, [c1,c2]
        else:
            return x


class trans_Block(nn.Module):
    def __init__(self, model_dim, ff_dim, out_dim=1, input_dim=1, dropout=0, n_heads=4) -> None: 
        super().__init__()

        self.local_att = helpers.MultiHeadAttentionBlock(
            model_dim, model_dim, n_heads, logit_scale=True, qkv_proj=True
            )
        
        self.mlp_layer = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            nn.Dropout(dropout),
            nn.LeakyReLU(inplace=True),
            nn.Linear(ff_dim, out_dim),
            nn.LeakyReLU(inplace=True)
        )

        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(model_dim)
        

        if out_dim != input_dim:
            self.compute_res = False
            self.dropout2 = nn.Identity()
            self.norm2 = nn.Identity()
        else:
            self.compute_res = True
            self.dropout2 = nn.Dropout(dropout)
            self.norm2 = nn.LayerNorm(model_dim)

    def forward(self, x, rp_bias, return_debug=False):
        e = x.shape[-1]

        x = self.norm1(x)
        q = k = v = x

        if return_debug:
            att_out, att, _ = self.local_att(q, k, v, rp_bias, return_debug=return_debug)
        else:
            att_out = self.local_att(q, k, v, rp_bias)

        x = x + self.dropout1(att_out)
        x = self.norm2(x)
        
        mlp_out = self.dropout2(self.mlp_layer(x))

        if self.compute_res:
            x = x + mlp_out
        else:
            x = mlp_out

        if return_debug:
            return x, att
        
        return x



class EncoderBlock(nn.Module):
    def __init__(self, model_dim, nh, ff_dim, output_dim=None, n_heads=10, dropout=0.1, logit_scale=True):
        super().__init__()

    def forward(self, x, rel_coords, return_debug=False):
        return x


class DecoderBlock(nn.Module):
    def __init__(self, model_dim, ff_dim, g_RPE, l_RPE, output_dim=None, nh=4, n_heads=10, dropout=0.1, logit_scale=True):
        super().__init__()

        self.g_RPE = g_RPE
        self.trans_Block = trans_Block(model_dim, ff_dim, out_dim=output_dim, input_dim=model_dim, n_heads=n_heads)
        
 #       self.nh_net = nh_Block_g(nh, model_dim, ff_dim, l_RPE, out_dim=output_dim, dropout=dropout, n_heads=n_heads)

               
    def forward(self, x, rel_coords, return_debug=False):
        b,t,e = x.shape

        rp_bias = self.g_RPE(rel_coords[0], rel_coords[1])

        if return_debug:
            x, att = self.trans_Block(x, rp_bias, return_debug)
            return x, att, rp_bias
        else:
            x = self.trans_Block(x, rp_bias)

    #        x = self.nh_net(x, rel_coords)
            return x
            


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
                output, att, rel_emb, cs = layer(x, rel_coords, pos, return_debug=return_debug)
                atts.append(att)

        if return_debug:
            return output, atts, rel_emb, cs
        else:
            return output


class Decoder(nn.Module):

    def __init__(self, n_layers, model_dim, ff_dim, nh_mix=4, n_heads=10, dropout=0.1, logit_scale=True, train_interpolation=False):
        super().__init__()
        
        self.train_interpolation = train_interpolation

        model_dim_nh = model_dim // nh_mix
        ff_dim_nh = ff_dim // nh_mix

        PE = helpers.RelativePositionEmbedder_mlp(model_dim_nh, ff_dim, transform=False)
        self.nh_input_net = nh_Block(nh_mix, model_dim_nh, ff_dim_nh, PE, dropout=dropout, n_heads=n_heads)

        #nh output net for interpolation
        self.interpolation =  nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            nn.Dropout(dropout),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            nn.Linear(ff_dim, 1),
            nn.LeakyReLU(inplace=True, negative_slope=0.2)
        )

        g_PE = helpers.RelativePositionEmbedder_mlp(n_heads, ff_dim, transform=True)

    #    l_PE = helpers.RelativePositionEmbedder_mlp(n_heads, ff_dim_nh, transform=False)
        l_PE = None

        self.layers = nn.ModuleList()
        for n in range(n_layers):
            output_dim=1 if n == n_layers-1 else model_dim
            self.layers.append(DecoderBlock(model_dim,
                                                ff_dim,
                                                g_PE,
                                                l_PE,
                                                output_dim=output_dim,
                                                n_heads=n_heads,
                                                dropout=dropout,
                                                logit_scale=logit_scale))
        
        
    def forward(self, x, rel_coords_target_source, rel_coords_target, return_debug=False):
        
        atts = []
        rel_embs = []

        if return_debug:
            x, att, rel_emb, _ = self.nh_input_net(x, rel_coords_target_source, return_debug=return_debug)
            atts.append(att)
            rel_embs.append(rel_emb)
        else:
            x = self.nh_input_net(x, rel_coords_target_source)

        out_proj = self.interpolation(x)

        for layer in self.layers:
            if return_debug:
                x, att, rel_emb = layer(x, rel_coords_target, return_debug)
                atts.append(att)
                rel_embs.append(rel_emb)
            else:
                x = layer(x, rel_coords_target)

        if self.train_interpolation:
            x = out_proj
        else:
            x = x + out_proj 

        if return_debug:
            return x, atts, rel_embs
        else:
            return x
        

class SpatialTransNet(tm.transformer_model):
    def __init__(self, model_settings) -> None: 
        super().__init__(model_settings)
        
        model_settings = self.model_settings
        dropout = model_settings['dropout']
        model_dim = model_settings['model_dim']
        n_heads = model_settings['n_heads']
        logit_scale = model_settings['logit_scale']

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
            nh_mix=model_settings['feat_nh'],
            n_heads=n_heads,
            dropout=dropout,
            logit_scale=logit_scale,
            train_interpolation=model_settings['train_interpolation']
        )

        self.mlp_out = nn.Sequential(nn.Linear(model_dim, 1),nn.LeakyReLU(negative_slope=0.2,inplace=True))
        

    def forward(self, x, coord_dict, return_debug=False):
        

        rel_coords_source = coord_dict['rel']['source']
        rel_coords_target = coord_dict['rel']['target']
        rel_coords_target_source = coord_dict['rel']['target-source']
        coords_source = coord_dict['abs']['source']
        coords_target = coord_dict['abs']['target']
   
        out = self.Decoder(x, rel_coords_target_source, rel_coords_target, return_debug=return_debug)

        if return_debug:
            atts = out[1]
            rel_emb = out[2]
            out = out[0]


        if return_debug:

            debug_dict = {
                          'atts':atts,
                          'rel_emb': rel_emb,
                        }
            
            return out, debug_dict
        else:
            return out
    
