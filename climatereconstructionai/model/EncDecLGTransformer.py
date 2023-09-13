import torch
import torch.nn as nn
import torch.functional as F

import climatereconstructionai.model.transformer_helpers as helpers
import climatereconstructionai.model.transformer_model as tm
from .. import config as cfg
from ..utils import grid_utils as gu

class nh_Block_self(nn.Module):
    def __init__(self, nh, model_dim, ff_dim, PE=None, out_dim=1, input_dim=1, dropout=0, n_heads=4) -> None: 
        super().__init__()
        
        self.nn_layer = helpers.nn_layer(nh, cart=True, both_dims=True)

        self.PE = PE

        self.nh = nh
        self.md = model_dim
        self.n_heads = n_heads

        self.local_att = helpers.MultiHeadAttentionBlock(
            model_dim, model_dim, n_heads, logit_scale=True, qkv_proj=True
            )
        
        self.mlp_layer_nh = nn.Sequential(
            nn.Linear(model_dim, ff_dim, bias=False),
            nn.Dropout(dropout),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            nn.Linear(ff_dim, 1, bias=False),
            nn.LeakyReLU(inplace=True, negative_slope=0.2)
        )

        self.mlp_layer_unfold = nn.Sequential(
            nn.Linear(model_dim*nh, model_dim*nh, bias=False),
            nn.Dropout(dropout),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            nn.Linear(model_dim*nh, out_dim, bias=False),
            nn.LeakyReLU(inplace=True, negative_slope=0.2)
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.norm3 = nn.LayerNorm(model_dim*nh)

    def forward(self, x, coords_rel, return_debug=False):
                
        x = self.norm1(x)
        
        #get nearest neighbours
        x, indices , cs = self.nn_layer(x, coords_rel)

        b, t, nh, e = x.shape
        x = x.reshape(b*t,nh,e)

        batched = cs.shape[0] == b
        
        rel_p_bias = self.PE(cs, batched=batched)

        if batched:
            rel_p_bias = rel_p_bias.reshape(b*t,self.nh, self.nh, self.n_heads)
        else:
            rel_p_bias = rel_p_bias.repeat(b,1,1,1)

        q = k = v = x

        att_out, att = self.local_att(q, k, v, rel_pos_bias=rel_p_bias, return_debug=True)

        x = x + self.dropout1(att_out)
        x = self.norm2(x)
    
        x = x + self.dropout2(self.mlp_layer_nh(x))

        x = x.view(b,t,self.nh,self.md)
        x = x.view(b,t,self.nh*self.md)

        x = self.norm3(x)
        x = self.dropout3(self.mlp_layer_unfold(x))

        if return_debug:
            return x, att, rel_p_bias, cs
        else:
            return x


class voting_layer(nn.Module):
    def __init__(self, nh, model_dim, ff_dim, PE=None, dropout=0, n_heads=4, reduction=0.1) -> None: 
        super().__init__()

        self.nh_Block = nh_Block_self(nh, model_dim, ff_dim, PE=PE, out_dim=model_dim, input_dim=model_dim, dropout=dropout, n_heads=n_heads)
    
    def forward(self, att, indices):
        pass


class nh_Block_mix(nn.Module):
    def __init__(self, nh, model_dim, ff_dim, PE=None, input_dim=1, dropout=0, n_heads=4) -> None: 
        super().__init__()
        
        self.nn_layer = helpers.nn_layer(nh, cart=True, both_dims=False)

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

        #get nearest neighbours
        x, _, cs = self.nn_layer(x, coords_rel)

        b, t, nh, e = x.shape

        v = x.reshape(b*t, nh,e)
        v = self.v_proj(v)

        batched = cs.shape[0] == b
        pe = self.pe_dropout(self.PE(cs, batched=batched))

        x = x + pe
        x = self.norm1(x)
        x = x.view(b*t,self.nh,self.md)

        q = self.q_proj(x)
        k = self.k_proj(x)

        att_out = self.local_att(q, k, v, return_debug=return_debug)

        if return_debug:
            att = att_out[1]
            att_out = att_out[0]

        x = x + self.dropout1(att_out)
        x = self.norm2(x)
    
        x = x + self.dropout2(self.mlp_layer_nh(x))

        x = x.view(b,t,self.nh,self.md)
        x = x.view(b,t,self.nh*self.md)

        x = self.norm3(x)
        x = self.dropout3(self.mlp_layer_unfold(x))

        if return_debug:
            return x, [att], [pe], cs
        else:
            return x


class trans_Block(nn.Module):
    def __init__(self, model_dim, ff_dim, out_dim=1, input_dim=1, dropout=0, n_heads=4) -> None: 
        super().__init__()

        self.local_att = helpers.MultiHeadAttentionBlock(
            model_dim, model_dim, n_heads, logit_scale=True, qkv_proj=True
            )
        
        self.mlp_layer = nn.Sequential(
            nn.Linear(model_dim, ff_dim, bias=False),
            nn.Dropout(dropout),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            nn.Linear(ff_dim, out_dim, bias=False),
            nn.LeakyReLU(inplace=True, negative_slope=0.2)
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

class Block(nn.Module):
    def __init__(self, model_dim, ff_dim, g_RPE, l_RPE, output_dim=None, nh=4, n_heads=10, dropout=0.1, global_att=True):
        super().__init__()

        self.g_RPE = g_RPE
        
        if not global_att:
            self.trans_Block = None
        else:
            self.trans_Block = trans_Block(model_dim, ff_dim, out_dim=model_dim, input_dim=model_dim, n_heads=n_heads)
        
        self.nh_net = nh_Block_self(nh, model_dim, ff_dim, l_RPE, out_dim=output_dim, dropout=dropout, n_heads=n_heads)

               
    def forward(self, x, rel_coords, return_debug=False):
        b,t,e = x.shape

        if not self.trans_Block is None:
            batched = rel_coords.shape[0] == b
            rp_bias = self.g_RPE(rel_coords, batched=batched)
        else:
            rp_bias = None
        atts = []
        rp_biass = [rp_bias]

        if return_debug:

            if not self.trans_Block is None:
                x, att = self.trans_Block(x, rp_bias, return_debug)
                atts.append(att)

            x, att_nh, rp_bias_nh, _ = self.nh_net(x, rel_coords, return_debug)
            atts.append(att_nh)

            rp_biass.append(rp_bias_nh)

            return x, atts, rp_biass
        
        else:

            if not self.trans_Block is None:
                x = self.trans_Block(x, rp_bias)

            x = self.nh_net(x, rel_coords)

            return x


class Encoder(nn.Module):

    def __init__(self, n_layers, model_dim, nh, ff_dim, local_PE, n_heads=10, dropout=0.1, global_att=True):
        super().__init__()

        model_dim_nh = model_dim // nh
        ff_dim_nh = ff_dim // nh

        if global_att:
            g_RPE = helpers.RelativePositionEmbedder_mlp(n_heads, ff_dim, transform=True)
        else:
            g_RPE = None

        self.nh_input_net = nh_Block_mix(nh, model_dim_nh, ff_dim_nh, local_PE, dropout=dropout, n_heads=n_heads)

        self.layers = nn.ModuleList()

        for n in range(n_layers):
            output_dim=model_dim if n == n_layers-1 else model_dim
            self.layers.append(Block(model_dim,
                                        ff_dim,
                                        g_RPE,
                                        helpers.RelativePositionEmbedder_mlp(n_heads, ff_dim_nh, transform=True),
                                        output_dim=output_dim,
                                        n_heads=n_heads,
                                        dropout=dropout,
                                        global_att=global_att))

    def forward(self, x, rel_coords, return_debug=False):
            
            atts = []
            rel_embs = []

            x = self.nh_input_net(x, rel_coords, return_debug=return_debug)

            if return_debug:
                atts += x[1]
                rel_embs += x[2]
                x = x[0]

            for layer in self.layers:
                x = layer(x, rel_coords, return_debug)
                if return_debug:
                    atts += x[1]
                    rel_embs += x[2]
                    x = x[0]

            if return_debug:
                return x, atts, rel_embs
            else:
                return x


class Decoder(nn.Module):

    def __init__(self, dec_layers, cross_layers, model_dim, nh, ff_dim, n_heads=10, dropout=0.1, global_RPE=None, global_att=False):
        super().__init__()
    
        self.global_RPE = global_RPE

        self.cross_att = helpers.MultiHeadAttentionBlock(
            model_dim, model_dim, n_heads, logit_scale=True, qkv_proj=True
            )
        
        self.layers = nn.ModuleList()
        for _ in range(dec_layers):
            self.layers.append(Block(model_dim,
                                        ff_dim,
                                        global_RPE,
                                        helpers.RelativePositionEmbedder_mlp(n_heads, ff_dim, transform=True),
                                        output_dim=model_dim,
                                        nh=nh,
                                        n_heads=n_heads,
                                        dropout=dropout,
                                        global_att=global_att))
        
        self.layers_cross = nn.ModuleList()
        for k in range(cross_layers):
            output_dim = 1 if k == cross_layers - 1 else model_dim
            self.layers_cross.append(Block(model_dim,
                                            ff_dim,
                                            global_RPE,
                                            helpers.RelativePositionEmbedder_mlp(n_heads, ff_dim, transform=True),
                                            output_dim=output_dim,
                                            nh=nh,
                                            n_heads=n_heads,
                                            dropout=dropout,
                                            global_att=global_att))
        
        
    def forward(self, x, x_enc, rel_coords_target_source, rel_coords_target, return_debug=False):
        
        atts = []
        rel_embs = []

        for layer in self.layers:
            x = layer(x, rel_coords_target, return_debug)

            if return_debug:
                atts += x[1]
                rel_embs += x[2]
                x = x[0]
        
        batched = rel_coords_target_source.shape[0] == x.shape[0]
        cross_pos_bias = self.global_RPE(rel_coords_target_source, batched=batched)

        if cross_pos_bias.dim() == x.dim():
            cross_pos_bias = cross_pos_bias.unsqueeze(dim=0).repeat(x.shape[0],1,1,1)

        x = self.cross_att(x, x_enc, x_enc, cross_pos_bias, return_debug)

        if return_debug:
            atts.append(x[1])
            rel_embs.append(x[2])
            x = x[0]

        for layer in self.layers_cross:
            x = layer(x, rel_coords_target, return_debug)

            if return_debug:
                atts += x[1]
                rel_embs += x[2]
                x = x[0]

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
        nh = model_settings['feat_nh']
        ff_dim = model_settings['ff_dim']
        self.train_interpolation = model_settings['train_interpolation']

        model_dim_nh = model_dim // nh
        ff_dim_nh = ff_dim // nh

        self.abs_pos_emb_local = helpers.RelativePositionEmbedder_mlp(model_dim_nh, ff_dim_nh, transform=True)

        self.nh_input_net = nh_Block_mix(nh, model_dim_nh, ff_dim_nh, self.abs_pos_emb_local, input_dim=1, dropout=dropout, n_heads=n_heads)

        self.interpolation =  nn.Sequential(
            nn.Linear(model_dim, ff_dim, bias=True),
            nn.Dropout(dropout),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            nn.Linear(ff_dim, 1, bias=True)
            #nn.LeakyReLU(inplace=True, negative_slope=0.2)
        )

        if not self.train_interpolation:
            self.cross_pos_emb_bias = helpers.RelativePositionEmbedder_mlp(n_heads, ff_dim, transform=True)
            
            self.Encoder = Encoder(
                model_settings['encoder']['n_layers'],
                model_settings['model_dim'],
                model_settings['feat_nh'],
                model_settings['ff_dim'],
                self.abs_pos_emb_local,
                n_heads=n_heads,
                dropout=dropout,
                global_att=model_settings['encoder']['global_att']
            )

            self.Decoder = Decoder(
                model_settings['decoder']['n_layers'],
                model_settings['decoder']['n_layers'],
                model_settings['model_dim'],
                model_settings['feat_nh'],
                model_settings['ff_dim'],
                global_RPE=self.cross_pos_emb_bias,
                n_heads=n_heads,
                dropout=dropout,
                global_att=model_settings['decoder']['global_att']
            )

        self.mlp_out = nn.Sequential(nn.Linear(model_dim, 1),nn.LeakyReLU(negative_slope=0.2,inplace=True))
        

    def forward(self, x, coord_dict, return_debug=False):
        
        rel_coords_source = coord_dict['rel']['source']
        rel_coords_target = coord_dict['rel']['target']
        rel_coords_target_source = coord_dict['rel']['target-source']

        atts = []
        rel_embs = []

        if not self.train_interpolation:
            x_enc = self.Encoder(x, rel_coords_source, return_debug=return_debug)

            if return_debug:
                atts += x_enc[1]
                rel_embs += x_enc[2]
                x_enc = x_enc[0]

        x_inp = self.nh_input_net(x, rel_coords_target_source, return_debug=return_debug)

        if return_debug:
            atts.append(x_inp[1])
            rel_embs.append(x_inp[2])
            x_inp = x_inp[0]

        out_proj = self.interpolation(x_inp)

        if not self.train_interpolation:
            x = self.Decoder(x_inp, x_enc, rel_coords_target_source, rel_coords_target, return_debug=return_debug)

            if return_debug:
                atts += x[1]
                rel_embs += x[2]
                x = x[0]
            
            x = x + out_proj 
        
        else:
            x = out_proj

        if return_debug:

            debug_dict = {
                          'atts':atts,
                          'rel_emb': rel_embs,
                        }
            
            return x, debug_dict
        else:
            return x
    
