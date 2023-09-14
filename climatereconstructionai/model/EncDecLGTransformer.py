import torch
import torch.nn as nn
import torch.nn.functional as F

import climatereconstructionai.model.transformer_helpers as helpers
import climatereconstructionai.model.transformer_model as tm
from .. import config as cfg
from ..utils import grid_utils as gu

class nh_Block_self(nn.Module):
    def __init__(self, nh, model_dim, ff_dim, RPE=None, out_dim=1, input_dim=1, dropout=0, n_heads=4, transform_coords=True) -> None: 
        super().__init__()
        
        self.nn_layer = helpers.nn_layer(nh, both_dims=True)

        if RPE is None:
            self.RPE = helpers.RelativePositionEmbedder_mlp(n_heads, ff_dim, transform=transform_coords)
        else:
            self.RPE  = RPE

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
        x, indices , cs = self.nn_layer(x, coords_rel, coords_rel, skip_self=True)

        b, t, nh, e = x.shape
        x = x.reshape(b*t,nh,e)

        batched = cs.shape[0] == b
        
        rel_p_bias = self.RPE(cs, batched=batched)

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
            return x, att, indices, rel_p_bias, cs
        else:
            return x, att, indices


def fetch_data(x, indices):
    b, n, e = x.shape
    n_k = indices.shape[-1]
    indices = indices.view(b,n_k,1).repeat(1,1,e)
    x = torch.gather(x, dim=1, index=indices)
    return x

def fetch_coords(rel_coords, indices):
    b, n_c, n, _ = rel_coords.shape
    n_k = indices.shape[-1]
    indices = indices.view(b,1,n_k,1).repeat(1,2,1,1)
    rel_coords = torch.gather(rel_coords, dim=2, index=indices)
    return rel_coords


class voting_layer(nn.Module):
    def __init__(self, nh, n_heads=4, p_reduction=0.4) -> None: 

        super().__init__()
        self.p_reduction = p_reduction
        self.head_reduction = nn.Sequential(nn.Linear(n_heads, n_heads, bias=True),
                                            nn.Linear(n_heads, 1, bias=True))

        self.nh_voting = nn.Sequential(nn.Linear(nh*nh, nh*nh, bias=True),
                                       nn.Linear(nh*nh, nh, bias=True),
                                        nn.Linear(nh, 1, bias=True))
        
    
    def forward(self, x, att_nh, coords):
        
        b, n, e = x.shape
        bt, n_heads, nh, _ = att_nh.shape 
                
        att_nh = att_nh.reshape(b, n, n_heads, nh, nh)

        att_nh = att_nh.view(b, n, nh, nh, n_heads)

        att_vote = self.head_reduction(att_nh).view(b,n,nh*nh)

        att_vote = F.softmax(self.nh_voting(att_vote).squeeze(),dim=1)
        
        indices_vote = att_vote.sort(dim=-1, descending=True).indices

        n_keep = torch.tensor((1 - self.p_reduction)*indices_vote.shape[1]).long()
        indices_keep = indices_vote[:,:n_keep]

        coords= fetch_coords(coords.clone(), indices_keep)
        x = fetch_data(x, indices_keep)

        return x, coords

class reduction_layer(nn.Module):
    def __init__(self, nh, n_heads=4, p_reduction=0.4) -> None: 
        super().__init__()
        self.nn_layer = helpers.nn_layer(nh, both_dims=False)
        self.p_reduction = p_reduction

    def forward(self, x, coords):

        x_nh, _, rel_coords = self.nn_layer(x, coords, coords)  
        b, n, nh, e = x_nh.shape
        
        #local_m = x_nh.mean(dim=-2).abs()
        local_dist = (x_nh[:,:,1:,:] - x_nh[:,:,[0],:]).pow(2).sum(dim=-2).sum(dim=-1).sqrt()
        #local_s = x_nh.std(dim=-2)
        #local_rel = (local_s/local_m).sum(dim=-1)
       # local_rel = local_dist.sum(dim=-1)

        _, indices = local_dist.sort(dim=1, descending=True)

        n_keep = torch.tensor((1 - self.p_reduction)*indices.shape[1]).long()
        indices = indices[:,:n_keep]

        x = torch.gather(x, dim=1, index=indices.view(b,n_keep,1).repeat(1,1,e))
        coords = torch.gather(coords, dim=2, index=indices.view(b,1,n_keep,1).repeat(1,coords.shape[1],1,1))
        
        return x, coords, local_dist

            
            

class nh_Block_mix(nn.Module):
    def __init__(self, nh, model_dim, ff_dim, PE=None, input_dim=1, dropout=0, n_heads=4, transform_coords=True) -> None: 
        super().__init__()
        
        self.nn_layer = helpers.nn_layer(nh, both_dims=False)

        if PE is None:
            self.PE = helpers.RelativePositionEmbedder_mlp(model_dim, ff_dim, transform=transform_coords)
        else:
            self.PE  = PE

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


    def forward(self, x, coords_target, coords_source, return_debug=False):

        #get nearest neighbours
        x, _, coords_target_source_nh = self.nn_layer(x, coords_target, coords_source)

        b, t, nh, e = x.shape

        v = x.reshape(b*t, nh,e)
        v = self.v_proj(v)

        batched = coords_target_source_nh.shape[0] == b
        pe = self.pe_dropout(self.PE(coords_target_source_nh, batched=batched))

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
            return x, [att], [pe], coords_target_source_nh
        else:
            return x


class attention_Block(nn.Module):
    def __init__(self, model_dim, ff_dim, out_dim=1, input_dim=1, dropout=0, n_heads=4, rpe_PE=None, transform_coords=True) -> None: 
        super().__init__()

        if rpe_PE is None:
            self.g_RPE = helpers.RelativePositionEmbedder_mlp(n_heads, ff_dim, transform=transform_coords)
        else:
            self.g_RPE  = rpe_PE


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
        
        if out_dim != input_dim:
            self.compute_res = False
            self.dropout2 = nn.Identity()
            self.norm2 = nn.Identity()
        else:
            self.compute_res = True
            self.dropout2 = nn.Dropout(dropout)
            self.norm2 = nn.LayerNorm(model_dim)

    def forward(self, q, k, v, rel_coords, return_debug=False):
        e = v.shape[-1]

        batched = rel_coords.shape[0] == v.shape[0]
        rp_bias = self.g_RPE(rel_coords, batched=batched)

        att_out = self.local_att(q, k, v, rp_bias, return_debug)

        if return_debug:
            att = att_out[1]
            att_out = att_out[0]

        x = q + self.dropout1(att_out)
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
    def __init__(self, model_dim, ff_dim, l_RPE=None, output_dim=None, p_reduction=0, nh=4, n_heads=10, dropout=0.1, self_att=True, g_RPE=None):
        super().__init__()

        if self_att:
            self.att_block = attention_Block(model_dim, ff_dim, out_dim=output_dim, input_dim=model_dim, n_heads=n_heads, rpe_PE=g_RPE)
        else:
            self.att_block = None

        self.nh_block = nh_Block_self(nh, model_dim, ff_dim, l_RPE, out_dim=model_dim, dropout=dropout, n_heads=n_heads)
        
        if p_reduction > 0:
            self.voting_layer = reduction_layer(nh*2, p_reduction=p_reduction)
            #self.voting_layer = voting_layer(nh, n_heads=n_heads, p_reduction=p_reduction)
        else:
            self.voting_layer = None
               
    def forward(self, x, coords, return_debug=False):
        b,t,e = x.shape

        if return_debug:
            atts = []

        output = self.nh_block(x, coords, return_debug)
        x = output[0]
        att_nh = output[1]
        indices = output[2]

        if return_debug:
            atts.append(att_nh)

        if self.voting_layer is not None:
            #x, coords = self.voting_layer(x, att_nh, coords)
            x, coords, local_dist = self.voting_layer(x, coords)
        else:
            local_dist= []
        
        if self.att_block is not None:
            rel_coords = helpers.get_coord_relation(coords, coords)
            x = self.att_block(x, x, x, rel_coords, return_debug)

            if return_debug:
                atts.append(x[1])
                x = x[0]

        if return_debug:
            return x, coords, atts, local_dist
        
        return x, coords


class DecoderBlock(nn.Module):
    def __init__(self, model_dim, ff_dim, l_RPE=None, output_dim=None, nh=4, n_heads=10, dropout=0.1, cross_att=True, g_RPE=None):
        super().__init__()

        if cross_att:
            self.att_block = attention_Block(model_dim, ff_dim, out_dim=output_dim, input_dim=model_dim, n_heads=n_heads, rpe_PE=g_RPE)
        else:
            self.att_block = None

        self.nh_block = nh_Block_self(nh, model_dim, ff_dim, l_RPE, out_dim=model_dim, dropout=dropout, n_heads=n_heads)
    
               
    def forward(self, x, x_enc, coords_target, coords_source, return_debug=False):
        b,t,e = x.shape

        if return_debug:
            atts = []

        output = self.nh_block(x, coords_target, return_debug)
        x = output[0]
        att_nh = output[1]

        if return_debug:
            atts.append(att_nh)

        if self.att_block is not None:
            rel_coords = helpers.get_coord_relation(coords_target, coords_source)
            x = self.att_block(x, x_enc, x_enc, rel_coords, return_debug)

            if return_debug:
                atts.append(x[1])
                x = x[0]

        if return_debug:
            return x, atts
        
        return x

class Encoder(nn.Module):

    def __init__(self, n_layers, self_att_layers, p_reduction_layers, model_dim, nh, ff_dim, n_heads=10, dropout=0.1, PE=None, share_local_rpe=False, share_global_rpe=False):
        super().__init__()

        model_dim_nh = model_dim // nh
        ff_dim_nh = ff_dim // nh

        if PE is None:
            PE = helpers.RelativePositionEmbedder_mlp(model_dim_nh, ff_dim_nh, transform=True)
        else:
            PE = None

        if share_local_rpe:
            l_RPE = helpers.RelativePositionEmbedder_mlp(n_heads, ff_dim, transform=True)
        else:
            l_RPE = None
        
        if share_global_rpe:
            g_RPE = helpers.RelativePositionEmbedder_mlp(n_heads, ff_dim, transform=True)
        else:
            g_RPE = None

        if n_layers>0:
            self.nh_input_net = nh_Block_mix(nh, model_dim_nh, ff_dim_nh, PE, dropout=dropout, n_heads=n_heads)
        else:
            self.nh_input_net = None

        self.layers = nn.ModuleList()

        for n in range(n_layers):
            
            self.layers.append(EncoderBlock(model_dim,
                                        ff_dim,
                                        g_RPE=g_RPE,
                                        l_RPE=l_RPE,
                                        output_dim=model_dim,
                                        n_heads=n_heads,
                                        dropout=dropout,
                                        self_att=self_att_layers[n],
                                        p_reduction=p_reduction_layers[n]))

    def forward(self, x, coords_source, return_debug=False):
            
        atts = []
        local_dists = []

        if self.nh_input_net is not None:
            x = self.nh_input_net(x, coords_source, coords_source, return_debug=return_debug)

            if return_debug:
                atts += x[1]
                x = x[0]

        for layer in self.layers:
            x = layer(x, coords_source, return_debug)

            if return_debug:
                atts += x[2]
                local_dists.append(x[3])

            coords_source = x[1]
            x = x[0]

    
        if return_debug:
            return x, coords_source, atts, local_dists
        else:
            return x, coords_source


class Decoder(nn.Module):

    def __init__(self, dec_layers, model_dim, nh, ff_dim, output_dim=1, n_heads=10, dropout=0.1, PE=None, share_local_rpe=False, share_global_rpe=False):
        super().__init__()
    
        model_dim_nh = model_dim // nh
        ff_dim_nh = ff_dim // nh

        if PE is None:
            PE = helpers.RelativePositionEmbedder_mlp(model_dim_nh, ff_dim_nh, transform=True)
        else:
            PE = None

        if share_local_rpe:
            l_RPE = helpers.RelativePositionEmbedder_mlp(n_heads, ff_dim, transform=True)
        else:
            l_RPE = None
        
        if share_global_rpe:
            g_RPE = helpers.RelativePositionEmbedder_mlp(n_heads, ff_dim, transform=True)
        else:
            g_RPE = None
  
        self.nh_input_net = nh_Block_mix(nh, model_dim_nh, ff_dim_nh, PE, input_dim=1, dropout=dropout, n_heads=n_heads)

        self.interpolation =  nn.Sequential(
            nn.Linear(model_dim, ff_dim, bias=True),
            nn.Dropout(dropout),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            nn.Linear(ff_dim, 1, bias=True)
        )

        self.layers = nn.ModuleList()
        for n in range(dec_layers):
            out_dim = output_dim if n==dec_layers-1 else model_dim
            self.layers.append(DecoderBlock(model_dim,
                                        ff_dim,
                                        g_RPE=g_RPE,
                                        l_RPE=l_RPE,
                                        output_dim=out_dim,
                                        nh=nh,
                                        n_heads=n_heads,
                                        dropout=dropout,
                                        cross_att=True))
        
        
        
    def forward(self, x, x_enc, coords_target, coords_source, coords_source_enc, return_debug=False):

        atts = []

        x = self.nh_input_net(x, coords_target, coords_source, return_debug=return_debug)

        if return_debug:
            atts += x[1]
            x = x[0]

        out_proj = self.interpolation(x)

        for layer in self.layers:
            x = layer(x, x_enc, coords_target, coords_source_enc, return_debug)

            if return_debug:
                atts += x[1]
                x = x[0]
        
        if x.shape[-1]==out_proj.shape[-1]:
            x = x + out_proj
        else:
            x = out_proj

        if return_debug:
            return x, atts
        else:
            return x
        

class SpatialTransNet(tm.transformer_model):
    def __init__(self, model_settings) -> None: 
        super().__init__(model_settings)
        
        model_settings = self.model_settings
        dropout = model_settings['dropout']
        model_dim = model_settings['model_dim']
        n_heads = model_settings['n_heads']
        self.train_interpolation = model_settings['train_interpolation']
        nh = model_settings['feat_nh']
        ff_dim = model_settings['ff_dim']

        model_dim_nh = model_dim // nh
        ff_dim_nh = ff_dim // nh

        if model_settings['share_PE']:
            PE = helpers.RelativePositionEmbedder_mlp(model_dim_nh, ff_dim_nh, transform=True)
        else:
            PE = None

        self.skip_encoder = model_settings['encoder']['n_layers']==0
        self.Encoder = Encoder(
            model_settings['encoder']['n_layers'],
            model_settings['encoder']['self_att_layers'],
            model_settings['encoder']['p_reduction_layers'],
            model_settings['model_dim'],
            model_settings['feat_nh'],
            model_settings['ff_dim'],
            PE=PE,
            n_heads=n_heads,
            dropout=dropout,
            share_local_rpe=self.model_settings['encoder']['share_local_rpe'],
            share_global_rpe=self.model_settings['encoder']['share_global_rpe'],
        )

        self.Decoder = Decoder(
            model_settings['decoder']['n_layers'],
            model_settings['model_dim'],
            model_settings['feat_nh'],
            model_settings['ff_dim'],
            PE=PE,
            output_dim = model_settings['output_dim'],
            share_local_rpe=self.model_settings['decoder']['share_local_rpe'],
            share_global_rpe=self.model_settings['decoder']['share_global_rpe'],
            n_heads=n_heads,
            dropout=dropout
        )
        

    def forward(self, x, coord_dict, return_debug=False):
        
        coords_source = coord_dict['rel']['source']
        coords_target = coord_dict['rel']['target']

        atts = []
        #rel_embs = []

        if not self.skip_encoder:
            output = self.Encoder(x, coords_source, return_debug=return_debug)

            x_enc = output[0]
            coords_source_enc = output[1]

            if return_debug:
                atts += output[2]
                local_dists = output[3]
        else:
            x_enc = x
            coords_source_enc = coords_source

        
        x = self.Decoder(x, x_enc, coords_target, coords_source, coords_source_enc, return_debug=return_debug)

        if return_debug:
            atts += x[1]
            #rel_embs += x[2]
            x = x[0]
    

        if return_debug:

            debug_dict = {
                          'atts':atts,
                          'coords_source': coords_source_enc,
                          'local_dists': local_dists
                        }
            
            return x, debug_dict
        else:
            return x
    
