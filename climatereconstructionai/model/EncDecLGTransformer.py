import torch
import torch.nn as nn
import torch.nn.functional as F

import climatereconstructionai.model.transformer_helpers as helpers
import climatereconstructionai.model.transformer_model as tm
from .. import config as cfg
from ..utils import grid_utils as gu

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


def merge_debug_information(debug_info, debug_info_new):
    for key in debug_info_new.keys():
        if key in debug_info:
            if not isinstance(debug_info[key],list):
                debug_info[key] = [debug_info[key]]
            if isinstance(debug_info_new[key],list):
                debug_info[key] += debug_info_new[key]
            else:
                debug_info[key].append(debug_info_new[key])
        else:
            debug_info[key] = debug_info_new[key]

    return debug_info

class nh_Block_layer(nn.Module):
    def __init__(self, nh, model_dim, model_dim_nh, ff_dim_nh, n_heads=4, dropout=0, input_dim=None, PE=None, RPE=None, nh_conv=True) -> None: 
        super().__init__()

        self.nh = nh
        self.md_nh = model_dim_nh
        self.model_dim = model_dim
        self.n_heads = n_heads

        if PE is not None:
            self.nn_layer = helpers.nn_layer(nh, both_dims=False)
        else:
            self.nn_layer = helpers.nn_layer(nh, both_dims=True)

        self.PE = PE
        self.RPE = RPE

        self.local_att = helpers.MultiHeadAttentionBlock(
            model_dim_nh, model_dim_nh, n_heads, logit_scale=True, qkv_proj=False
            )
        
        if input_dim is None:
            input_dim = model_dim_nh
    

        if model_dim == model_dim_nh or input_dim==1:
            self.mlp_nh_reduction = nn.Identity()

        else:
            self.mlp_nh_reduction = nn.Sequential(
                nn.Linear(model_dim, model_dim_nh, bias=False),
                nn.Dropout(dropout),
                nn.LeakyReLU(inplace=True, negative_slope=0.2)
            )

        self.q_proj = nn.Linear(model_dim_nh, model_dim_nh, bias=False)
        self.k_proj = nn.Linear(model_dim_nh, model_dim_nh, bias=False)
        self.v_proj = nn.Linear(model_dim_nh, model_dim_nh, bias=False)

        self.mlp_layer_nh = nn.Sequential(
            nn.Linear(model_dim_nh, ff_dim_nh, bias=False),
            nn.Dropout(dropout),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            nn.Linear(ff_dim_nh, 1, bias=False),
            nn.LeakyReLU(inplace=True, negative_slope=0.2)
        )

        if not nh_conv:
            self.conv_layer_nh = nn.Identity()
            self.mlp_layer_feat = nn.Sequential(
                nn.Linear(model_dim_nh*nh, model_dim_nh*nh, bias=False),
                nn.Dropout(dropout),
                nn.LeakyReLU(inplace=True, negative_slope=0.2),
                nn.Linear(model_dim_nh*nh, model_dim, bias=False),
                nn.LeakyReLU(inplace=True, negative_slope=0.2)
                )
        else:
            self.conv_layer_nh = nn.Sequential(nn.Conv2d(1, nh, kernel_size=(nh, 1), padding='valid', bias=False),
                                            nn.LeakyReLU(inplace=True, negative_slope=0.2))
            self.mlp_layer_feat = nn.Sequential(
                nn.Linear(model_dim_nh*nh, model_dim, bias=False),
                nn.Dropout(dropout),
                nn.LeakyReLU(inplace=True, negative_slope=0.2),
            )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout) if model_dim > 1 else nn.Identity()
        self.pe_dropout = nn.Dropout(dropout) if PE is not None else nn.Identity()

        self.norm1 = nn.LayerNorm(model_dim_nh) if input_dim > 1 else nn.Identity()
        self.norm2 = nn.LayerNorm(model_dim_nh)
        self.norm3 = nn.LayerNorm(model_dim) if model_dim > 1 else nn.Identity()

    def forward(self, x, coords, x_cross, coords_cross=None, return_debug=False):
        
        pos_enc = None # for debug information

        x = x_cross

        x = self.mlp_nh_reduction(x)

        x = self.norm1(x)
        
        #get nearest neighbours
        x_nh, _, cs_nh = self.nn_layer(x, coords, coords_cross, skip_self=False)

        b, t, nh, e = x_nh.shape
        batched = cs_nh.shape[0] == b

        if self.PE:
            pe = self.pe_dropout(self.PE(cs_nh, batched=batched))

            v = q = k = (x_nh + pe).reshape(b*t,nh,self.md_nh)

            v = self.v_proj(v)
            q = self.q_proj(q)
            k = self.k_proj(k)

            x_nh = x_nh.reshape(b*t,nh,e)

            if return_debug:
                pos_enc = pe

        else: 
            x_nh = x_nh.reshape(b*t,nh,e)
            q = k = v = x_nh
        
        if self.RPE is not None:
            rel_p_bias = self.RPE(cs_nh, batched=batched)

            if batched:
                rel_p_bias = rel_p_bias.reshape(b*t,self.nh, self.nh, self.n_heads)
            else:
                rel_p_bias = rel_p_bias.repeat(b,1,1,1)

            if return_debug:
                pos_enc = rel_p_bias
        else:
            rel_p_bias = None
        
        att_out, att = self.local_att(q, k, v, rel_pos_bias=rel_p_bias, return_debug=True)
            
        x_nh = x_nh + self.dropout1(att_out)
        x_nh = self.norm2(x_nh)
    
        x_nh = x_nh + self.dropout2(self.mlp_layer_nh(x_nh))

    
        x_nh = self.conv_layer_nh(x_nh.unsqueeze(dim=1)).squeeze()

        x_nh = x_nh.view(b*t,self.nh*self.md_nh)
        x_nh = self.dropout3(self.mlp_layer_feat(x_nh).view(b,t,self.model_dim))

        x = self.norm3(x_nh)

        if return_debug:
            debug_information = {"atts": att.detach(),
                                 "pos_encs":pos_enc}
            
            return x, debug_information
        else:
            return x
            

class attention_Block(nn.Module):
    def __init__(self, model_dim, ff_dim, out_dim=1, input_dim=1, dropout=0, n_heads=4, PE=None, RPE=None) -> None: 
        super().__init__()

        self.PE = PE
        self.RPE = RPE

        self.norm_PEq = nn.LayerNorm(model_dim) if PE is not None else nn.Identity()
        self.norm_PEk = nn.LayerNorm(model_dim) if PE is not None else nn.Identity()

        self.dropout_PEq = nn.Dropout(dropout) if PE is not None else nn.Identity()
        self.dropout_PEk = nn.Dropout(dropout) if PE is not None else nn.Identity()
    

        self.global_att = helpers.MultiHeadAttentionBlock(
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
            self.norm3 = nn.Identity()
        else:
            self.compute_res = True
            self.dropout2 = nn.Dropout(dropout)
            self.norm2 = nn.LayerNorm(model_dim)
            self.norm3 = nn.LayerNorm(model_dim)

    def forward(self, x, coords, x_cross=None, coords_cross=None, return_debug=False):
        
        pos_enc = None

        if x_cross is not None:
            k = v = x_cross
        else:
            k = v = x

        q = x

        if coords_cross is not None:
            coords_cross_ = coords_cross
        else:
            coords_cross_ = coords

        if self.RPE is not None:
            rel_coords = helpers.get_coord_relation(coords, coords_cross_, cart=True)
            batched = rel_coords.shape[0] == v.shape[0]
            rel_p_bias = self.RPE(rel_coords, batched=batched)

            if return_debug:
                pos_enc = rel_p_bias

        else:
            rel_p_bias = None
        
        if self.PE is not None:
            batched = coords.shape[0] == q.shape[0]
            pe_q = self.dropout_PEq(self.PE(coords, batched=batched))
            pe_k = self.dropout_PEk(self.PE(coords_cross_, batched=batched))

            q = self.norm_PEq(q + pe_q)
            k = self.norm_PEk(k + pe_k)

            if return_debug:
                pos_enc = pe_q

        att_out = self.global_att(q, k, v, rel_p_bias, return_debug)

        if return_debug:
            debug_information = {"atts": att_out[1].detach(),
                                 "pos_encs":pos_enc}
            att_out = att_out[0]

        x = x + self.dropout1(att_out)
        x = self.norm2(x)
        
        mlp_out = self.dropout2(self.mlp_layer(x))

        if self.compute_res:
            x = x + mlp_out
        else:
            x = mlp_out
        
        x = self.norm3(x)

        if return_debug:
            return x, debug_information
        
        return x

class reduction_layer(nn.Module):
    def __init__(self, p_reduction, nh) -> None: 
        super().__init__()
        nh = nh*2
        self.nn_layer = helpers.nn_layer(nh, both_dims=False)
        self.p_reduction = p_reduction

    def forward(self, x, coords):

        x_nh, _, rel_coords = self.nn_layer(x, coords, coords)  
        b, n, nh, e = x_nh.shape
        
        #local_dist = (x_nh[:,:,1:,:] - x_nh[:,:,[0],:]).pow(2).sum(dim=-2).sum(dim=-1).sqrt()
        #local_m = x_nh.abs().mean(dim=-2)
        global_m = x.mean(dim=1).abs()
        local_s = x_nh.std(dim=-2)
        local_dist = (local_s/global_m.unsqueeze(dim=1)).sum(dim=-1)
       
        n_keep = torch.tensor((1 - self.p_reduction)*local_dist.shape[1]).long()

        indices = torch.topk(local_dist,dim=1,k=n_keep).indices

        x = torch.gather(x, dim=1, index=indices.view(b,n_keep,1).repeat(1,1,e))
        coords = torch.gather(coords, dim=2, index=indices.view(b,1,n_keep,1).repeat(1,coords.shape[1],1,1))
    
        return x, coords, local_dist

class reduction_Block(nn.Module):
    def __init__(self, nh, p_reduction) -> None: 
        super().__init__()
        if p_reduction >0:
            self.reduction_layer = reduction_layer(p_reduction, nh)
        else:
            self.reduction_layer = None

    def forward(self, layer, x, coords, x_cross=None, coords_cross=None, return_debug=False):
        
        if self.reduction_layer is not None and x_cross is not None:
            x_cross, coords_cross, local_dist = self.reduction_layer(x_cross, coords_cross)
        else:
            coords_cross = coords_cross
            x_cross = x_cross
            local_dist = None

        x = layer(x, coords, x_cross=x_cross, coords_cross=coords_cross, return_debug=return_debug)

        if return_debug:
            debug_information = x[1]
            if local_dist is not None:
                local_dist = local_dist.detach()
            debug_information["local_dist"] = [local_dist]
            x = x[0]

            return x, debug_information
        else:
            return x

def get_pos_encoder(enc_type, model_settings, transform='inv'):

    if enc_type=="pe":
        out_dim_ = model_settings["model_dim_nh"]
        hidden_dim = model_settings["hidden_dim_pos_enc_nh"]

    elif enc_type=="rel_bias":
        out_dim_ = model_settings["n_heads"]
        hidden_dim = model_settings["hidden_dim_pos_enc"]

    elif enc_type=="abs":
        out_dim_ = model_settings["model_dim"]
        hidden_dim = model_settings["hidden_dim_pos_enc"]
    
    else:
        return None

    return helpers.RelativePositionEmbedder_mlp(out_dim_, hidden_dim, transform=transform)

def get_pos_encoders_layer(pe_types, model_settings, share, transform='inv'):

    encoders = {"pe": [],
                "rel_bias": []}
    
    for pe_type in pe_types:

        if pe_type == "pe" or pe_type == "rel_bias":
            if share: 
                if len(encoders[pe_type]) == 0:
                    encoders[pe_type].append(get_pos_encoder(pe_type, model_settings, transform=transform))
            else:
                encoders[pe_type].append(get_pos_encoder(pe_type, model_settings, transform=transform))
        else:
            encoders[pe_type]=None
    
    return encoders


class Block(nn.Module):
    def __init__(self, layer_type, input_dim, model_dim, ff_dim, model_dim_nh, ff_dim_nh, nh_conv, p_reduction=0, nh=4, n_heads=10, dropout=0.1, output_dim=None, PE=None, RPE=None):
        super().__init__()

        if layer_type == "nh":
            self.layer = nh_Block_layer(nh, model_dim, model_dim_nh, ff_dim_nh, input_dim=input_dim, dropout=dropout, n_heads=n_heads, PE=PE, RPE=RPE, nh_conv=nh_conv)
        else:
            self.layer = attention_Block(model_dim, ff_dim, out_dim=output_dim, input_dim=model_dim, n_heads=n_heads, PE=PE, RPE=RPE)

        self.reduction_layer = reduction_Block(nh, p_reduction)

        self.norm = nn.LayerNorm(model_dim)
        
               
    def forward(self, x, coords, x_cross, coords_cross, return_debug=False):
        
        if x_cross is None:
            x_cross = x

        if coords_cross is None:
            coords_cross = coords

        out = self.reduction_layer(self.layer, x , coords, x_cross, coords_cross, return_debug)

        if return_debug:
            debug_information = out[1]
            out = out[0] 

        x = self.norm(x + out)

        if return_debug:
            return x, debug_information
        
        return x


class EncDecBlock(nn.Module):

    def __init__(self, n_layers, layer_types, p_reduction_layers, pos_encoder_types, model_dim, ff_dim, model_dim_nh, ff_dim_nh, nh, nh_conv, input_dim=None, n_heads=10, dropout=0.1, cross_att_layers=None, pos_encoders=None, share=False):
        super().__init__()

        self.cross_att_layers = cross_att_layers

        pe_counter = {"pe": 0,
                      "rel_bias":0}

        self.layers = nn.ModuleList()
        for n in range(n_layers):

            pe_type = pos_encoder_types[n]

            if pe_type == "pe":
                pe = pos_encoders[pe_type][pe_counter[pe_type]]
                rpe = None
            elif pe_type == "rel_bias":
                rpe = pos_encoders[pe_type][pe_counter[pe_type]]
                pe =None
            else:
                rpe = pe = None

            if not share and (rpe is not None or pe is not None):
                pe_counter[pe_type] += 1

            if input_dim is None or n>0:
                input_dim=model_dim
            self.layers.append(Block(layer_types[n],
                                    input_dim,
                                    model_dim,
                                    ff_dim,
                                    model_dim_nh,
                                    ff_dim_nh,
                                    p_reduction=p_reduction_layers[n],
                                    nh=nh,
                                    nh_conv=nh_conv,
                                    n_heads=n_heads,
                                    dropout=dropout,
                                    output_dim=model_dim,
                                    PE=pe,
                                    RPE=rpe
                                    ))

    def forward(self, x, coords, x_cross=None, coords_cross=None, return_debug=False):
        if return_debug:
            debug_information = {}

        for n, layer in enumerate(self.layers):
            if x_cross is not None and self.cross_att_layers[n]:
                x_cross_ = x_cross
                coords_cross__ = coords_cross
            else:
                x_cross_ = None
                coords_cross__ = None

            x = layer(x, coords, x_cross=x_cross_, coords_cross=coords_cross__, return_debug=return_debug)

            if return_debug:
                debug_information = merge_debug_information(debug_information, x[1])
                x = x[0]


        if return_debug:
            return x, debug_information
        
        return x
    
class shortcut_block(nn.Module):
    def __init__(self, model_dim, ff_dim, model_dim_nh, ff_dim_nh, nh_conv, PE, nh=4, n_heads=10, dropout=0.1, output_dim=1):
        super().__init__()

        self.nh_layer = nh_Block_layer(nh, model_dim, model_dim_nh, ff_dim_nh, input_dim=1, dropout=dropout, n_heads=n_heads, PE=PE, nh_conv=nh_conv)

        
        self.mlp_out = nn.Sequential(
                nn.Linear(model_dim, ff_dim, bias=True),
                nn.Dropout(dropout),
                nn.LeakyReLU(inplace=True, negative_slope=0.2),
                nn.Linear(ff_dim, output_dim, bias=True),
                nn.LeakyReLU(inplace=True, negative_slope=0.2)
            )
        
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, x_in, coords_target, coords_source, return_debug=False):

        out = self.nh_layer(x_in, coords_target, x_in, coords_cross=coords_source)

        out = self.mlp_out(out)

        return out

class GaussActivation(nn.Module):
    def __init__(self, activation_mu=nn.Identity()):
        super().__init__() 
        self.activation_mu = activation_mu
        self.activation_std = nn.Softplus()

    def forward(self, x):
        mu = self.activation_mu(x[:,:,0])
        std = self.activation_std(x[:,:,1])
        
        return torch.stack((mu,std),-1)

class SpatialTransNet(tm.transformer_model):
    def __init__(self, model_settings) -> None: 
        super().__init__(model_settings)
        
        model_settings = self.model_settings


        self.use_gauss = model_settings["use_gauss"]
        if self.use_gauss:
            output_dim = 2
        else:
            output_dim = 1

        dropout = model_settings['dropout']
        model_dim = model_settings['model_dim']
        model_dim_nh = model_settings['model_dim_nh']
        ff_dim_nh = model_settings['ff_dim_nh']
        n_heads = model_settings['n_heads']
        ff_dim = model_settings['ff_dim']

        nh_conv = model_settings['nh_conv']
        nh = model_settings['nh']

        if model_settings["share_global_pe"]:
            share = True
            pe_types = ["pe"] + model_settings["encoder"]["att_pe_type_layers"] + model_settings["decoder"]["att_pe_type_layers"]
            pos_encoders_enc = get_pos_encoders_layer(pe_types, model_settings, share)
            pos_encoders_dec = pos_encoders_enc

        elif model_settings["share_encdec_pe"]:
            share = True
            pos_encoders_enc = get_pos_encoders_layer(["pe"]+model_settings["encoder"]["att_pe_type_layers"], model_settings, True)
            pos_encoders_dec = get_pos_encoders_layer(model_settings["decoder"]["att_pe_type_layers"], model_settings, True)
        
        else:
            share = False
            pos_encoders_enc = get_pos_encoders_layer(model_settings["encoder"]["att_pe_type_layers"], model_settings, False)
            pos_encoders_dec = get_pos_encoders_layer(model_settings["decoder"]["att_pe_type_layers"], model_settings, False)

        if model_settings['add_ape']:
            self.APE = get_pos_encoder("abs", model_settings, True)
            input_dim = model_settings['model_dim']
        else:
            self.APE = None
            input_dim = 1

        if share:
            short_cut_pe = pos_encoders_enc["pe"][0]
        else:
            short_cut_pe = get_pos_encoder("pe", model_settings, True)

        self.shortcut_block = shortcut_block(
            model_dim,
            ff_dim,
            model_dim_nh,
            ff_dim_nh,
            nh_conv,
            short_cut_pe,
            nh=nh,
            n_heads=n_heads,
            dropout=dropout,
            output_dim=1
        )

        self.dropout_ape_s = nn.Dropout(dropout)
        self.dropout_ape_t = nn.Dropout(dropout)
        
        self.Encoder = EncDecBlock(
            model_settings['encoder']['n_layers'],
            model_settings['encoder']['layer_types'],
            model_settings['encoder']['p_reduction_layers'],
            model_settings['encoder']['att_pe_type_layers'],
            model_dim,
            ff_dim,
            model_dim_nh,
            ff_dim_nh,
            nh,
            nh_conv,
            input_dim=input_dim,
            n_heads=n_heads,
            dropout=dropout,
            pos_encoders=pos_encoders_enc,
            share=share
        )
      
        self.Decoder = EncDecBlock(
            model_settings['decoder']['n_layers'],
            model_settings['decoder']['layer_types'],
            model_settings['decoder']['p_reduction_layers'],
            model_settings['decoder']['att_pe_type_layers'],
            model_dim,
            ff_dim,
            model_dim_nh,
            ff_dim_nh,
            nh,
            nh_conv,
            n_heads=n_heads,
            dropout=dropout,
            cross_att_layers=model_settings["decoder"]["cross_att_layers"],
            pos_encoders=pos_encoders_dec,
            share=share
        )
        if model_settings["use_gauss"]:
            last_layer_activation = GaussActivation(activation_mu=nn.LeakyReLU(negative_slope=0.2))
        else:
            last_layer_activation = nn.LeakyReLU(negative_slope=0.2)

        self.mlp_out = nn.Sequential(
                nn.Linear(model_dim, ff_dim, bias=True),
                nn.Dropout(dropout),
                nn.LeakyReLU(inplace=True, negative_slope=0.2),
                nn.Linear(ff_dim, output_dim, bias=True),
                last_layer_activation
            )

        self.check_pretrained()

    def forward(self, x, coord_dict, return_debug=False, rev=False):
        
        if not rev:
            coords_source = coord_dict['rel']['source']
            coords_target = coord_dict['rel']['target']
        else:
            coords_source = coord_dict['rel']['target']
            coords_target = coord_dict['rel']['source']

        if return_debug:
            debug_information = {"atts": [],
                                 "pos_encs":[],
                                 "pos_enc_abs":[],
                                 "local_dist":[]}

        x_interpolation = self.shortcut_block(x, coords_target, coords_source)
        
        if self.APE is not None:
            batched = coords_source.shape[0] == x.shape[0]
            ape_enc = self.dropout_ape_s(self.APE(coords_source, batched=batched))
            x_enc = x + ape_enc
            x_dec = x_interpolation + self.dropout_ape_t(self.APE(coords_target, batched=batched))
        else:
            ape_enc = None
            x_enc = x
            x_dec = x_interpolation

       
        x_enc = self.Encoder(x_enc, coords_source, return_debug=return_debug)

        if return_debug:
            ape_enc = ape_enc.detach if ape_enc is not None else ape_enc
            debug_information['pos_enc_abs'] = [ape_enc, x_dec.detach()]
            debug_information = merge_debug_information(debug_information, x_enc[1])
            x_enc = x_enc[0]
        
        x = self.Decoder(x_dec, coords_target, x_enc, coords_source, return_debug=return_debug)

        if return_debug:
            debug_information = merge_debug_information(debug_information, x[1])
            x = x[0]
        
        if self.use_gauss:
            x = self.mlp_out(x)
            x[:,:,0] = x[:,:,0] + x_interpolation.squeeze()
           # x = self.mlp_out(x) + x_interpolation
        else:
            x = self.mlp_out(x) + x_interpolation

        if return_debug:
            return x, debug_information
        else:
            return x
    
