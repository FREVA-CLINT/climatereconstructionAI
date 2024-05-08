import torch
import torch.nn as nn
import torch.nn.functional as F

import climatereconstructionai.model.transformer_helpers as helpers
import climatereconstructionai.model.pyramid_step_model as psm
from .. import config as cfg
from ..utils import grid_utils as gu



            

class attention_Block(nn.Module):
    def __init__(self, model_dim, ff_dim, out_dim=1, input_dim=1, dropout=0, n_heads=4, PE=None, RPE=None, add_pe=False) -> None: 
        super().__init__()

        self.PE = PE
        self.RPE = RPE

        #self.norm_PEq = nn.LayerNorm(model_dim) if PE is not None else nn.Identity()
        #self.norm_PEk = nn.LayerNorm(model_dim) if PE is not None else nn.Identity()

        self.add_pe = add_pe

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
            batched = coords.shape[0] == k.shape[0]
            pe_q = self.dropout_PEq(self.PE(coords, batched=batched))
            pe_k = self.dropout_PEk(self.PE(coords_cross_, batched=batched))

            if q is not None:
                q = self.norm_PEq(q + pe_q)
            else:
                q = self.norm_PEq(pe_q)

            k = self.norm_PEk(k + pe_k)

            if self.add_pe:
                v = v + pe_k

            if return_debug:
                pos_enc = pe_q

        att_out = self.global_att(q, k, v, rel_p_bias, return_debug)

        if return_debug:
            debug_information = {"atts": att_out[1].detach(),
                                 "pos_encs":pos_enc}
            att_out = att_out[0]

        if x is not None:
            x = x + self.dropout1(att_out)
        else:
            x = att_out

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

        rel_coords_pad = (coords>999)[:,0,:,0]
        local_dist[rel_coords_pad] = local_dist[rel_coords_pad] - 10e3

        n_keep = torch.tensor((1 - self.p_reduction)*local_dist.shape[1]).long()

        indices = torch.topk(local_dist, dim=1, k=n_keep).indices

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


class DecBlock(nn.Module):
    def __init__(self, input_ch, output_ch, kernel_size=5, dropout=0, stride=1):
        super().__init__()

        self.conv_up = nn.ConvTranspose2d(input_ch, output_ch ,padding='same', kernel_size=kernel_size, stride=stride)
        self.activation = nn.LeakyReLU(negative_slope=-0.2)
        #self.batchnorm = nn.BatchNorm2d(output_ch)

    def forward(self, x):
        

        x = self.conv_up(x)
        x = self.activation(x)
        #x = self.batchnorm
        return x

class EncBlock(nn.Module):
    def __init__(self, input_ch, output_ch, kernel_size=5, dropout=0, stride=1):
        super().__init__()

        self.conv = nn.Conv2d(input_ch, output_ch ,padding='same', kernel_size=kernel_size, stride=stride)
        #self.dropout = nn.Dropout(dropout)
        self.maxPool = nn.MaxPool2d(2,2)
        self.activation = nn.LeakyReLU(negative_slope=-0.2)
        self.batchnorm = nn.BatchNorm2d(output_ch)

    def forward(self):
        x = self.conv(x)
        x = self.maxPool(x)
        x = self.activation(x)
        x = self.batchnorm(x)
        return x

class nh_spa_mapper(nn.Module):
    def __init__(self, nh, input_dim, ff_dim, model_dim, output_dim, n_heads=4, dropout=0, PE=None, add_pe=False, polar=False) -> None: 
        super().__init__()

        self.nh = nh
        self.md_nh = model_dim // nh
        ff_dim_nh = ff_dim // nh
        self.n_heads = n_heads

        self.add_pe = add_pe

        self.nn_layer = helpers.nn_layer(nh, both_dims=False)

        self.polar=polar

        self.PE = PE
        
        if input_dim>1:
            self.mlp_input = nn.Sequential(
                nn.Linear(input_dim, self.md_nh, bias=True),
                nn.LeakyReLU(inplace=True, negative_slope=0.2),
                nn.Linear(self.md_nh, self.md_nh, bias=True),
            )
        else:
            self.mlp_input = nn.Identity()
     
        
        self.local_att = helpers.MultiHeadAttentionBlock(
            self.md_nh, self.md_nh, n_heads, logit_scale=True, qkv_proj=False
            )

        self.q_proj = nn.Linear(self.md_nh, self.md_nh, bias=False)
        self.k_proj = nn.Linear(self.md_nh, self.md_nh, bias=False)

        #if add_pe or input_dim>1:
        v_input_dim = self.md_nh
       # else:
        #    v_input_dim = 1

        self.v_proj = nn.Linear(v_input_dim, self.md_nh, bias=False)
        
        self.mlp_layer_nh = nn.Sequential(
            nn.Linear(self.md_nh, ff_dim_nh, bias=True),
            nn.Dropout(dropout),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            nn.Linear(ff_dim_nh, 1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.2)
            )

        self.mlp_layer_output = nn.Sequential(
            nn.Linear(self.md_nh*nh, ff_dim, bias=True),
            nn.Dropout(dropout),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            nn.Linear(ff_dim, output_dim, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.2)
            )
        

        self.mlp_layer_nh = nn.Identity()

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.pe_dropout = nn.Dropout(dropout) if PE is not None else nn.Identity()

        self.norm1 = nn.LayerNorm(self.md_nh)
        self.norm2 = nn.LayerNorm(self.md_nh)
       #self.norm3 = nn.LayerNorm(self.md_nh*nh)

    def forward(self, x, coords_target, coords_source, d_mat=None, return_debug=False):
        
        pos_enc = None 
       
        x = self.mlp_input(x)
        
        #get nearest neighbours
        x_nh, _, cs_nh = self.nn_layer(x, coords_target, coords_source, d_mat=d_mat, skip_self=False)
        
        if self.polar:
            d_mat, phi = helpers.to_polar(cs_nh[:,0,:,:], cs_nh[:,1,:,:])
            cs_nh = torch.stack((d_mat,phi),dim=1)

        b, t, nh, e = x_nh.shape
        batched = cs_nh.shape[0] == b
        
        pe = self.pe_dropout(self.PE(cs_nh, batched=batched))

        if self.add_pe:
            v = q = k = self.norm1(x_nh + pe).reshape(b*t,nh,self.md_nh)   
        else:
            q = k = self.norm1(x_nh + pe).reshape(b*t,nh,self.md_nh)
            v = x_nh.reshape(b*t,nh,x_nh.shape[-1])

        v = self.v_proj(v)
        q = self.q_proj(q)
        k = self.k_proj(k)

        x_nh = x_nh.reshape(b*t,nh,e)

        if return_debug:
            pos_enc = pe

        att_out, att = self.local_att(q, k, v, rel_pos_bias=None, return_debug=True)
            
        x_nh = x_nh + self.mlp_layer_nh(att_out)
        
        x_nh = self.norm2(x_nh)
        x_nh = x_nh.view(b,t,self.nh*self.md_nh)
        
        #x_nh = x_nh.transpose(-2,-1)
        x = self.mlp_layer_output(x_nh) #+ m


        if return_debug:
            debug_information = {"atts": att.detach(),
                                 "pos_encs":pos_enc}
            
            return x, debug_information
        else:
            return x
        
class Unet(nn.Module):
    def __init__(self, n_layers, n_features_in):
        super().__init__()
        
        self.e1 = EncBlock(n_features_in, n_features_in*2)
        self.e2 = EncBlock(n_features_in*2, n_features_in*4)
        self.e3 = EncBlock(n_features_in*4, n_features_in*4)

        self.d1 = DecBlock(n_features_in*4, n_features_in*4)
        self.d2 = DecBlock(n_features_in*8, n_features_in*4)
        self.d3 = DecBlock(n_features_in*6, n_features_in)


     #   enc_list = nn.ModuleList()
      #  for k in range(n_layers):
     #       n_features_in = 
     #       enc_list.append(EncBlock(n_features_in, n_features_in*k))

     #   dec_list = nn.ModuleList()

    def forward(self, x):

        x1 = self.e1(x)
        x2 = self.e2(x1)
        x3 = self.e3(x2)

        x = self.d1(x3)

        x = torch.concat((x, x2), dim=-3)
        x = self.d2(x)

        x = torch.concat((x, x1), dim=-3)
        x = self.d3(x)

        return x

class TransUNet(psm.pyramid_step_model):
    def __init__(self, model_settings) -> None: 
        super().__init__(model_settings)

        model_settings = self.model_settings

        input_dim = model_settings["input_dim"]
        ff_dim = model_settings["ff_dim"]
        dropout = model_settings['dropout']
        model_dim = model_settings['model_dim']
        nh = model_settings['model_dim']
        model_dim_nh = model_dim // nh

    
        #self.unet = Unet(4,model_settings['model_dim'])


    def forward(self, x, coord_dict, return_debug=False, rev=False):
        
        if not rev:
            coords_source = coord_dict['rel']['source']
            coords_target = coord_dict['rel']['target']
        else:
            coords_source = coord_dict['rel']['target']
            coords_target = coord_dict['rel']['source']


        return x
    
