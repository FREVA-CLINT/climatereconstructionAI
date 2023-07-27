import torch
import torch.nn as nn
import torch.functional as F

import climatereconstructionai.model.transformer_helpers as helpers
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

        self.input_dropout = nn.Dropout(dropout)
        self.source_inp_layer = nn.Linear(emb_settings['emb_dim'], model_dim)
        self.target_inp_layer = nn.Linear(emb_settings['emb_dim'], emb_dim_target)

        self.t_proj = helpers.nearest_proj_layer(5)

    def forward(self, x, coord_dict):
        
        b,t,e = x.shape
        x = self.input_dropout(x)

        ape_emb_s ,b_idx_as =  self.APE_phys(coord_dict['abs']['source'][0], coord_dict['abs']['source'][1])
        xs = x + ape_emb_s

        xt = self.t_proj(x, torch.sqrt(coord_dict['rel']['target-source'][0]**2+coord_dict['rel']['target-source'][1]**2))
        ape_emb_t, b_idx_at =  self.APE_phys(coord_dict['abs']['target'][0], coord_dict['abs']['target'][1])
        xt = xt + ape_emb_t
        #xt = torch.zeros((b,len(coord_dict['abs']['target'][0]),1), device='cpu') + ape_emb_t.permute(1,2,0)

        xs = self.source_inp_layer(xs)
        xt = self.target_inp_layer(xt)

        return xs, xt, b_idx_as, b_idx_at


class CRTransNetBlock(nn.Module):
    def __init__(self, model_dim, emb_dim_target, ff_dim, RPE_phys, n_heads=10, dropout=0.1, is_final_layer=False, logit_scale=True):
        super().__init__()

        self.is_final_layer = is_final_layer

        self.q = nn.Linear(emb_dim_target, model_dim)
        self.k = nn.Linear(model_dim, model_dim)
        self.v = nn.Linear(model_dim, model_dim)


        self.att_layer = helpers.MultiHeadAttentionBlock(
            model_dim, emb_dim_target, RPE_phys, n_heads, logit_scale=logit_scale
            )
                
        self.mlp_layer = nn.Sequential(
            nn.Linear(emb_dim_target, ff_dim),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(ff_dim, emb_dim_target),
            nn.ReLU(inplace=True)
        )
       
        self.norm1 = nn.LayerNorm(emb_dim_target)

        if is_final_layer:
            self.norm2 = nn.Identity()
        else:
            self.norm2 = nn.LayerNorm(emb_dim_target)

        self.dropout = nn.Dropout(dropout)

    def forward(self, xs, xt, rel_coords):
        
        q = self.q(xt)
        k = self.k(xs)
        v = self.v(xs)

        att_out, att, rel_emb, b_emb = self.att_layer(q, k, v, rel_coords)

        x = xt + self.dropout(att_out)
        x = self.norm1(x)

        if self.is_final_layer:
            x = x + self.mlp_layer(x)  
        else:
            x = x + self.dropout(self.mlp_layer(x))
            x = self.norm2(x)

        return x, att, rel_emb, b_emb


class CRTransNet(nn.Module):
    def __init__(self, model_settings) -> None: 
        super().__init__()
        
        dropout = model_settings['dropout']
        n_heads = model_settings['mixed_layer']['n_heads']

        if model_settings['embeddings']['rel']['polar']:
            emb_class = helpers.RelativePositionEmbedder_polar
        else:
            emb_class = helpers.RelativePositionEmbedder_cart

        model_settings['embeddings']['rel']['emb_dim'] = n_heads
        self.RPE_phys = emb_class(model_settings['embeddings']['rel'])

        self.input_net = Input_Net(
            model_settings['embeddings']['abs'],
            model_settings['model_dim'],
            model_settings['emb_dim_target'])


        self.EncoderList = nn.ModuleList()
        for layer in range(model_settings['encoder']['n_layers']):

            if not model_settings['share_rel_emb']:
                self.RPE_phys = emb_class(model_settings['embeddings']['rel'])

            self.EncoderList.append(CRTransNetBlock(
                model_dim=model_settings['model_dim'],
                emb_dim_target=model_settings['model_dim'],
                RPE_phys=self.RPE_phys,
                ff_dim=model_settings['encoder']['ff_dim'],
                n_heads=n_heads,
                dropout=dropout,
                is_final_layer=False,
                logit_scale=model_settings['logit_scale']))
            

        self.DecoderList = nn.ModuleList()
        for layer in range(model_settings['encoder']['n_layers']):
            
            if not model_settings['share_rel_emb']:
                self.RPE_phys = emb_class(model_settings['embeddings']['rel'])

            self.DecoderList.append(CRTransNetBlock(
                model_dim=model_settings['emb_dim_target'],
                emb_dim_target=model_settings['emb_dim_target'],
                RPE_phys=self.RPE_phys,
                ff_dim=model_settings['decoder']['ff_dim'],
                n_heads=n_heads,
                dropout=dropout,
                is_final_layer=False,
                logit_scale=model_settings['logit_scale']))

        
        self.MixedList = nn.ModuleList()

        for layer in range(model_settings['mixed_layer']['n_layers']):
            if layer==0:
                kv_dim = model_settings['model_dim']
            else:
                kv_dim = model_settings['emb_dim_target']

            if layer==model_settings['mixed_layer']['n_layers']-1:
                is_final_layer=True
            else:
                is_final_layer=False

            self.MixedList.append(
                CRTransNetBlock(
                    model_dim=kv_dim,
                    emb_dim_target=model_settings['emb_dim_target'],
                    RPE_phys=self.RPE_phys,
                    ff_dim=model_settings['mixed_layer']['ff_dim'],
                    n_heads=n_heads,
                    dropout=dropout,
                    is_final_layer=is_final_layer,
                    logit_scale=model_settings['logit_scale']))
        

        out_dim_mixed = model_settings['emb_dim_target']
        self.mlp_out = nn.Sequential(nn.Linear(out_dim_mixed,1),nn.ReLU(inplace=True))
        

    def forward(self, x, coord_dict):
        b,s,e = x.shape

        xs, xt, b_idx_as, b_idx_at = self.input_net(x, coord_dict)

        rel_embs=[]
        for layer in self.EncoderList:
            xs, att, rel_emb, _ = layer(xs, xs, coord_dict['rel']['source'])
            rel_embs.append(rel_emb)

        for layer in self.DecoderList:
            xt, att, rel_emb, _ = layer(xt, xt, coord_dict['rel']['target'])
            rel_embs.append(rel_emb)

        x, att, rel_emb, b_emb = self.MixedList[0](xs, xt, coord_dict['rel']['target-source'])
        rel_embs.append(rel_emb)

        for layer in self.MixedList[1:]:
            x, att, rel_emb, b_emb= layer(x, x, coord_dict['rel']['target'])
            rel_embs.append(rel_emb)
        x = self.mlp_out(x)

        debug_dict = {'RPE_emb_table': {'lat':self.RPE_phys.emb_table_lon.embeddings_table,'lon':self.RPE_phys.emb_table_lat.embeddings_table},
                      'att_mixed': att,
                      'abs_emb_idx_t':b_idx_at,
                      'abs_emb_idx_s':b_idx_as,
                      'rel_emb':rel_embs,
                      'b_emb': b_emb
                      }

        return x, debug_dict
    
