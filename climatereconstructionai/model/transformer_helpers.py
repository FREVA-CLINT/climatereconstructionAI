import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import scipy.interpolate as inter
from ..utils import grid_utils as gu

radius_earth = 6371


class ConvSelfAttention(nn.Module):
    def __init__(self, in_channel, hw, n_heads=1):
        super().__init__()

        self.n_heads = n_heads

        self.norm = nn.LayerNorm([in_channel, hw[1], hw[0]])
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        b, c, h, w = input.shape

        n_heads = self.n_heads
        head_dim = c // self.n_heads

        norm = self.norm(input)
        qkv = self.qkv(norm).view(b, n_heads, head_dim * 3, h, w)
        q, k, v = qkv.chunk(3, dim=2)  

        out = scaled_dot_product_rpe_swin(q, k, v)[0]

        out = self.out(out.view(b, c, h, w))

        return out + input


       

def scaled_dot_product_rpe(q=None, k=None, v=None, aq=None, ak=None, av=None, bias=None, mask=None, logit_scale=None):
    # with relative position embeddings by shaw et al. (2018)
    
    if q is not None and k is not None and v is not None:

        b, n_heads, t, head_dim = q.shape
        s = k.shape[2]

        # q is the size of (t, dk)
        d_z = q.size()[-1] # embedding dimension

        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        
        if ak is not None:
            attn_logits_ak = torch.matmul(q.unsqueeze(dim=-2), ak.unsqueeze(dim=1).transpose(-2, -1)).view(b, n_heads, t, s)
            attn_logits = (attn_logits + attn_logits_ak)

        if aq is not None:
            attn_logits_aq = torch.matmul(k.unsqueeze(dim=-2), aq.unsqueeze(dim=1).transpose(-2, -1)).view(b, n_heads, t, s)
            attn_logits = (attn_logits + attn_logits_aq)    
        
        if logit_scale is not None:
            logit_scale = torch.clamp(logit_scale, max=math.log(100.0)).exp()
            attn_logits = attn_logits * logit_scale           

        if bias is not None:
            attn_logits = attn_logits + bias

        attn_logits = attn_logits/torch.sqrt(torch.tensor(d_z))

    else:
        attn_logits = bias

    if mask is not None:
        mask = mask.view(mask.shape[0],1,1,mask.shape[1]).repeat_interleave(attn_logits.shape[1],dim=1)
        mask_value = -1e10 if attn_logits.dtype == torch.float32 else -1e4
        attn_logits = attn_logits.masked_fill(mask, mask_value)

    #softmax and scale
    attention = F.softmax(attn_logits, dim=-1)

    values = torch.matmul(attention, v)

    if av is not None:
        attention_av = torch.matmul(attention.unsqueeze(dim=-2), av.unsqueeze(dim=1)).squeeze(dim=-2)
        values = values + attention_av

    return values, attention


def scaled_dot_product_rpe_swin(q, k, v, b=None, logit_scale=None, mask=None):
    # with relative position embeddings swin transformer (2022)
    
    # q is the size of (t, dk)
    d_z = q.size()[-1] # embedding dimension

    if logit_scale is not None:
        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        logit_scale = torch.clamp(logit_scale, max=math.log(100.0)).exp()
        attn = attn * logit_scale
    
    else:
        attn = torch.matmul(q, k.transpose(-2, -1))
        attn = (attn)/torch.sqrt(torch.tensor(d_z))
    
   # b,n,s,s = attn.shape
    if b is not None:
        attn =  attn + b

    if mask is not None:
        mask = mask.reshape(attn.shape[0],1,mask.shape[-2],mask.shape[-1]).repeat(1,attn.shape[1],1,1)
        attn = attn.masked_fill(mask, -1e10)

    attn = F.softmax(attn, dim=-1)
    values = torch.matmul(attn, v)

    return values, attn



class PositionEmbedder_angular(nn.Module):

    def __init__(self, min_pos_phys, max_pos_phys, n_pos_emb, n_heads=10, device='cpu', special_token=True, constant_init=False, continuous_projection=None):
        super().__init__()
        self.max_pos_phys = max_pos_phys
        self.min_pos_phys = min_pos_phys
        self.n_pos_emb = n_pos_emb      
       
        if constant_init:
            self.embeddings_table = nn.Parameter(torch.ones(n_pos_emb + 1, n_heads))
            if special_token:
                self.special_token = nn.Parameter(torch.ones(1, n_heads))
            else:
                self.special_token = None
        else:
            self.embeddings_table = nn.Parameter(torch.Tensor(n_pos_emb + 1, n_heads))
            if special_token:
                self.special_token = nn.Parameter(torch.Tensor(1, n_heads))
            else:
                self.special_token = None
            nn.init.xavier_uniform_(self.special_token)
            nn.init.xavier_uniform_(self.embeddings_table)


    def forward(self, coord, special_token_mask=None):
        
        coord_pos = self.n_pos_emb * (coord -self.min_pos_phys) / (self.max_pos_phys - self.min_pos_phys)

       # if special_token_mask is not None:

        coord_pos_clipped = torch.clamp(coord_pos, 0, self.n_pos_emb)
        
        embeddings = self.embeddings_table[coord_pos_clipped.long()]

        if special_token_mask is not None:
            embeddings[special_token_mask,:] = self.special_token

        return embeddings


class PositionEmbedder_phys(nn.Module):

    def __init__(self, min_pos_phys, max_pos_phys, n_pos_emb, n_heads=10, device='cpu', special_token=True, constant_init=False):
        super().__init__()
        self.max_pos_phys = max_pos_phys
        self.min_pos_phys = min_pos_phys
        self.n_pos_emb = n_pos_emb
        

        if constant_init:
            self.embeddings_table = nn.Parameter(torch.ones(n_pos_emb, n_heads))
            if special_token:
                self.special_token = nn.Parameter(torch.ones(1, n_heads))
            else:
                self.special_token = None
        else:
            self.embeddings_table = nn.Parameter(torch.Tensor(n_pos_emb + 1, n_heads))
            if special_token:
                self.special_token = nn.Parameter(torch.Tensor(1, n_heads))
            else:
                self.special_token = None
            nn.init.xavier_uniform_(self.special_token)
            nn.init.xavier_uniform_(self.embeddings_table)


    def forward(self, coord, special_token_mask=None):
        
        coord_pos = self.n_pos_emb * (coord -self.min_pos_phys) / (self.max_pos_phys - self.min_pos_phys)

       # if special_token_mask is not None:

        coord_pos_clipped = torch.clamp(coord_pos, 0, self.n_pos_emb)
        
        embeddings = self.embeddings_table[coord_pos_clipped.long()]

        if special_token_mask is not None:
            embeddings[special_token_mask,:] = self.special_token

        return embeddings


class PositionEmbedder_phys_log(nn.Module):

    def __init__(self, min_pos_phys, max_pos_phys, n_pos_emb, n_heads=10, device='cpu'):
        super().__init__()
        self.max_pos_phys = max_pos_phys
        self.min_pos_phys = min_pos_phys
        self.n_pos_emb = n_pos_emb
        self.mn_mx_log = torch.tensor([self.min_pos_phys, self.max_pos_phys]).log10()

        self.embeddings_table = nn.Parameter(torch.Tensor(n_pos_emb + 1, n_heads))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, d_mat):
        
        # not ready
        sgn = torch.sign(d_mat)

        d_mat_pos = torch.clamp(d_mat, min=self.min_pos_phys, max=self.max_pos_phys)
        d_mat_pos = d_mat_pos.log10()

        d_mat_pos = (d_mat_pos - self.mn_mx_log[0])/(self.mn_mx_log[1] - self.mn_mx_log[0])

        d_mat_pos = self.n_pos_emb * d_mat_pos
        
        embeddings = self.embeddings_table[d_mat_pos.long()]

        return embeddings


class RelPositionEmbedder_phys_log(nn.Module):

    def __init__(self, min_dist_phys, max_dist_phys, n_pos_emb, n_heads=10, device='cpu'):
        super().__init__()
        self.max_pos_phys = max_dist_phys
        self.min_pos_phys = min_dist_phys
        self.n_pos_emb = n_pos_emb

        self.rng_dist_log = torch.tensor([min_dist_phys, max_dist_phys]).log().to(device)
        self.phys_log_scale = torch.logspace(self.rng_dist_log[0], self.rng_dist_log[1], (n_pos_emb)//2+1, base=torch.e)
 
        self.embeddings_table = nn.Parameter(torch.Tensor(n_pos_emb + 1, n_heads))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, d_mat, return_emb_idx=False):
       
        sgn = torch.sign(d_mat)

        dist_log = torch.log(d_mat.abs())
        dist_log = dist_log.clamp(min=self.rng_dist_log[0], max=self.rng_dist_log[1])

        #normalize
        dist_log = (dist_log-self.rng_dist_log[0])/(self.rng_dist_log[1]-self.rng_dist_log[0])

        #scale
        dist_log = dist_log * torch.tensor(self.n_pos_emb/2)

        dist_log[sgn<0] = dist_log[sgn < 0] + (self.n_pos_emb/2 - 1)
        dist_log[sgn>=0] = (self.n_pos_emb/2 - 1) - dist_log[sgn >= 0]

        embeddings = self.embeddings_table[dist_log.long()]
        #embeddings=dist_log

        if return_emb_idx:
            return embeddings, dist_log.long()
        else:
            return embeddings


class LinearPositionEmbedder_mlp(nn.Module):
    def __init__(self, model_dim, hidden_dim, device='cpu',conv_coordinates=False):
        super().__init__()

        self.conv_coordinates = conv_coordinates

        self.rpe_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            nn.Linear(hidden_dim, model_dim, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.2)
            )
       

    def forward(self, d_mat_lon, d_mat_lat):
        
        rpe = self.rpe_mlp(torch.concat((d_mat_lon.unsqueeze(dim=-1), d_mat_lat.unsqueeze(dim=-1)),dim=-1).squeeze())
   
        return rpe

class RelativePositionEmbedder_mlp(nn.Module):
    def __init__(self, model_dim, hidden_dim, device='cpu', transform='linear',polar=False):
        super().__init__()

        self.transform = transform

        self.polar = polar
        self.rpe_mlp = nn.Sequential(
            nn.Linear(2+int(self.polar), hidden_dim, bias=False),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            nn.Linear(hidden_dim, model_dim, bias=False))

    def forward(self, coords, batched=False):
        
        if self.transform == 'log':
            if self.polar:
                coords[0] = conv_coordinates_log(coords[0])
            else:
                coords = conv_coordinates_log(coords)
        elif self.transform == 'inv':
            if self.polar:
                coords[0] = conv_coordinates_inv(coords[0])
            else:
                coords = conv_coordinates_inv(coords)
        

        if batched:
            if self.polar:
                dist, phi = coords.split(1, dim=1)
                y = torch.sin(phi)
                x = torch.cos(phi)
                coords = torch.concat((dist,x,y),dim=1)
            coords = coords.unsqueeze(dim=-1).swapaxes(1,-1).squeeze()
        else:
            if self.polar:
                dist, phi = coords.split(1, dim=0)
                y = torch.sin(phi)
                x = torch.cos(phi)
                coords = torch.concat((dist,x,y),dim=0)
            coords = coords = coords.unsqueeze(dim=-1).swapaxes(0,-1).squeeze()

        
        rpe = self.rpe_mlp(coords)
    
        return rpe
  
def conv_coordinates_log(coords):
    sgn = coords.sign()
    coords = sgn * ((1+coords.abs()).log())
    return coords

def conv_coordinates_inv_sig_log(coords):
    sgn = coords.sign()
    coords = sgn * (1 - torch.sigmoid((coords.abs()).log()))
    return coords

def conv_coordinates_sig_log(coords):
    sgn = coords.sign()
    coords = sgn * (torch.sigmoid((coords.abs()).log()))
    return coords


def conv_coordinates_inv(coords, epsilon=1e-5):
    sign = torch.sign(coords)
    sign[sign==0]=1
    coords = torch.log10(1/(coords.abs()+epsilon))   
    return sign * coords

class RelativePositionEmbedder_par(nn.Module):
    def __init__(self, n_params, min_vals, max_vals, out_dim, device='cpu'):
        super().__init__()

        self.embedding1 = PositionEmbedder_phys(min_vals[0], max_vals[0], n_params[0], n_heads=out_dim, device=device)
        self.embedding2 = PositionEmbedder_phys(min_vals[1], max_vals[1], n_params[1], n_heads=out_dim, device=device)

    def forward(self, coords):
        
        embedding1 = self.embedding1(coords[:,0,:,:])[0]
        embedding2 = self.embedding2(coords[:,1,:,:])[0]

        pos_emb = embedding1 * torch.sin(embedding2)
        return pos_emb


class SelfAttentionRPPEBlock(nn.Module):
    def __init__(self, input_dim, embed_dim, output_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.input_dim = input_dim

        self.qkv_projection = nn.Linear(input_dim, embed_dim * 3)
        
        self.output_projection = nn.Linear(embed_dim, output_dim)
        

    def forward(self, x, b):
        # batch, sequence length, embedding dimension
        b, t, e = x.shape
        qkv = self.qkv_projection(x)
        
        q, k, v = qkv.chunk(3, dim=-1)

        values, _ = scaled_dot_product_rpe_swin(q, k, v, b)
        
        x = self.output_projection(values)

        return x 

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self,  model_dim, output_dim, n_heads, input_dim= None, qkv_proj=False, dropout=0, use_bias=True, qkv_bias=True, v_proj=True):
        super().__init__()

        self.qkv_bias = qkv_bias
        self.n_heads = n_heads
        self.head_dim = model_dim // n_heads
        if input_dim is None:
            input_dim = model_dim
        
        self.output_projection = nn.Linear(model_dim, output_dim, bias=False) if v_proj else nn.Identity()
                
        if qkv_proj and qkv_bias and v_proj:
            self.qkv_projection = nn.ModuleList([nn.Linear(input_dim, model_dim, bias=False) for _ in range(3)])

        elif qkv_proj and qkv_bias:
            self.qkv_projection = nn.ModuleList([nn.Linear(input_dim, model_dim, bias=False)for _ in range(2)] + [nn.Identity()])

        elif qkv_proj and v_proj:
            self.v_projection = nn.Linear(input_dim, model_dim, bias=False)
        else:
            self.qkv_projection = nn.ModuleList([nn.Identity() for _ in range(3)])
            self.v_projection = nn.Identity()
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if use_bias and qkv_bias:
            self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((n_heads, 1, 1))))
        else:
            self.logit_scale = None

    def forward(self, q=None, k=None, v=None, ak=None, av=None, aq=None, bias=None, return_debug=False, mask=None):
        # batch, sequence length, embedding dimension

        if self.qkv_bias:
            bq, t = q.shape[0], q.shape[1] 
            bk, s = k.shape[0], k.shape[1] 
            bv, s = v.shape[0], v.shape[1] 

            q = self.qkv_projection[0](q)
            k = self.qkv_projection[1](k)
            v = self.qkv_projection[2](v)

            q = q.reshape(bq, t, self.n_heads, self.head_dim).permute(0,2,1,3)
            k = k.reshape(bk, s, self.n_heads, self.head_dim).permute(0,2,1,3)
            v = v.reshape(bv, s, self.n_heads, self.head_dim).permute(0,2,1,3)

            if ak is not None:
                ak = ak.reshape(bk, t, s, self.head_dim)
            if av is not None:
                av = av.reshape(bk, t, s, self.head_dim)
            if aq is not None:
                aq = aq.reshape(bk, t, s, self.head_dim).transpose(1,2)
            
            if bias is not None:
                bias = bias.reshape(bk, t, s, self.n_heads).permute(0,-1,1,2)

            values, att = scaled_dot_product_rpe(q=q, k=k, v=v, ak=ak, av=av, aq=aq, bias=bias, mask=mask, logit_scale=self.logit_scale)
        else:
            b, k, t, s, nheads = bias.shape
            bv = bk = b*k

            v = self.v_projection(v).reshape(bk, s, self.n_heads, self.head_dim).permute(0,2,1,3)
            if bias is not None:
                bias = bias.reshape(bk, t, s, self.n_heads).permute(0,-1,1,2)

            values, att = scaled_dot_product_rpe(v=v, bias=bias, mask=mask, logit_scale=self.logit_scale)

        values = values.permute(0,2,1,3)
        values = values.reshape(bv, t, self.head_dim*self.n_heads)

        x = self.output_projection(values)

        if return_debug:
            return x , att
        else:
            return x    
        
class MultiHeadAttentionBlock_swin(nn.Module):
    def __init__(self, model_dim, output_dim, n_heads, logit_scale=False, qkv_proj=False, dropout=0):
        super().__init__()

        self.n_heads = n_heads
        self.head_dim = model_dim // n_heads
               
        self.output_projection = nn.Linear(model_dim, output_dim, bias=False)
        
        if logit_scale:
            self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((n_heads, 1, 1))))
        else:
            self.logit_scale = None
        
        if qkv_proj:
            self.qkv_projection = nn.ModuleList([nn.Linear(model_dim, model_dim, bias=False) for _ in range(3)])
        else:
            self.qkv_projection = nn.ModuleList([nn.Identity()  for _ in range(3)])
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()


    def forward(self, q, k, v, rel_pos_bias=None, return_debug=False, mask=None):
        # batch, sequence length, embedding dimension
        bq, t = q.shape[0], q.shape[1] 
        bk, s = k.shape[0], k.shape[1] 
        bv, s = v.shape[0], v.shape[1] 

        q = self.qkv_projection[0](q)
        k = self.qkv_projection[1](k)
        v = self.qkv_projection[2](v)

        q = q.reshape(bq, t, self.n_heads, self.head_dim).permute(0,2,1,3)
        k = k.reshape(bk, s, self.n_heads, self.head_dim).permute(0,2,1,3)
        v = v.reshape(bv, s, self.n_heads, self.head_dim).permute(0,2,1,3)

        if rel_pos_bias is not None:
            if len(rel_pos_bias.shape)>3:
                rel_pos_bias = rel_pos_bias.permute(0,3,1,2)
            else:
                rel_pos_bias = rel_pos_bias.permute(-1,0,1)
        else:
            rel_pos_bias=None

        values, att = scaled_dot_product_rpe_swin(q, k, v, rel_pos_bias, self.logit_scale, mask=mask)

        values = values.permute(0,2,1,3)
        values = values.reshape(bv, t, self.head_dim*self.n_heads)

        x = self.output_projection(values)

        if return_debug:
            return x , att
        else:
            return x    

def get_coord_relation(coords_target, coords_source, cart=True):
    
    diff = coords_target.unsqueeze(dim=-1) - coords_source.unsqueeze(dim=-1).transpose(-2,-1)
    
    if cart:
        return diff
    else:
        d_mat = torch.sqrt(diff[:,0,:,:]**2 + diff[:,1,:,:]**2)
        phi = torch.atan2(diff[:,1,:,:], diff[:,0,:,:])
        return torch.stack((d_mat, phi), dim=1)

def to_polar(dlon, dlat):
    d_mat = torch.sqrt(dlon**2 + dlat**2)
    phi = torch.atan2(dlon, dlat)
    return d_mat, phi


def nh_computation(x, coords_target, coords_source, nh, skip_self=False, d_mat=None, both_dims=False):
    b,s,e = x.shape

    coord_diff = get_coord_relation(coords_target, coords_source)

    if coord_diff.dim()==3:
        c1 = coord_diff[0]
        c2 = coord_diff[1]
    else:
        c1 = coord_diff[:,0,:,:]
        c2 = coord_diff[:,1,:,:]

    if d_mat is None:
        d_mat = (c1**2 + c2**2).sqrt()

    t = d_mat.shape[-2] 

    # leave out central datapoint? attention of datapoint to neighbourhood?
    if d_mat.dim()==2:
        _, indices = d_mat.sort(dim=1, descending=False)
        indices = indices[:,int(skip_self):nh+int(skip_self)]
        x_bs = x[:,indices]

        c1 = torch.gather(c1,dim=0,index=indices).unsqueeze(dim=-1)
        c2 = torch.gather(c2,dim=0,index=indices).unsqueeze(dim=-1)

        if both_dims:
            c1 = (c1 - c1.transpose(-1,1)) 
            c2 = (c2 - c2.transpose(-1,1))
        
        cs = torch.stack([c1,c2],dim=0)

    else:
        _, indices = d_mat.sort(dim=-1, descending=False)
            
        idx_shift = (torch.arange(indices.shape[0],device=indices.device)*s).view(b,1,1)

        c_ix_shifted = indices+idx_shift
        c_ix_shifted = c_ix_shifted.reshape(b*t,s)
        x_bs = x.view(b*s,e)
        x_bs = x_bs[c_ix_shifted].view(b,t,s,e)
    
        x_bs = x_bs[:,:,int(skip_self):nh+int(skip_self),:]
        indices = indices[:,:,int(skip_self):nh+int(skip_self)]

        c1 = torch.gather(c1, dim=-1, index=indices)#.transpose(-1,1)
        c2 = torch.gather(c2, dim=-1, index=indices)#.transpose(-1,1)

        if both_dims:
            c1 = c1.unsqueeze(dim=-1)
            c2 = c2.unsqueeze(dim=-1)
            c1 = (c1 - c1.transpose(-1,-2)) 
            c2 = (c2 - c2.transpose(-1,-2))
        
        cs = torch.stack([c1,c2],dim=1)
    
    return x_bs, indices, cs


class nn_layer(nn.Module):
    def __init__(self, nh, both_dims=False, cart=True, batch_size=-1):
        super().__init__()

        self.nh = nh
        self.both_dims = both_dims
        self.cart = cart
        self.batch_size= batch_size


    def forward(self, x, coords_target, coords_source, d_mat=None, skip_self=False):
        b,s,e = x.shape

        if self.batch_size != -1:
            n_chunks = s // self.batch_size 
        else:
            n_chunks = 1
        
        coords_target_chunks = torch.chunk(coords_target, n_chunks, dim=-1)

        x_bs = [] 
        indices = []
        cs = []
        for coords_target_chunk in coords_target_chunks:
            output = nh_computation(x, coords_target_chunk, coords_source, self.nh, skip_self=skip_self, both_dims=self.both_dims, d_mat=d_mat)
            x_bs.append(output[0])
            indices.append(output[1])
            cs.append(output[2])

        x_bs = torch.concat(x_bs, dim=1)
        indices = torch.concat(indices, dim=1)
        cs = torch.concat(cs, dim=2)

        return x_bs, indices, cs

class lin_interpolator(nn.Module):
    def __init__(self,device):
        super().__init__()

        self.device = device

    def forward(self, x, coord_source, coord_target):
        
        b,s,e = x.shape
        e,t = coord_target[0].shape 

        out_tensor = torch.zeros(b,t,1)
        for sample_idx, x_ in enumerate(x):
            LinInter = inter.LinearNDInterpolator(list(zip(coord_source[0].squeeze().cpu(), coord_source[1].squeeze().cpu())), x_.cpu())
            out_tensor[sample_idx] = torch.tensor(LinInter(coord_target[0].cpu(), coord_target[1].cpu()))

        return out_tensor.to(self.device)
    

def normal(x, mu, s):
    return torch.exp(-0.5*((x-mu)/s)**2)/((s**2).sqrt()*(math.sqrt(2*math.pi)))

class nu_grid_sampler_simple(nn.Module):

    def __init__(self):
        super().__init__()
        self.softmax2d = nn.Softmax2d()

    def forward(self, x, coords):
        b, n, nc = coords.shape
        _, c, nx, ny = x.shape

        positionsx = (coords[:,:,1])*(x.shape[-2]-1)
        positionsy = (coords[:,:,0])*(x.shape[-1]-1)    

        positionsx = torch.clamp(positionsx, min=0, max=x.shape[-2])
        positionsy = torch.clamp(positionsy, min=0, max=x.shape[-1])

        x = x[:,:,positionsx.long(), positionsy.long()]

        diag_m = torch.arange(b, device=x.device).long()
        x = x[diag_m,:,diag_m]

        return x

class nu_grid_sampler(nn.Module):

    def __init__(self, n_res=90, s=0.5, nh=5, train_s=False):
        super().__init__()

        nh_m = ((nh-1)/2) + 0.5
        
        if train_s:
            self.s = nn.Parameter(torch.tensor(s), requires_grad=True)
        else:
            self.s = nn.Parameter(torch.tensor(s), requires_grad=False)

        self.pixel_offset_normal = nn.Parameter(torch.linspace(nh_m, -nh_m, n_res),requires_grad=False)
        self.pixel_offset_indices = nn.Parameter(torch.linspace(-(nh-1)//2, (nh-1)//2, nh),requires_grad=False)

        self.n_res = n_res
        self.nh = nh
        self.softmax2d = nn.Softmax2d()

    def forward(self, x, coords, return_sum=True):
        b, n, nc = coords.shape
        _, c, nx, ny = x.shape

        positionsx = (coords[:,:,1])*(x.shape[-2]-1)
        positionsy = (coords[:,:,0])*(x.shape[-1]-1)    

        #fine grid for normal distribution
        p_x_o = positionsx.round().view(b,n,1) - self.pixel_offset_normal.view(1,-1)
        p_y_o = positionsy.round().view(b,n,1) - self.pixel_offset_normal.view(1,-1)

        p_x_o = torch.clamp(p_x_o, min=0, max=x.shape[-2])
        p_y_o = torch.clamp(p_y_o, min=0, max=x.shape[-2])

        weights_x = normal(p_x_o, positionsx.view(b,n,1), self.s)
        weights_y = normal(p_y_o, positionsy.view(b,n,1), self.s)

        weights_x = weights_x.reshape(x.shape[0], n, self.nh, self.n_res//self.nh).sum(axis=-1)
        weights_y = weights_y.reshape(x.shape[0], n, self.nh, self.n_res//self.nh).sum(axis=-1)

        weights_2d = torch.matmul(weights_x.view(b, n, self.nh, 1), weights_y.view(b,n,1,self.nh))
        weights_2d = weights_2d/weights_2d.view(b,n,self.nh*self.nh).sum(dim=-1).view(b,n,1,1)
        
        #course grid for pixel locations
        p_x_o = positionsx.round().view(b,n,1) - self.pixel_offset_indices.view(1,-1)
        p_y_o = positionsy.round().view(b,n,1) - self.pixel_offset_indices.view(1,-1)

        p_x_o = torch.clamp(p_x_o.round(), min=0, max=x.shape[-2]-1).long()
        p_y_o = torch.clamp(p_y_o.round(), min=0, max=x.shape[-2]-1).long()

        p_x_o = p_x_o.unsqueeze(dim=-1).repeat(1,1,1,self.nh)
        p_y_o = p_y_o.unsqueeze(dim=-2).repeat(1,1,self.nh,1)

        p_x_o = p_x_o.view(b,n*self.nh**2)
        p_y_o = p_y_o.view(b,n*self.nh**2)

        x = x.permute(2,3,0,1)[p_x_o, p_y_o]
        x = x.permute(0,2,1,-1)

        diag_m = torch.arange(b, device=x.device).long()
        x = x[diag_m, diag_m]
        x = x.permute(0,-1,1)

        x = x.view(b,c,n,self.nh,self.nh)

        
        if return_sum:
            x = x*weights_2d.unsqueeze(dim=1)
            x = x.view(b, c, n, self.nh**2).sum(dim=-1)
        else:
            x = x.view(b, c, n, self.nh**2)

        return x
        
def scale_coords(coords, rngx, rngy=None):
    coords_scaled = {}
    non_valid = {}

    if rngy is None:
        rngy=rngx

    for spatial_dim, coords_ in coords.items():
        coords_scaled[spatial_dim] = torch.stack([(coords_[:,0] - rngx[0])/(rngx[1] - rngx[0]), (coords_[:,1] - rngy[0])/(rngy[1] - rngy[0])], dim=1)
        non_valid[spatial_dim] = torch.logical_or(
            torch.logical_or(coords_scaled[spatial_dim][:,0]>1, 
                            coords_scaled[spatial_dim][:,0]<0),
            torch.logical_or(coords_scaled[spatial_dim][:,1]>1, 
                            coords_scaled[spatial_dim][:,1]<0))

        if non_valid[spatial_dim].any():
            pass
        coords_scaled[spatial_dim] = torch.clamp(coords_scaled[spatial_dim], min=0, max=1)
    return coords_scaled, non_valid


def get_buckets_1d_batched(coords, n_q=2, equal_size=True):

    if equal_size: 
        keep =  coords.shape[-1] - coords.shape[-1] % (n_q)
        coords = coords[:,:keep]

    b,n = coords.shape

    offset = coords.shape[0]*torch.arange(coords.shape[0], device=coords.device).view(-1,1)
    coords = coords + offset

    quants = torch.linspace(1/n_q, 1-1/n_q, n_q-1, device=coords.device)
    qs = coords.quantile(quants, dim=-1)

    o_b = (coords.shape[0]+1)/2
    boundaries = torch.concat((offset.T-o_b, qs, offset.T+o_b))
    
    buckets = torch.bucketize(coords.flatten(), boundaries.flatten().sort().values, right=True)

    qs = qs - offset.T

    offset_buckets = boundaries.shape[0]*torch.arange(qs.shape[1], device=coords.device).view(-1,1)+1
    
    buckets = buckets.reshape(b,n) - offset_buckets

    idx_sort = buckets.argsort(dim=1) 
    
    indices = torch.stack(idx_sort.chunk(n_q, dim=-1),dim=1)

    return indices, qs, buckets


def get_field(n_output, coords, source, f=8):
    
    x = y = torch.linspace(0, 1, n_output, device=coords.device)
        
    b,n1,n2,c,nf = source.shape
    source_v = source.view(b,n1,n2,-1)

    coords_inter = torch.nn.functional.interpolate(coords, scale_factor=f, mode="bilinear", align_corners=True)
    source_inter = torch.nn.functional.interpolate(source_v.permute(0,-1,1,2), scale_factor=f, mode="bicubic", align_corners=True)
    source_inter = source_inter.view(b,c,nf,source_inter.shape[-2],source_inter.shape[-1])


    dev0 = ((coords_inter[:,[0]].unsqueeze(dim=-1) - x.view(1,-1))).abs()

    _, ind0 = dev0.min(dim=2)

    index_c = ind0.transpose(-2,-1).repeat(1,2,1,1)
    coords_inter0 = torch.gather(coords_inter, dim=2, index=index_c)

    source_inter = torch.gather(source_inter.mean(dim=1), dim=2, index=ind0.transpose(-2,-1).repeat(1,nf,1,1))

    dev1 = ((coords_inter0[:,[1]].unsqueeze(dim=-1) - y.view(1,-1))).abs()
    _, ind = dev1.min(dim=3)

    source_inter = torch.gather(source_inter, dim=3, index=ind.repeat(1,nf,1,1))

    return source_inter.transpose(-2,-1)



def scale_coords_pad(coords, min_val, max_val):
    c0 = coords[:,0]
    c1 = coords[:,1]
 
    scaled_c0 = (c0 - min_val)/(max_val-min_val)
    scaled_c1 = (c1 - min_val)/(max_val-min_val)

    mn0,mx0 = scaled_c0.min(dim=1).values, scaled_c0.max(dim=1).values
    mn1,mx1 = scaled_c1.min(dim=-1).values, scaled_c1.max(dim=-1).values

    zero_t = mn0.unsqueeze(dim=1) - 0.5
    one_t = mx0.unsqueeze(dim=1) + 0.5
    scaled_c0 = torch.concat((zero_t, scaled_c0, one_t),dim=1)
    scaled_c0 = torch.concat((scaled_c0[:,:,[0]], scaled_c0, scaled_c0[:,:,[-1]]),dim=2)


    zero_t = mn1.unsqueeze(dim=2) - 0.5
    one_t = mx1.unsqueeze(dim=2) + 0.5
    scaled_c1 = torch.concat((zero_t,scaled_c1, one_t),dim=2)
    scaled_c1 = torch.concat((scaled_c1[:,[0]], scaled_c1, scaled_c1[:,[-1]]),dim=1)

    return torch.stack((scaled_c0, scaled_c1), dim=1)



def batch_coords(coords, n_q):

    indices1, _, _ = get_buckets_1d_batched(coords[:,0], n_q, equal_size=True)

    c = coords[:,0]
    c1 = coords[:,1]

    b,nb,n = indices1.shape

    cs0 = torch.gather(c, dim=1, index=indices1.view(b,-1))
    cs0 = cs0.view(b,nb,n)
    cs1 = torch.gather(c1, dim=1, index=indices1.view(b,-1))
    cs1 = cs1.view(b,nb,n)

    b,n1,n = cs1.shape
    indices, _, _ = get_buckets_1d_batched(cs1.view(b*n1,n), n_q, equal_size=True)
    indices2 = indices.view(b,n1,n_q,-1)

    b,n1,n2,n = indices2.shape
    cs1_new = torch.gather(cs1, dim=2, index=indices2.view(b,n1,-1))
    cs1_new = cs1_new.view(b,n1,n2,n)

    cs0_new = torch.gather(cs0, dim=2, index=indices2.view(b,n1,-1))
    cs0_new = cs0_new.view(b,n1,n2,n)


    indices_tot = torch.gather(indices1, dim=-1, index=indices2.view(b,n1,n2*n)).view(b,n1,n2,n)
    cs2 = torch.gather(coords, dim=2, index=indices_tot.view(b,1,-1).repeat(1,2,1))
    cs2 = cs2.view(b,2,n1,n2,n)

    cs2_m = cs2.median(dim=-1).values
    #cs2_s = cs2.min(dim=-1).values
    #cs2_e = cs2.max(dim=-1).values

    return indices_tot, cs2_m


class quant_discretizer():
    def __init__(self, min_val, max_val, n) -> None: 
        super().__init__()

        self.min_nq = 16
        self.min_f = 4
        self.n_min = 8

        self.min_val = min_val
        self.max_val = max_val
        self.n = n

    def __call__(self, x, coords_source):
                
        n = x.shape[1]

        n_q = int((x.shape[1] // self.n_min)**0.5)
        n_q = self.min_nq if n_q < self.min_nq else n_q
        n_q = self.n if n_q > self.n else n_q
        f = (self.n // n_q) + 1
        f = self.min_f if f < self.min_f else f

        coords = coords_source
        data = x

        indices, cm = batch_coords(coords, n_q=n_q)
        b,n1,n2,n = indices.shape

        data_buckets  = torch.gather(data, dim=1, index=indices.view(b,-1,1).repeat(1,1,data.shape[-1]))
        data_buckets = data_buckets.view(b,n1,n2,n,data.shape[-1])

        scaled_m = scale_coords_pad(cm, self.min_val, self.max_val)

        data = torch.concat((data_buckets[:,:,[0]], data_buckets, data_buckets[:,:,[-1]]),dim=2)
        data = torch.concat((data[:,[0]], data, data[:,[-1]]),dim=1)

        b,n1,n2,n,nf = data.shape
        data = data.view(b,n1,n2,1,n*nf)

        x = get_field(self.n, scaled_m, data, f=f)

        b,_,n1,n2 = x.shape
        x = x.view(b,n,nf,n1,n2)

        return x
    

class unstructured_to_reg_qdiscretizer():
    def __init__(self, output_dim, coord_range):
        super().__init__()

        self.discretizer = quant_discretizer(coord_range[0], coord_range[1], output_dim)


    def __call__(self, x: dict, coords_source: dict, spatial_dim_var_dict: dict):

        x_spatial_dims = []

        for spatial_dim, vars in spatial_dim_var_dict.items():
            data = torch.concat([x[var] for var in vars], dim=-1)
            data_out = self.discretizer(data, coords_source[spatial_dim])
            data_out = data_out.mean(dim=1)
            x_spatial_dims.append(data_out)
        x = torch.concat(x_spatial_dims, dim=1)

        return x


class unstructured_to_reg_interpolator():
    def __init__(self, output_dim, coord_rangex, coord_rangey, method='linear'):
        super().__init__()
   
        x = np.linspace(coord_rangex[0],
                        coord_rangex[1],
                        output_dim[0])
        
        y = np.linspace(coord_rangey[0],
                        coord_rangey[1],
                        output_dim[1])
       
        
        self.inter = gu.grid_interpolator(x,y, method=method)

    def __call__(self, x, coords_source, spatial_dim_var_dict):

        x_spatial_dims = []
        nh_mapping_iter = 0
        for spatial_dim, vars in spatial_dim_var_dict.items():
            for var in vars:
                data = x[var]

                if coords_source[spatial_dim].dim()>2:
                    data_out = torch.stack([self.inter(data[idx,:,0], coords_source[spatial_dim][idx]) for idx in range(data.shape[0])])
                else:
                    data_out = self.inter(data[:,0], coords_source[spatial_dim])

                x_spatial_dims.append(data_out)
            nh_mapping_iter += 1

        if coords_source[spatial_dim].dim()>2:
            x = torch.stack(x_spatial_dims, dim=1)
        else:
            x = torch.stack(x_spatial_dims, dim=0)

        return x
    


def unique_values(row):
    return np.unique(row)

def unique_count(row):
    return np.unique(row, return_counts=True)[1].max()

def get_nh_of_batch_indices(cell_indices, adjc):
    
    cells_nh = adjc[cell_indices]

    out_of_fov = torch.logical_or(cells_nh > cell_indices.amax(dim=(-1)).reshape(-1,1,1),
                        cells_nh < cell_indices.amin(dim=(-1)).reshape(-1,1,1))
    
    ind = torch.where(out_of_fov)
    cells_nh[ind] = cell_indices[ind[0],ind[1]]
    
    return cells_nh, out_of_fov

def coarsen_global_cells(cells, eoc, acoe, global_level=1, coarsen_level=None, nh=1):
    if coarsen_level is None:
        coarsen_level = global_level
    
    coarsen_level = global_level

    n_cells = cells.shape[-1]
    n_cells_coarse = n_cells // 4**coarsen_level

    if len(cells.shape)>1:
        batched=True
        cells = cells.reshape(cells.shape[0],-1,4**coarsen_level)
        adjc = acoe.T[eoc.T[cells]]
        for _ in range(nh-1):
            adjc = acoe.T[eoc.T[adjc]]

        adjc = adjc.reshape(cells.shape[0],n_cells_coarse,-1) // 4**global_level
    else:
        batched=False
        cells = cells.reshape(-1,4**global_level)
        adjc = acoe.T[eoc.T[cells]].reshape(n_cells_coarse,-1) // 4**global_level
        for _ in range(nh-1):
            adjc = acoe.T[eoc.T[adjc]]
        
        adjc = adjc.reshape(n_cells_coarse,-1) // 4**global_level

    local_cells = cells // 4**global_level

    out_of_fov = None

    adjc_unique = (adjc).long().unique(dim=-1)

    if batched:
        self_indices = local_cells[:,:,[0]]
    else:
        self_indices = local_cells[:,[0]]
    
    is_self = adjc_unique - self_indices == 0

    is_self_count = is_self.sum(axis=-1)

    if torch.all(is_self_count==is_self_count[0]):

        cells_nh = adjc_unique[~is_self]

        if batched:
            cells_nh = cells_nh.reshape(cells.shape[0], n_cells_coarse,-1)
            
            out_of_fov = torch.logical_or(cells_nh > local_cells.amax(dim=(-2,-1)).reshape(-1,1,1),
                            cells_nh < local_cells.amin(dim=(-2,-1)).reshape(-1,1,1))
            ind = torch.where(out_of_fov)
            cells_nh[ind] = local_cells[ind[0],ind[1],0]
        else:
            cells_nh = cells_nh.reshape(cells.shape[0],3)

    else:
        cells_nh = None

    return cells, local_cells, cells_nh, out_of_fov


def get_nh_values(values, indices_nh=None, sample_indices=None, coarsest_level=4, global_level=0, nh=1):
    if sample_indices is None:
        return values[indices_nh]
    else:
        b,n,e = values.reshape(values.shape[0],values.shape[1],-1).shape
        indices_offset_level = sample_indices*(4**(coarsest_level-global_level))
        indices_level = indices_nh - indices_offset_level.reshape(-1,1,1)

        return torch.gather(values.reshape(b,-1,e),1, index=indices_level.reshape(b,-1,1).repeat(1,1,e)).reshape(b,n,3,e)
