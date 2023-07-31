import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import scipy.interpolate as inter

radius_earth = 6371


def scaled_dot_product_rpe(q, k, v, a_k, a_v):
    # with relative position embeddings by shaw et al. (2018)

    # q is the size of (t, dk)
    d_z = q.size()[-1] # embedding dimension

    attn_logits_k = torch.matmul(q, k.transpose(-2, -1))
    attn_logits_ak = torch.matmul(q, a_k.transpose(-2, -1))

    attn_logits = (attn_logits_k + attn_logits_ak)/torch.sqrt(torch.tensor(d_z))

    #softmax and scale
    attention = F.softmax(attn_logits, dim=-1)

    attention_v = torch.matmul(attention, v)
    attention_av = torch.matmul(attention, a_v)

    values = attention_v + attention_av

    return values, attention


def scaled_dot_product_rpe_swin(q, k, v, b, logit_scale=None):
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
    attn =  attn + b

    attn = F.softmax(attn, dim=-1)

    values = torch.matmul(attn, v)

    return values, attn


class PositionEmbedder_phys(nn.Module):

    def __init__(self, min_pos_phys, max_pos_phys, n_pos_emb, n_heads=10,device='cpu'):
        super().__init__()
        self.max_pos_phys = max_pos_phys
        self.min_pos_phys = min_pos_phys
        self.n_pos_emb = n_pos_emb
        
        self.embeddings_table = nn.Parameter(torch.Tensor(n_pos_emb + 1, n_heads)).to(device)
        nn.init.xavier_uniform_(self.embeddings_table)


    def forward(self, d_mat):

        d_mat_pos = self.n_pos_emb * (d_mat -self.min_pos_phys) / (self.max_pos_phys - self.min_pos_phys)

        d_mat_clipped = torch.clamp(d_mat_pos, 0, self.n_pos_emb)
        
        embeddings = self.embeddings_table[d_mat_clipped.long()]

        return embeddings,d_mat_clipped.long()


class PositionEmbedder_phys_log(nn.Module):

    def __init__(self, min_pos_phys, max_pos_phys, n_pos_emb, n_heads=10, device='cpu'):
        super().__init__()
        self.max_pos_phys = max_pos_phys
        self.min_pos_phys = min_pos_phys
        self.n_pos_emb = n_pos_emb

        self.embeddings_table = nn.Parameter(torch.Tensor(n_pos_emb + 1, n_heads)).to(device)
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, d_mat, return_emb_idx=False):
        
        d_mat_pos = (d_mat-self.min_pos_phys) / (self.max_pos_phys - self.min_pos_phys)
    
        d_mat_clipped = torch.clamp(torch.log(d_mat_pos), max=torch.tensor(0)).exp()
        d_mat_clipped = self.n_pos_emb * d_mat_clipped
        
        embeddings = self.embeddings_table[d_mat_clipped.long()]

        if return_emb_idx:
            return embeddings, d_mat_clipped.long()
        else:
            return embeddings


class RelPositionEmbedder_phys_log(nn.Module):

    def __init__(self, min_dist_phys, max_dist_phys, n_pos_emb, n_heads=10, device='cpu'):
        super().__init__()
        self.max_pos_phys = max_dist_phys
        self.min_pos_phys = min_dist_phys
        self.n_pos_emb = n_pos_emb

        self.rng_dist_log = torch.tensor([min_dist_phys, max_dist_phys]).log().to(device)
        self.phys_log_scale = torch.logspace(self.rng_dist_log[0], self.rng_dist_log[1], (n_pos_emb)//2+1, base=torch.e)
 
        self.embeddings_table = nn.Parameter(torch.Tensor(n_pos_emb + 1, n_heads)).to(device)
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

class RelativePositionEmbedder_polar(nn.Module):
    def __init__(self, settings, rot_emb_table=None, distance_emb_table=None):
        super().__init__()

        max_dist=settings['max_dist']
        n_dist=settings['n_dist']
        n_phi=settings['n_phi']
        use_mlp=settings['use_mlp']
        emb_dim=settings['emb_dim']

        self.use_mlp = use_mlp
        self.n_heads = emb_dim

        if use_mlp:
            n_heads_pos = 1
            self.rpe_mlp = nn.Sequential(nn.Linear(2, settings['mlp_hidden_dim'], bias=True), nn.ReLU(inplace=True), nn.Linear(settings['mlp_hidden_dim'], emb_dim))
        else:
            n_heads_pos = emb_dim
            self.rpe_mlp = nn.Identity()

        if distance_emb_table is None:
            self.distance_emb_table = PositionEmbedder_phys_log(0, max_dist/radius_earth, n_dist, n_heads=n_heads_pos)
        else:
            self.distance_emb_table = distance_emb_table
        
        if rot_emb_table is None:
            self.rot_emb_table = PositionEmbedder_phys(-torch.pi, torch.pi, n_phi, n_heads=n_heads_pos)
        else:
            self.rot_emb_table = rot_emb_table


    def forward(self, d_mat, phi_mat):
        
        a_d,idx_d  = self.distance_emb_table(d_mat)
        a_phi,idx_phi = self.rot_emb_table(phi_mat)

        if self.use_mlp:
            rpe = self.rpe_mlp(torch.concat((a_d, a_phi),dim=3))
        else:
            rpe = (a_d + a_phi)

        return rpe.permute(2,0,1), (idx_d,idx_phi)

class RelativePositionEmbedder_cart(nn.Module):
    def __init__(self, settings, emb_table_lon=None, emb_table_lat=None, device='cpu'):
        super().__init__()

        max_dist=settings['max_dist']
        n_dist=settings['n_dist']
        use_mlp=settings['use_mlp']
        emb_dim=settings['emb_dim']
        min_dist=settings['min_dist']
        
        self.use_mlp = use_mlp
        self.n_heads = emb_dim

        if use_mlp:
            n_heads_pos = 1
            self.rpe_mlp = nn.Sequential(nn.Linear(2, settings['mlp_hidden_dim'], bias=True), nn.ReLU(inplace=True), nn.Linear(settings['mlp_hidden_dim'], emb_dim))
        else:
            n_heads_pos = emb_dim
            self.rpe_mlp = nn.Identity()

        if emb_table_lon is None:
            self.emb_table_lon = RelPositionEmbedder_phys_log(min_dist/radius_earth, max_dist/radius_earth, n_dist, n_heads=n_heads_pos,device=device)
        else:
            self.emb_table_lon = emb_table_lon
        
        if emb_table_lat is None:
            self.emb_table_lat = RelPositionEmbedder_phys_log(min_dist/radius_earth, max_dist/radius_earth, n_dist, n_heads=n_heads_pos,device=device)
        else:
            self.emb_table_lat = emb_table_lat


    def forward(self, d_mat_lon, d_mat_lat, return_emb_idx=False):
        
        a_lon = self.emb_table_lon(d_mat_lon, return_emb_idx=return_emb_idx)
        a_lat = self.emb_table_lat(d_mat_lat, return_emb_idx=return_emb_idx)

        if return_emb_idx:
            idx_lon = a_lon[1]
            a_lon = a_lon[0]
            idx_lat = a_lat[1]
            a_lat = a_lat[0]

        if self.use_mlp:
           rpe = self.rpe_mlp(torch.concat((a_lon, a_lat),dim=-1).squeeze())
        else:
            rpe = (a_lon + a_lat)

        if return_emb_idx:
            return rpe, (idx_lon,idx_lat)
        else:
            return rpe
  

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
    def __init__(self, model_dim, output_dim, RPE_phys, n_heads, logit_scale=False):
        super().__init__()

        self.n_heads = n_heads
        self.head_dim = model_dim // n_heads

        self.RPE_phys = RPE_phys
               
        self.output_projection = nn.Linear(model_dim, output_dim)
        
        if logit_scale:
            self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((n_heads, 1, 1))))
        else:
            self.logit_scale = None

    def forward(self, q, k, v, rel_coords, return_debug=False):
        # batch, sequence length, embedding dimension
        b, t, e = q.shape
        b, s, e = k.shape
        b, s, e = v.shape

        q = q.reshape(b, t, self.n_heads, self.head_dim).permute(0,2,1,3)
        k = k.reshape(b, s, self.n_heads, self.head_dim).permute(0,2,1,3)
        v = v.reshape(b, s, self.n_heads, self.head_dim).permute(0,2,1,3)

        if return_debug:
            rel_pos_bias, rel_pos_bias_idx = self.RPE_phys(rel_coords[0], rel_coords[1], return_emb_idx=return_debug)

        else:
            rel_pos_bias = self.RPE_phys(rel_coords[0], rel_coords[1])

        rel_pos_bias = rel_pos_bias.view(-1,self.n_heads,s,s)

        if return_debug:
            values, att = scaled_dot_product_rpe_swin(q, k, v, rel_pos_bias, self.logit_scale)
        else:
            values = scaled_dot_product_rpe_swin(q, k, v, rel_pos_bias, self.logit_scale)[0]
           
        values = values.permute(0,2,1,3)
        values = values.reshape(b, t, self.head_dim*self.n_heads)

        x = self.output_projection(values)

        if return_debug:
            return x , att, rel_pos_bias, rel_pos_bias_idx
        else:
            return x    
        
    
class nearest_proj_layer(nn.Module):
    def __init__(self, inter_dim):
        super().__init__()
        self.simple_proj = nn.Parameter(torch.ones(inter_dim,1)/inter_dim, requires_grad=True)
        self.inter_dim = inter_dim

    def forward(self, x, d_mat):
        b,s,e = x.shape
        t = d_mat.shape[-2] 

        indices = d_mat.topk(self.inter_dim, largest=False).indices

        if len(d_mat.shape)>2:
            indices = indices.view(b,-1)+s*torch.arange(b).view(b,-1)
            x_nearest = x.view(-1)[indices.view(-1)].view(b,t,self.inter_dim)
        else:
            x_nearest = x.view(b,-1)[:,indices.view(-1)].view(b,t,self.inter_dim)

        return torch.matmul(x_nearest, self.simple_proj)


class nn_layer(nn.Module):
    def __init__(self, nh):
        super().__init__()

        self.nh = nh

    def forward(self, x, d_lon, d_lat):
        b,s,e = x.shape
        d_mat = (d_lon**2 + d_lat**2).sqrt()
        t = d_mat.shape[-2] 

        indices_dist = d_mat.topk(self.nh, largest=False).indices
        indices_dlon = d_lon.abs().topk(self.nh, largest=False).indices
        indices_dlat = d_lat.abs().topk(self.nh, largest=False).indices

        x_nearest = x.view(b,-1)[:,indices_dist.view(-1)].view(b,t,self.nh,1)

        return x_nearest, indices_dist, indices_dlon, indices_dlat
    

class interpolator(nn.Module):
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