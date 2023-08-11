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


def scaled_dot_product_rpe_swin(q, k, v, b=None, logit_scale=None):
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

    attn = F.softmax(attn, dim=-1)

    values = torch.matmul(attn, v)

    return values, attn


class PositionEmbedder_phys(nn.Module):

    def __init__(self, min_pos_phys, max_pos_phys, n_pos_emb, n_heads=10,device='cpu'):
        super().__init__()
        self.max_pos_phys = max_pos_phys
        self.min_pos_phys = min_pos_phys
        self.n_pos_emb = n_pos_emb
        
        self.embeddings_table = nn.Parameter(torch.Tensor(n_pos_emb + 1, n_heads))
        nn.init.xavier_uniform_(self.embeddings_table)


    def forward(self, d_mat):

        d_mat_pos = self.n_pos_emb * (d_mat -self.min_pos_phys) / (self.max_pos_phys - self.min_pos_phys)

        d_mat_clipped = torch.clamp(d_mat_pos, 0, self.n_pos_emb)
        
        embeddings = self.embeddings_table[d_mat_clipped.long()]

        return embeddings, d_mat_clipped.long()


class PositionEmbedder_phys_log(nn.Module):

    def __init__(self, min_pos_phys, max_pos_phys, n_pos_emb, n_heads=10, device='cpu'):
        super().__init__()
        self.max_pos_phys = max_pos_phys
        self.min_pos_phys = min_pos_phys
        self.n_pos_emb = n_pos_emb

        self.embeddings_table = nn.Parameter(torch.Tensor(n_pos_emb + 1, n_heads))
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
    def __init__(self, mdoel_dim, hidden_dim, device='cpu',conv_coordinates=False):
        super().__init__()

        self.conv_coordinates = conv_coordinates

        self.rpe_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim, bias=False),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            nn.Linear(hidden_dim, mdoel_dim, bias=False),
            nn.Sigmoid())
       

    def forward(self, d_mat_lon, d_mat_lat):
        
        rpe = self.rpe_mlp(torch.concat((d_mat_lon.unsqueeze(dim=-1), d_mat_lat.unsqueeze(dim=-1)),dim=-1).squeeze())
   
        return rpe

class RelativePositionEmbedder_mlp(nn.Module):
    def __init__(self, settings, emb_table_lon=None, emb_table_lat=None, device='cpu'):
        super().__init__()
 
        emb_dim=settings['emb_dim']

        self.rpe_mlp = nn.Sequential(
            nn.Linear(2, settings['mlp_hidden_dim'], bias=False),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            nn.Linear(settings['mlp_hidden_dim'], emb_dim, bias=False),
            nn.LeakyReLU(inplace=True, negative_slope=0.2))

    def forward(self, d_mat_lon, d_mat_lat):
        
        d_mat_lon = conv_coordinates_inv(d_mat_lon)
        d_mat_lat = conv_coordinates_inv(d_mat_lat)
                      
        rpe = self.rpe_mlp(torch.concat((d_mat_lon.unsqueeze(dim=-1), d_mat_lat.unsqueeze(dim=-1)),dim=-1).squeeze())
    
        return rpe
  
def conv_coordinates(coords):
    sign = torch.sign(coords)
    coords_log_m = torch.log10(1000.*6371.*(coords.abs()))
    coords_log_m = torch.clamp(coords_log_m, min=0)
    return sign * coords_log_m

def conv_coordinates_inv(coords, epsilon=1e-10):
    sign = torch.sign(coords)
    coords = sign*torch.log10(1/(coords.abs()+epsilon))   
    return sign * coords

class RelativePositionEmbedder_par(nn.Module):
    def __init__(self, settings, emb_table_lon=None, emb_table_lat=None, device='cpu'):
        super().__init__()

        max_dist=settings['max_dist']
        n_dist=settings['n_dist']
        emb_dim=settings['emb_dim']
        min_dist=settings['min_dist']
        
        self.n_heads = emb_dim

        if emb_table_lon is None:
            self.emb_table_lon = RelPositionEmbedder_phys_log(min_dist/radius_earth, max_dist/radius_earth, n_dist, n_heads=emb_dim,device=device)
        else:
            self.emb_table_lon = emb_table_lon
        
        if emb_table_lat is None:
            self.emb_table_lat = RelPositionEmbedder_phys_log(min_dist/radius_earth, max_dist/radius_earth, n_dist, n_heads=emb_dim,device=device)
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

        rpe = a_lon + a_lat

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
    def __init__(self, model_dim, output_dim, n_heads, rel_pos=None, logit_scale=False, qkv_proj=False):
        super().__init__()

        self.n_heads = n_heads
        self.head_dim = model_dim // n_heads

        self.rel_pos = rel_pos
               
        self.output_projection = nn.Linear(model_dim, output_dim)
        
        if logit_scale:
            self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((n_heads, 1, 1))))
        else:
            self.logit_scale = None
        
        if qkv_proj:
            self.qkv_projection = nn.ModuleList([nn.Linear(model_dim, model_dim, bias=False)]*3)
        else:
            self.qkv_projection = nn.ModuleList([nn.Identity()]*3)


    def forward(self, q, k, v, rel_coords=None, return_debug=False):
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

        if self.rel_pos is not None:
            rel_pos_bias = self.RPE_phys(rel_coords[0], rel_coords[1])

            if len(rel_pos_bias.shape)>3:
                rel_pos_bias = rel_pos_bias.permute(0,3,1,2)
            else:
                rel_pos_bias = rel_pos_bias.permute(-1,0,1)
        else:
            rel_pos_bias=None

        if return_debug:
            values, att = scaled_dot_product_rpe_swin(q, k, v, rel_pos_bias, self.logit_scale)
        else:
            values = scaled_dot_product_rpe_swin(q, k, v, rel_pos_bias, self.logit_scale)[0]
           
        values = values.permute(0,2,1,3)
        values = values.reshape(bv, t, self.head_dim*self.n_heads)

        x = self.output_projection(values)

        if return_debug:
            return x , att, rel_pos_bias
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

        if self.nh==-1:
            self.nh = d_mat.shape[1]
        indices_dist = d_mat.topk(self.nh, largest=False).indices
        indices_dlon = d_lon.abs().topk(self.nh, largest=False).indices
        indices_dlat = d_lat.abs().topk(self.nh, largest=False).indices

        if len(indices_dlon.shape)>2:
            x_nearest = torch.gather(x.repeat(1,1,self.nh),dim=1,index=indices_dlon).view(b,t,self.nh,1)
        else:
            x_nearest = x.view(b,-1)[:,indices_dist.view(-1)].view(b,t,self.nh,1)

        return x_nearest.float(), indices_dist, indices_dlon, indices_dlat
    

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



class interpolator_iwd(nn.Module):
    def __init__(self, nh):
        super().__init__()
        self.nh = nh

    def forward(self, x, coords_rel):

        coords_dist = (coords_rel[0]**2 + coords_rel[1]**2).sqrt()
        dist_abs, indices = torch.topk(coords_dist, self.nh, dim=1, largest=False)

        b, nh, t = dist_abs.shape
        dist_abs = 1/(dist_abs+1e-15)**2
        dist_abs = (dist_abs.view(t,b,nh)/dist_abs.sum(dim=2)).view(b,t,nh)

        x = torch.gather(x.repeat(1,1,self.nh),dim=1,index=indices.view(b,t,nh)).view(b,t,self.nh)

        x = (x*dist_abs).sum(dim=2)

        return x.unsqueeze(dim=-1)


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, dim9=False, device="cpu"):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)   # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    device = torch.device(device)

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature