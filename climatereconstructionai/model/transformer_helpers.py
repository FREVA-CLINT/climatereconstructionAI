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


    def forward(self, coord):

        coord_pos = self.n_pos_emb * (coord -self.min_pos_phys) / (self.max_pos_phys - self.min_pos_phys)

        coord_pos_clipped = torch.clamp(coord_pos, 0, self.n_pos_emb)
        
        embeddings = self.embeddings_table[coord_pos_clipped.long()]

        return embeddings, coord_pos_clipped.long()


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
    sign = torch.sign(coords)
    coords_log_m = torch.log10(1000.*6371.*(coords.abs()))
    coords_log_m = torch.clamp(coords_log_m, min=0)
    return sign * coords_log_m

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
            self.qkv_projection = nn.ModuleList([nn.Linear(model_dim, model_dim, bias=False)]*3)
        else:
            self.qkv_projection = nn.ModuleList([nn.Identity()]*3)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()


    def forward(self, q, k, v, rel_pos_bias=None, return_debug=False):
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

        values, att = scaled_dot_product_rpe_swin(q, k, v, rel_pos_bias, self.logit_scale)

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