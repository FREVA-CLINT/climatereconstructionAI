import torch.nn as nn
import torch
import climatereconstructionai.model.pyramid_step_model as psm
import climatereconstructionai.model.transformer_helpers as helpers
import math

def global_pad(x, padding):
    x = nn.functional.pad(x, (padding,padding,0,0),mode='circular')
    x = nn.functional.pad(x, (0,0,padding,padding),mode='replicate')
    return x


def ICNR(tensor, initializer, upscale_factor=2, *args, **kwargs):
    upscale_factor_squared = upscale_factor * upscale_factor
    
    sub_kernel = torch.empty(tensor.shape[0] // upscale_factor_squared,
                             *tensor.shape[1:])
    sub_kernel = initializer(sub_kernel, *args, **kwargs)
    return sub_kernel.repeat_interleave(upscale_factor_squared, dim=0)


class DepthEncoding(nn.Module):
    def __init__(self, dim, max_period=1e4):
        super().__init__()
        self.dim = dim
        self.max_period=max_period

    def forward(self, depth):
        count = self.dim // 2
        step = torch.arange(count, dtype=depth.dtype,
                            device=depth.device) / count
        encoding = depth.unsqueeze(
            1) * torch.exp(-math.log(self.max_period) * step.unsqueeze(0))
        encoding = torch.cat([torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding
    

class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, with_scale=False):
        super(FeatureWiseAffine, self).__init__()
        self.with_scale = with_scale
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels*(1+self.with_scale))
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.with_scale:
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1)
        return x

class shuffle_upscaling(nn.Module):
    def __init__(self, in_channels, upscale_factor, k_size=3, bias=False, global_padding=False):
        super().__init__()

        out_channels = in_channels * upscale_factor**2
        self.global_padding = global_padding
        self.padding = k_size // 2

        if not global_padding:
            self.conv = nn.Conv2d(in_channels, out_channels, stride=1, padding=k_size//2, kernel_size=k_size, padding_mode='replicate', bias=bias)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, stride=1, kernel_size=k_size, bias=bias)
        
        weight = ICNR(self.conv.weight, initializer=nn.init.kaiming_normal_,upscale_factor=upscale_factor)
        self.conv.weight.data.copy_(weight)
                        
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        if self.global_padding:
            x = global_pad(x, self.padding)
        x = self.conv(x)
        x = self.pixel_shuffle(x)

        return x

class res_net_block(nn.Module):
    def __init__(self, hw, in_channels, out_channels, k_size=3, batch_norm=True, stride=1, groups=1, dropout=0, with_att=False, with_res=True, bias=True, out_activation=True, global_padding=False, depth_embedding_dim=0, instance_norm=False):
        super().__init__()

        self.global_padding = global_padding
        self.padding = k_size // 2

        if not global_padding:
            self.conv1 = nn.Conv2d(in_channels, out_channels, stride=stride, padding=k_size//2, kernel_size=k_size, padding_mode='replicate',groups=groups, bias=bias)
            self.conv2 = nn.Conv2d(out_channels, out_channels, stride=1, padding=k_size//2, kernel_size=k_size, padding_mode='replicate',groups=groups, bias=bias)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, stride=stride, kernel_size=k_size, groups=groups, bias=bias)
            self.conv2 = nn.Conv2d(out_channels, out_channels, stride=1, kernel_size=k_size, groups=groups, bias=bias)

        if with_res:
            self.reduction = nn.Conv2d(in_channels, out_channels, stride=stride, padding=0, kernel_size=1, groups=groups, bias=bias)

        self.with_depth_embedding = True if depth_embedding_dim>0 else False

        if self.with_depth_embedding:
            self.feat_affine = FeatureWiseAffine(depth_embedding_dim, out_channels, with_scale=True)

        self.with_res = with_res

        if instance_norm:
            self.bn1 = nn.InstanceNorm2d(out_channels, affine=True) if batch_norm else nn.Identity()
            self.bn2 = nn.InstanceNorm2d(out_channels, affine=True) if batch_norm else nn.Identity()
        else:
            self.bn1 = nn.BatchNorm2d(out_channels, affine=True) if batch_norm else nn.Identity()
            self.bn2 = nn.BatchNorm2d(out_channels, affine=True) if batch_norm else nn.Identity()

        self.dropout1 = nn.Dropout(dropout) if dropout>0 else nn.Identity()
        self.dropout2 = nn.Dropout(dropout) if dropout>0 else nn.Identity()

        self.activation1 = nn.SiLU() if out_activation else nn.Identity()
        self.activation2 = nn.SiLU() if out_activation else nn.Identity()

        self.with_att = with_att
        if with_att:
            self.att = helpers.ConvSelfAttention(out_channels, hw, n_heads=1)

    def forward(self, x, depth_emb=None):

        if self.with_res:
            x_res = self.reduction(x)

        if self.global_padding:
            x = global_pad(x, self.padding)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.dropout1(x)

        if self.with_depth_embedding:
            x = self.feat_affine(x, depth_emb)

        if self.global_padding:
            x = global_pad(x, self.padding)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.with_res:
            x = x + x_res

        x = self.activation2(x)
        x = self.dropout2(x)

        if self.with_att:
            x = self.att(x)
        return x
        

class res_blocks(nn.Module): 
    def __init__(self, hw, n_blocks, in_channels, out_channels, k_size=3, batch_norm=True, groups=1, dropout=0, with_att=False, with_res=True, out_activation=True, bias=True, global_padding=False, factor=2, down_method='max', depth_embedding_dim=0,instance_norm=True):
        super().__init__()

        self.res_net_blocks = nn.ModuleList()

        self.res_net_blocks.append(res_net_block(hw, in_channels, out_channels, k_size=k_size, batch_norm=batch_norm, groups=groups, dropout=dropout, with_att=with_att, with_res=with_res, out_activation=out_activation, bias=bias, global_padding=global_padding, depth_embedding_dim=depth_embedding_dim, instance_norm=instance_norm))
        
        for _ in range(n_blocks-1):
            self.res_net_blocks.append(res_net_block(hw, out_channels, out_channels, k_size=k_size, batch_norm=batch_norm, groups=groups, dropout=dropout, with_att=False, with_res=with_res, out_activation=out_activation, bias=bias, global_padding=global_padding, depth_embedding_dim=depth_embedding_dim, instance_norm=instance_norm))

        if down_method=='max':
            self.pool = nn.MaxPool2d(kernel_size=factor) if factor>1 else nn.Identity()

        elif down_method=='avg':
            self.pool = nn.AvgPool2d(kernel_size=factor) if factor>1 else nn.Identity()

    def forward(self, x, depth_emb=None):
        for layer in self.res_net_blocks:
            x = layer(x, depth_emb=depth_emb)

        return self.pool(x), x

class encoder(nn.Module): 
    def __init__(self, hw_in, factor, n_levels, n_blocks, u_net_channels, in_channels, k_size=3, k_size_in=7, batch_norm=True, n_groups=1, dropout=0, bias=True, global_padding=False, initial_res=False,down_method='max', depth_embedding_dim=0,instance_norm=True):
        super().__init__()

        hw_in = torch.tensor(hw_in)
        
        self.layer_configs = []
        
        self.layers = nn.ModuleList()
        for n in range(n_levels):
            if n==0:
                in_channels_block = in_channels
                out_channels_block = u_net_channels * n_groups 
                with_res = initial_res
                groups = n_groups
                k_size = k_size_in
            else:
                in_channels_block = out_channels_block
                out_channels_block = u_net_channels*(2**(n))
                groups=1
                with_res = True
                k_size = k_size

            factor_level = 1 if n == n_levels-1 else factor
            hw = hw_in/(factor**(n))

            self.layer_configs.append({'in_channels': in_channels_block,
                                 'out_channels': out_channels_block,
                                 'hw': hw})
            

            self.layers.append(res_blocks(hw, n_blocks, in_channels_block, out_channels_block, k_size=k_size, batch_norm=batch_norm, groups=groups, with_res=with_res, factor=factor_level, dropout=dropout, bias=bias, global_padding=global_padding, down_method=down_method, depth_embedding_dim=depth_embedding_dim, instance_norm=instance_norm))

    
    def forward(self, x, depth_emb=None):
        outputs = []

        for layer in self.layers:
            x, x_skip = layer(x, depth_emb=depth_emb)
            outputs.append(x_skip)
        return x, outputs

class decoder_block_shuffle(nn.Module):
    def __init__(self, hw, n_blocks, in_channels, out_channels, skip_channels, k_size=3, dropout=0, upscale_factor=2, bias=True, global_padding=False, out_activation=True, groups=1, with_res=True, depth_embedding_dim=0, instance_norm=True):
        super().__init__()

        self.skip_channels = skip_channels
        self.up = shuffle_upscaling(in_channels, upscale_factor, k_size=k_size, bias=False, global_padding=global_padding) if upscale_factor > 1 else nn.Identity()
        
        in_channels = in_channels + skip_channels

        self.res_blocks = res_blocks(hw, n_blocks, in_channels, out_channels, k_size=k_size, batch_norm=False, groups=groups, dropout=dropout, with_res=with_res, bias=bias, global_padding=global_padding, out_activation=out_activation, factor=1, depth_embedding_dim=depth_embedding_dim, instance_norm=instance_norm)

    def forward(self, x, skip_channels=None, depth_emb=None):

        x = self.up(x)

        if self.skip_channels>0:
            x = torch.concat((x, skip_channels), dim=1)
      
        x = self.res_blocks(x, depth_emb=depth_emb)[0]

        return x

class decoder(nn.Module):
    def __init__(self, layer_configs, factor, n_blocks, out_channels, global_upscale_factor=1, k_size=3, dropout=0, n_groups=1, bias=True, global_padding=False, depth_embedding_dim=0, instance_norm=True):
        super().__init__()
        # define total number of layers, first n_levels-1 with skip, others not
        #watch out for n_groups in last layer before last

        #check dims!
        n_out_blocks = int(torch.max(torch.tensor([math.log2(global_upscale_factor), torch.tensor(1)])))
        self.skip_blocks = len(layer_configs)-1
        n_layers = n_out_blocks + self.skip_blocks

        self.decoder_blocks = nn.ModuleList()
        hw_in = layer_configs[-1]['hw']

        for n in range(n_layers):
            groups=1
            upscale_factor = factor

            if n < self.skip_blocks:
                in_channels_block = layer_configs[self.skip_blocks-n]['out_channels']
                out_channels_block = layer_configs[self.skip_blocks-n]['in_channels']
                skip_channels = layer_configs[self.skip_blocks-(n+1)]['out_channels']

                if n == n_layers-2:
                    out_channels_block = n_groups * (out_channels_block // n_groups + out_channels_block % n_groups)

                out_activation=True
      
            else:
                skip_channels = 0

                if n == n_layers-2:
                    in_channels_block = out_channels_block
                    out_channels_block = n_groups * (out_channels_block // n_groups + out_channels_block % n_groups)

                elif n == n_layers-1:
                    in_channels_block = out_channels_block
                    out_channels_block = out_channels
                    out_activation = False
                    groups=n_groups
                    if global_upscale_factor==1:
                        upscale_factor=1
                
                else:
                    in_channels_block = out_channels_block

            hw = hw_in/(factor**(n-1))
            self.decoder_blocks.append(decoder_block_shuffle(hw, n_blocks, in_channels_block, out_channels_block, skip_channels, k_size=k_size, dropout=dropout, bias=bias, global_padding=global_padding, upscale_factor=upscale_factor, groups=groups, out_activation=out_activation, depth_embedding_dim=depth_embedding_dim, instance_norm=instance_norm))      
        
    def forward(self, x, skip_channels, depth_emb=None):

        for k, layer in enumerate(self.decoder_blocks):
            if k < self.skip_blocks:
                x_skip = skip_channels[-(k+2)]  
                x = layer(x, x_skip, depth_emb=depth_emb)
            else:
                x = layer(x, depth_emb=depth_emb)
        return x

class mid(nn.Module):
    def __init__(self, hw, n_blocks, channels, k_size=3,  dropout=0, with_att=False, bias=True, global_padding=False, depth_embedding_dim=0):
        super().__init__()

        self.res_blocks = res_blocks(hw, n_blocks, channels, channels, k_size=k_size, batch_norm=False, groups=1,  dropout=dropout, with_att=with_att, bias=bias, global_padding=global_padding, factor=1, depth_embedding_dim=depth_embedding_dim)

    def forward(self, x, depth_emb=None):

        x = self.res_blocks(x, depth_emb=depth_emb)[0]

        return x
    

class out_net(nn.Module):
    def __init__(self, res_indices, hw_in, hw_out, res_mode=True, global_padding=False):
        super().__init__()

        self.res_indices_rhs = res_indices
        self.res_indices_lhs = torch.arange(len(res_indices))

        self.global_residual = True
        scale_factor = hw_out[0] / hw_in[1]
        self.res_mode = res_mode
        self.scale_factor = scale_factor

        if res_mode == 'core_inter':
            pass

        elif res_mode=='core_train':
            total_stride = int(1/scale_factor)
            stride1 = total_stride // 2 if total_stride > 2 else 2
            stride2 = stride1 // 2 if stride1 > 2 else 1
            self.res_interpolate = nn.Sequential(res_net_block(hw_out, len(res_indices), len(res_indices), k_size=5, batch_norm=False, stride=stride1, groups=len(res_indices), dropout=0, with_att=False, with_res=False, bias=False, out_activation=False, global_padding=global_padding),
                                                 res_net_block(hw_out, len(res_indices), len(res_indices), k_size=5, batch_norm=False, stride=stride2, groups=len(res_indices), dropout=0, with_att=False, with_res=False, bias=False, out_activation=False, global_padding=global_padding))

        elif res_mode=='core_train_res':
            if scale_factor>1:
                total_stride = int(1/scale_factor)
                stride1 = total_stride // 2 if total_stride > 2 else 2
                stride2 = stride1 // 2 if stride1 > 2 else 1
            else:
                stride1 = stride2 = 1
            self.res_interpolate = nn.Sequential(res_net_block(hw_out, len(res_indices), len(res_indices), k_size=5, batch_norm=False, stride=stride1, groups=len(res_indices), dropout=0, with_att=False, with_res=True, bias=False, out_activation=False, global_padding=global_padding),
                                                 res_net_block(hw_out, len(res_indices), len(res_indices), k_size=5, batch_norm=False, stride=stride2, groups=len(res_indices), dropout=0, with_att=False, with_res=True, bias=False, out_activation=False, global_padding=global_padding))

        else:
            self.res_interpolate = nn.Identity()

            self.global_residual = False
        
        
    def forward(self, x, x_res):
        
        if self.res_mode == 'core_inter':
            if self.scale_factor>1:
                x_res = nn.functional.interpolate(x_res[:,self.res_indices_rhs,:,:],scale_factor=self.scale_factor, mode = 'bicubic', align_corners=True)
            else:
                x_res = x_res[:,self.res_indices_rhs,:,:]
        else:
            x_res = self.res_interpolate(x_res[:,self.res_indices_rhs,:,:])
        
        if self.global_residual:
            x[:,self.res_indices_lhs] = x[:,self.res_indices_lhs] + x_res
        return x


class ResUNet(nn.Module): 
    def __init__(self, hw_in, hw_out, n_levels, factor, n_res_blocks, model_dim_unet, in_channels, out_channels, res_mode, res_indices, batch_norm=True, k_size=3, in_groups=1, out_groups=1, dropout=0, bias=True, global_padding=False, mid_att=False, initial_res=True, down_method="max",depth_embedding_dim=0, instance_norm=True):
        super().__init__()

        global_upscale_factor = int(torch.tensor([(hw_out[0]) // hw_in[0], 1]).max())

        self.encoder = encoder(hw_in, factor, n_levels, n_res_blocks, model_dim_unet, in_channels, k_size, 7, batch_norm=batch_norm, n_groups=in_groups, dropout=dropout, bias=bias, global_padding=global_padding, initial_res=initial_res, down_method=down_method,depth_embedding_dim=depth_embedding_dim,instance_norm=instance_norm)
        self.decoder = decoder(self.encoder.layer_configs, factor, n_res_blocks, out_channels, global_upscale_factor=global_upscale_factor, k_size=k_size, dropout=dropout, n_groups=out_groups, bias=bias, global_padding=global_padding,depth_embedding_dim=depth_embedding_dim,instance_norm=instance_norm)

        self.out_net = out_net(res_indices, hw_in, hw_out, res_mode=res_mode, global_padding=global_padding)
  
        hw_mid = torch.tensor(hw_in)// (factor**(n_levels-1))
        self.mid = mid(hw_mid, n_res_blocks, model_dim_unet*(2**(n_levels-1)), with_att=mid_att, bias=bias, global_padding=global_padding,depth_embedding_dim=depth_embedding_dim)

        if depth_embedding_dim:
            self.depth_level_mlp = nn.Sequential(
                    DepthEncoding(model_dim_unet),
                    nn.Linear(model_dim_unet, model_dim_unet * 4),
                    nn.SiLU(),
                    nn.Linear(model_dim_unet * 4, model_dim_unet)
                )
        else:
            self.depth_level_mlp = None
        
    def forward(self, x, depth=None):
        d = self.depth_level_mlp(depth) if self.depth_level_mlp is not None else None

        x_res = x
        x, layer_outputs = self.encoder(x, depth_emb=d)
        x = self.mid(x, depth_emb=d)
        x = self.decoder(x, layer_outputs, depth_emb=d)
        x = self.out_net(x, x_res)

        return {'x': x}


class core_ResUNet(psm.pyramid_step_model): 
    def __init__(self, model_settings, model_dir=None, eval=False):
        super().__init__(model_settings, model_dir=model_dir, eval=eval)
        
        model_settings = self.model_settings

        input_dim = len(model_settings["variables_source"])
        
        if 'output_dim_core' not in model_settings.keys():
            output_dim = len(model_settings["variables_target"]) - int(model_settings["calc_vort"]) * int('vort' in model_settings["variables_target"])
        else:
            output_dim = model_settings['output_dim_core']

        dropout = model_settings['dropout']

        n_blocks = model_settings['n_blocks_core']
        model_dim_core = model_settings['model_dim_core']
        depth = model_settings["depth_core"]
        batch_norm = model_settings["batch_norm"]
        instance_norm = model_settings["instance_norm"]
        mid_att = model_settings["mid_att"] if "mid_att" in model_settings.keys() else False
        full_res = model_settings["full_res"] if "full_res" in model_settings.keys() else False

        with_depth_embedding = model_settings["with_depth_embedding"] if "with_depth_embedding" in model_settings.keys() else False
        depth_embedding_dim = model_settings['model_dim_core'] if with_depth_embedding else 0

        dropout = 0 if 'dropout' not in model_settings.keys() else model_settings['dropout']
        grouped = False if 'grouped' not in model_settings.keys() else model_settings['grouped']

        self.time_dim= False

        if model_settings['gauss'] or model_settings['poly']:
            output_dim *=2

        if grouped:
            in_groups = input_dim
            out_groups = output_dim
        else:
            in_groups = out_groups = 1

        hw_in = model_settings['n_regular'][0]
        hw_out = model_settings['n_regular'][1]

        #input_stride = 1 if 'input_stride' not in model_settings.keys() else model_settings['input_stride']
        bias = True if 'bias' not in model_settings.keys() else model_settings['bias']

        res_indices = []
        for var_target in self.model_settings['variables_target']:
            var_in_source = [k for k, var_source in enumerate(self.model_settings['variables_source']) if var_source==var_target]
            if len(var_in_source)==1:
                res_indices.append(var_in_source[0])

        global_padding = True if model_settings['model_type']=="global" else False
        factor = model_settings['factor'] if "factor" in model_settings.keys() else 2
        initial_res = model_settings['initial_res'] if "initial_res" in model_settings.keys() else False
        down_method = model_settings['down_method'] if "down_method" in model_settings.keys() else "max"

        self.core_model = ResUNet(hw_in, hw_out, depth, factor, n_blocks, model_dim_core, input_dim, output_dim, model_settings['res_mode'], res_indices, batch_norm=batch_norm, in_groups=in_groups, out_groups=out_groups, dropout=dropout, bias=bias, global_padding=global_padding, mid_att=mid_att, initial_res=initial_res, down_method=down_method, depth_embedding_dim=depth_embedding_dim,instance_norm=instance_norm)

        if "pretrained_path" in self.model_settings.keys():
            self.check_pretrained(log_dir_check=self.model_settings['pretrained_path'])
        elif self.eval_mode:
            self.check_pretrained(log_dir_check=self.log_dir)