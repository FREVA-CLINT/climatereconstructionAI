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
    def __init__(self, hw, in_channels, out_channels, k_size=3, batch_norm=True, stride=1, groups=1, dropout=0, with_att=False, with_res=True, bias=True, out_activation=True, global_padding=False):
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

        self.with_res = with_res

        self.bn1 = nn.BatchNorm2d(out_channels, affine=True) if batch_norm else nn.Identity()
        self.bn2 = nn.BatchNorm2d(out_channels, affine=True) if batch_norm else nn.Identity()

        self.dropout1 = nn.Dropout(dropout) if dropout>0 else nn.Identity()
        self.dropout2 = nn.Dropout(dropout) if dropout>0 else nn.Identity()

        self.activation1 = nn.SiLU() if out_activation else nn.Identity()
        self.activation2 = nn.SiLU() if out_activation else nn.Identity()

        self.with_att = with_att
        if with_att:
            self.att = helpers.ConvSelfAttention(out_channels, hw, n_heads=1)

    def forward(self, x):

        if self.with_res:
            x_res = self.reduction(x)

        if self.global_padding:
            x = global_pad(x, self.padding)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.dropout1(x)

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
    def __init__(self, hw, n_blocks, in_channels, out_channels, k_size=3, with_reduction=True, batch_norm=True, groups=1, dropout=0, with_att=False, with_res=True, out_activation=True, bias=True, global_padding=False):
        super().__init__()

        self.res_net_blocks = nn.ModuleList()

        self.res_net_blocks.append(res_net_block(hw, in_channels, out_channels, k_size=k_size, batch_norm=batch_norm, groups=groups, dropout=dropout, with_att=with_att, with_res=with_res, out_activation=out_activation, bias=bias, global_padding=global_padding))
        
        for _ in range(n_blocks-1):
            self.res_net_blocks.append(res_net_block(hw, out_channels, out_channels, k_size=k_size, batch_norm=batch_norm, groups=groups, dropout=dropout, with_att=False, with_res=with_res, out_activation=out_activation, bias=bias, global_padding=global_padding))

        self.max_pool = nn.MaxPool2d(kernel_size=2) if with_reduction else nn.Identity()

    def forward(self,x):
        for layer in self.res_net_blocks:
            x = layer(x)

        return self.max_pool(x), x

class encoder(nn.Module): 
    def __init__(self, hw_in, n_levels, n_blocks, u_net_channels, in_channels, k_size=3, k_size_in=7, input_stride=1, batch_norm=True, n_groups=1, dropout=0, bias=True, global_padding=False):
        super().__init__()

        hw_in = torch.tensor(hw_in)
        u_net_channels_g = u_net_channels * n_groups
        self.in_layer = res_net_block(hw_in, in_channels, u_net_channels_g, stride=input_stride, k_size=k_size_in, batch_norm=batch_norm, groups=n_groups, dropout=dropout, with_res=False, bias=bias, global_padding=global_padding)
        
        self.layer_configs = []
        
        self.layers = nn.ModuleList()
        for n in range(n_levels):
            if n==0:
                in_channels_block = u_net_channels_g
                out_channels_block = u_net_channels 
            else:
                in_channels_block = out_channels_block
                out_channels_block = u_net_channels*(2**(n))

            reduction = False if n == n_levels-1 else True
            hw = hw_in/(input_stride*2**(n))

            self.layer_configs.append({'in_channels': in_channels_block,
                                 'out_channels': out_channels_block,
                                 'hw': hw})
            
            self.layers.append(res_blocks(hw, n_blocks, in_channels_block, out_channels_block, k_size=k_size, batch_norm=batch_norm, groups=1, with_reduction=reduction, dropout=dropout, bias=bias, global_padding=global_padding))

    
    def forward(self, x):
        outputs = []
        x = self.in_layer(x)

        for layer in self.layers:
            x, x_skip = layer(x)
            outputs.append(x_skip)
        return x, outputs

class decoder_block_shuffle(nn.Module):
    def __init__(self, hw, n_blocks, in_channels, out_channels, k_size=3, dropout=0, upscale_factor=2, bias=True, global_padding=False, out_activation=True, groups=1, with_res=True, with_skip=True):
        super().__init__()

        self.up = shuffle_upscaling(in_channels, upscale_factor, k_size=k_size, bias=False, global_padding=global_padding) if upscale_factor > 1 else nn.Identity()
        
        in_channels = in_channels + in_channels//2 if with_skip else in_channels

        self.res_blocks = res_blocks(hw, n_blocks, in_channels, out_channels, k_size=k_size, batch_norm=False, groups=groups, with_reduction=False, dropout=dropout, with_res=with_res, bias=bias, global_padding=global_padding, out_activation=out_activation)

    def forward(self, x, skip_channels=None):

        x = self.up(x)

        if not skip_channels is None:
            x = torch.concat((x, skip_channels), dim=1)
      
        x = self.res_blocks(x)[0]

        return x

class decoder(nn.Module):
    def __init__(self, layer_configs, n_blocks, out_channels, global_upscale_factor=1, k_size=3, dropout=0, n_groups=1, bias=True, global_padding=False):
        super().__init__()
        # define total number of layers, first n_levels-1 with skip, others not
        #watch out for n_groups in last layer before last

        #check dims!
        n_out_blocks = int(torch.max(torch.tensor([math.log2(global_upscale_factor), torch.tensor(1)])))
        self.skip_blocks = len(layer_configs) - 1
        n_layers = n_out_blocks + self.skip_blocks

        self.decoder_blocks = nn.ModuleList()
        hw_in = layer_configs[-1]['hw']

        for n in range(n_layers):
            groups=1
            upscale_factor = 2

            if n < self.skip_blocks:
                in_channels_block = layer_configs[(self.skip_blocks)-n]['out_channels']
                out_channels_block = layer_configs[(self.skip_blocks)-n]['in_channels']

                if n == n_layers-2:
                    out_channels_block = n_groups * (out_channels_block // n_groups + out_channels_block % n_groups)

                out_activation=True
                with_skip=True
            else:
                with_skip = False

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

            hw = hw_in/(2**(n-1))
            self.decoder_blocks.append(decoder_block_shuffle(hw, n_blocks, in_channels_block, out_channels_block, k_size=k_size, dropout=dropout, bias=bias, global_padding=global_padding, upscale_factor=upscale_factor, groups=groups, with_skip=with_skip, out_activation=out_activation))      
        
    def forward(self, x, skip_channels):

        for k, layer in enumerate(self.decoder_blocks):
            if k < self.skip_blocks:
                x_skip = skip_channels[-(k+2)]  
                x = layer(x, x_skip)
            else:
                x = layer(x)
        return x

class mid(nn.Module):
    def __init__(self, hw, n_blocks, channels, k_size=3,  dropout=0, with_att=False, bias=True, global_padding=False):
        super().__init__()

        self.res_blocks = res_blocks(hw, n_blocks, channels, channels, k_size=k_size, batch_norm=False, groups=1, with_reduction=False, dropout=dropout, with_att=with_att, bias=bias, global_padding=global_padding)

    def forward(self, x):

        x = self.res_blocks(x)[0]

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
            total_stride = int(1/scale_factor)
            stride1 = total_stride // 2 if total_stride > 2 else 2
            stride2 = stride1 // 2 if stride1 > 2 else 1
            self.res_interpolate = nn.Sequential(res_net_block(hw_out, len(res_indices), len(res_indices), k_size=5, batch_norm=False, stride=stride1, groups=len(res_indices), dropout=0, with_att=False, with_res=True, bias=False, out_activation=False, global_padding=global_padding),
                                                 res_net_block(hw_out, len(res_indices), len(res_indices), k_size=5, batch_norm=False, stride=stride2, groups=len(res_indices), dropout=0, with_att=False, with_res=True, bias=False, out_activation=False, global_padding=global_padding))

        else:
            self.res_interpolate = nn.Identity()

            self.global_residual = False
        
        
    def forward(self, x, x_res):
        
        if self.res_mode == 'core_inter':
            x_res = nn.functional.interpolate(x_res[:,self.res_indices_rhs,:,:],scale_factor=self.scale_factor, mode = 'bicubic', align_corners=True)
        else:
            x_res = self.res_interpolate(x_res[:,self.res_indices_rhs,:,:])
        
        if self.global_residual:
            x[:,self.res_indices_lhs] = x[:,self.res_indices_lhs] + x_res
        return x


class ResUNet(nn.Module): 
    def __init__(self, hw_in, hw_out, n_levels, n_res_blocks, model_dim_unet, in_channels, out_channels, res_mode, res_indices, input_stride, batch_norm=True, k_size=3, in_groups=1, out_groups=1, dropout=0, bias=True, global_padding=False):
        super().__init__()

        global_upscale_factor = int(torch.tensor([(hw_out[0]*input_stride) // hw_in[0], 1]).max())

        self.encoder = encoder(hw_in, n_levels, n_res_blocks, model_dim_unet, in_channels, k_size, 7, input_stride, batch_norm=batch_norm, n_groups=in_groups, dropout=dropout, bias=bias, global_padding=global_padding)
        self.decoder = decoder(self.encoder.layer_configs, n_res_blocks, out_channels, global_upscale_factor=global_upscale_factor, k_size=k_size, dropout=dropout, n_groups=out_groups, bias=bias, global_padding=global_padding)

        self.out_net = out_net(res_indices, hw_in, hw_out, res_mode=res_mode, global_padding=global_padding)
  
        hw_mid = torch.tensor(hw_in)// (input_stride*2**(n_levels-1))
        self.mid = mid(hw_mid, n_res_blocks, model_dim_unet*(2**(n_levels-1)), with_att=True, bias=bias, global_padding=global_padding)

    def forward(self, x):

        x_res = x
        x, layer_outputs = self.encoder(x)
        x = self.mid(x)
        x = self.decoder(x, layer_outputs)
        x = self.out_net(x, x_res)

        return x


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

        input_stride = 1 if 'input_stride' not in model_settings.keys() else model_settings['input_stride']
        bias = True if 'bias' not in model_settings.keys() else model_settings['bias']

        res_indices = []
        for var_target in self.model_settings['variables_target']:
            var_in_source = [k for k, var_source in enumerate(self.model_settings['variables_source']) if var_source==var_target]
            if len(var_in_source)==1:
                res_indices.append(var_in_source[0])

        global_padding = True if model_settings['km']==False else False

        self.core_model = ResUNet(hw_in, hw_out, depth, n_blocks, model_dim_core, input_dim, output_dim, model_settings['res_mode'], res_indices, input_stride, batch_norm=batch_norm, in_groups=in_groups, out_groups=out_groups, dropout=dropout, bias=bias, global_padding=global_padding)

        if "pretrained_path" in self.model_settings.keys():
            self.check_pretrained(log_dir_check=self.model_settings['pretrained_path'])
        elif self.eval_mode:
            self.check_pretrained(log_dir_check=self.log_dir)