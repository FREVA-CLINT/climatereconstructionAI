import torch.nn as nn
import torch
import climatereconstructionai.model.pyramid_step_model as psm
import climatereconstructionai.model.transformer_helpers as helpers
import math
    
  
class res_net_block(nn.Module):
    def __init__(self, hw, in_channels, out_channels, k_size=3, batch_norm=True, with_reduction=False, groups=1, dropout=0, with_att=False, with_res=True):
        super().__init__()

        padding = k_size // 2
        stride = 2 if with_reduction else 1

        self.conv1 = nn.Conv2d(in_channels, out_channels, stride=stride, padding=padding, kernel_size=k_size, padding_mode='replicate',groups=groups, bias=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, stride=1, padding=padding, kernel_size=k_size, padding_mode='replicate',groups=groups, bias=True)

        if with_res:
            self.reduction = nn.Conv2d(in_channels, out_channels, stride=stride, padding=0, kernel_size=1,groups=groups, bias=True)
        self.with_res = with_res

        self.bn1 = nn.BatchNorm2d(out_channels, affine=True) if batch_norm else nn.Identity()
        self.bn2 = nn.BatchNorm2d(out_channels, affine=True) if batch_norm else nn.Identity()

        self.dropout1 = nn.Dropout(dropout) if dropout>0 else nn.Identity()
        self.dropout2 = nn.Dropout(dropout) if dropout>0 else nn.Identity()

        self.activation1 = nn.SiLU()
        self.activation2 = nn.SiLU()

        self.with_att = with_att
        if with_att:
            self.att = helpers.ConvSelfAttention(out_channels, hw, n_heads=4)

    def forward(self, x):
        if self.with_res:
            x_res = self.reduction(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.dropout1(x)
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
    def __init__(self, hw, n_blocks, in_channels, out_channels, k_size=3, with_reduction=True, batch_norm=True, groups=1, dropout=0, with_att=False, with_res=True):
        super().__init__()

        self.res_net_blocks = nn.ModuleList()

        self.res_net_blocks.append(res_net_block(hw, in_channels, out_channels, k_size=k_size, batch_norm=batch_norm, groups=groups, dropout=dropout, with_att=with_att, with_res=with_res))
        
        for _ in range(n_blocks-1):
            self.res_net_blocks.append(res_net_block(hw, out_channels, out_channels, k_size=k_size, batch_norm=batch_norm, groups=groups, dropout=dropout, with_att=False, with_res=with_res))

        self.max_pool = nn.MaxPool2d(kernel_size=2) if with_reduction else nn.Identity()

    def forward(self,x):
        for layer in self.res_net_blocks:
            x = layer(x)

        return self.max_pool(x), x

class encoder(nn.Module): 
    def __init__(self, hw_in, n_levels, n_blocks, u_net_channels, in_channels, k_size=3, k_size_in=7, batch_norm=True, n_groups=1, dropout=0):
        super().__init__()

        u_net_channels_g = u_net_channels*n_groups
        self.in_layer = res_net_block(hw_in, in_channels, u_net_channels_g, k_size=k_size_in, batch_norm=batch_norm, groups=n_groups, dropout=dropout, with_res=False)

        self.layers = nn.ModuleList()
        for n in range(n_levels):
            if n==0:
                in_channels_block = u_net_channels_g
                out_channels_block = u_net_channels 
            else:
                in_channels_block = out_channels_block
                out_channels_block = u_net_channels*(4**(n))

            reduction = False if n == n_levels-1 else True
            hw = hw_in/(2**(n))
            self.layers.append(res_blocks(hw, n_blocks, in_channels_block, out_channels_block, k_size=k_size, batch_norm=batch_norm, groups=1, with_reduction=reduction, dropout=dropout))
        
    def forward(self, x):
        outputs = []
        x = self.in_layer(x)
        for layer in self.layers:
            x, x_skip = layer(x)
            outputs.append(x_skip)
        return x, outputs

class decoder_block_shuffle(nn.Module):
    def __init__(self, hw, n_blocks, in_channels, out_channels, k_size=3, with_skip=True, batch_norm=False, dropout=0):
        super().__init__()
  
        self.up = nn.PixelShuffle(2)

        self.with_skip=with_skip

        #self.res_block = res_net_block(out_channels*2, out_channels,  k_size=k_size, batch_norm=batch_norm, dropout=dropout)

        self.res_blocks = res_blocks(hw, n_blocks, in_channels, out_channels, k_size=k_size, batch_norm=False, groups=1, with_reduction=False, dropout=dropout, with_res=True)

    def forward(self, x, skip_channels=None):

        x = self.up(x)

        if self.with_skip:
            x = torch.concat((x, skip_channels), dim=1)
      
        x = self.res_blocks(x)[0]

        return x

class decoder(nn.Module):
    def __init__(self, hw_in, n_levels, n_blocks, u_net_channels, out_channels, upcale_factor=1, k_size=3, batch_norm=False, dropout=0, n_groups=1):
        super().__init__()

        self.decoder_blocks = nn.ModuleList()
        for n in range(n_levels-1, 0,-1):
            out_channels_block = u_net_channels*(4**(n-1))
            in_channels_block = out_channels_block*2
            hw = hw_in/(2**(n-1))

            if n==1:
                out_channels_block = u_net_channels*n_groups*upcale_factor**2
            self.decoder_blocks.append(decoder_block_shuffle(hw, n_blocks, in_channels_block, out_channels_block, k_size=k_size, with_skip=True, batch_norm=batch_norm, dropout=dropout))      
        
        out_channels_block = out_channels_block // upcale_factor**2

        self.upscale = nn.PixelShuffle(upcale_factor) if upcale_factor >1 else nn.Identity()
        self.out_layer = res_net_block(hw_in, out_channels_block, out_channels, k_size=k_size, batch_norm=batch_norm, groups=n_groups, dropout=dropout, with_res=False, with_att=False, with_reduction=False)

    def forward(self, x, skip_channels):

        for k, layer in enumerate(self.decoder_blocks):
            x_skip = skip_channels[-(k+2)]  
            x = layer(x, x_skip)

        x = self.upscale(x)
        x = self.out_layer(x)

        return x

class mid(nn.Module):
    def __init__(self, hw, n_blocks, channels, k_size=3,  dropout=0, with_att=False):
        super().__init__()

        self.res_blocks = res_blocks(hw, n_blocks, channels, channels, k_size=k_size, batch_norm=False, groups=1, with_reduction=False, dropout=dropout, with_att=with_att)

    def forward(self, x):

        x = self.res_blocks(x)[0]

        return x

class ResUNet(nn.Module): 
    def __init__(self, hw_in, hw_out, n_levels, n_res_blocks, model_dim_unet, in_channels, out_channels, batch_norm=True, k_size=3, full_res=True, in_groups=1, out_groups=1, dropout=0):
        super().__init__()

        upcale_factor = hw_out // hw_in

        self.encoder = encoder(hw_in, n_levels, n_res_blocks, model_dim_unet, in_channels, k_size, 5, batch_norm=batch_norm, full_res=full_res, n_groups=in_groups, dropout=dropout)
        self.decoder = decoder(hw_in, n_levels, n_res_blocks, model_dim_unet, out_channels, upcale_factor=upcale_factor, k_size=k_size, batch_norm=False, dropout=dropout, n_groups=out_groups)

        hw_mid = hw_in//(2**(n_levels-1))
        self.mid = mid(hw_mid, n_res_blocks, model_dim_unet*(4**(n_levels-1)), with_att=True)


    def forward(self, x):
        x, layer_outputs = self.encoder(x)
        x = self.mid(x)
        x = self.decoder(x, layer_outputs)
        return x


class core_ResUNet(psm.pyramid_step_model): 
    def __init__(self, model_settings):
        super().__init__(model_settings)
        
        model_settings = self.model_settings

        input_dim = len(model_settings["variables_source"])
        output_dim = len(model_settings["variables_target"])
        dropout = model_settings['dropout']

        n_blocks = model_settings['n_blocks_core']
        model_dim_core = model_settings['model_dim_core']
        depth = model_settings["depth_core"]
        batch_norm = model_settings["batch_norm"]

        dropout = 0 if 'dropout' not in model_settings.keys() else model_settings['dropout']
        full_res = True if 'full_res' not in model_settings.keys() else model_settings['full_res']
        grouped = False if 'grouped' not in model_settings.keys() else model_settings['grouped']

        self.time_dim= False

        if model_settings['gauss']:
            output_dim *=2

        if grouped:
            in_groups = input_dim
            out_groups = output_dim
        else:
            in_groups = out_groups = 1
        hw_in = model_settings['n_regular'][0]
        hw_out = model_settings['n_regular'][1]

        self.core_model = ResUNet(hw_in, hw_out, depth, n_blocks, model_dim_core, input_dim, output_dim, batch_norm=batch_norm, full_res=full_res, in_groups=in_groups, out_groups=out_groups, dropout=dropout)

        if "pretrained_path" in self.model_settings.keys():
            self.check_pretrained(model_dir_check=self.model_settings['pretrained_path'])