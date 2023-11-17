import torch.nn as nn
import torch
import climatereconstructionai.model.pyramid_step_model as psm
import math
    
  
class res_net_block(nn.Module):
    def __init__(self, in_channels, out_channels,  k_size=3, batch_norm=True, with_reduction=False):
        super().__init__()

        padding = k_size // 2
        stride = 2 if with_reduction else 1

        self.conv1 = nn.Conv2d(in_channels, out_channels, stride=stride, padding=padding, kernel_size=k_size, padding_mode='replicate')
        self.conv2 = nn.Conv2d(out_channels, out_channels, stride=1, padding=padding, kernel_size=k_size, padding_mode='replicate')

        self.reduction = nn.Conv2d(in_channels, out_channels, stride=stride, padding=0, kernel_size=1)

        self.bn1 = nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()
        self.bn2 = nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()
        self.activation2 = nn.SiLU()
        self.activation1 = nn.SiLU()

    def forward(self, x):
        x_res = self.reduction(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation2(x + x_res)
        return x
        

class res_blocks(nn.Module): 
    def __init__(self,  n_blocks, in_channels, out_channels, k_size=3, with_reduction=True, batch_norm=True):
        super().__init__()

        self.res_block1 = res_net_block(in_channels, out_channels,  k_size=k_size)
        
        self.res_net_blocks = nn.ModuleList()
        for n in range(n_blocks-1):
            self.res_net_blocks.append(res_net_block(out_channels, out_channels, k_size=k_size, batch_norm=batch_norm))

        self.max_pool = nn.MaxPool2d(kernel_size=2) if with_reduction else nn.Identity()

    def forward(self,x):
        x = self.res_block1(x)

        for layer in self.res_net_blocks:
            x = layer(x)

        return self.max_pool(x)

class encoder(nn.Module): 
    def __init__(self, n_levels, n_blocks, in_channels, u_net_channels, k_size=3, k_size_in=5):
        super().__init__()

        self.layers = nn.ModuleList()
        for n in range(n_levels):
            if n==0:
                in_channels_block = in_channels
                out_channels_block = u_net_channels
            else:
                in_channels_block = u_net_channels*(2**(n-1))
                out_channels_block = u_net_channels*(2**(n))
            self.layers.append(res_blocks(n_blocks, in_channels_block, out_channels_block, k_size=k_size))
        
    def forward(self, x):
        outputs = []
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)
        return x, outputs


class decoder_block(nn.Module):
    def __init__(self, in_channels, out_channels, n_blocks, k_size=3, with_skip=True):
        super().__init__()

        self.trans_conv = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1))
        
        self.with_skip=with_skip
        if not self.with_skip:
            in_channels = out_channels

        self.res_block = res_net_block(out_channels*2, out_channels,  k_size=k_size, batch_norm=False)

    def forward(self, x, skip_channels=None):

        x = self.trans_conv(x)

        if self.with_skip:
            x = torch.concat((x, skip_channels), dim=1)
      
        x = self.res_block(x)

        return x
    

class decoder(nn.Module):
    def __init__(self, n_blocks_unet, n_res_blocks, u_net_channels, out_channels, k_size=3):
        super().__init__()

        self.decoder_blocks = nn.ModuleList()
        for n in range(n_blocks_unet-1, 0,-1):
            in_channels_block = u_net_channels*(2**(n))
            out_channels_block = u_net_channels*(2**(n-1))

            self.decoder_blocks.append(decoder_block(in_channels_block, out_channels_block, n_res_blocks, k_size=k_size, with_skip=True))      
        
        self.out_layer = nn.Conv2d(out_channels_block, out_channels, stride=1, padding=1, kernel_size=k_size, padding_mode='replicate')

    def forward(self, x, skip_channels):

        for k, layer in enumerate(self.decoder_blocks):
            skip_channel = skip_channels[-(k+2)]  
            x = layer(x, skip_channel)

        x = self.out_layer(x)

        return x

class ResUNet(nn.Module): 
    def __init__(self, upcale_factor, n_blocks_unet, n_res_blocks, model_dim_unet, in_channels, out_channels, k_size=3):
        super().__init__()

        self.encoder = encoder(n_blocks_unet, n_res_blocks, in_channels, model_dim_unet, k_size, 5)

        out_channels_unet = upcale_factor**2*out_channels
        self.decoder = decoder(n_blocks_unet, n_res_blocks, model_dim_unet, out_channels_unet, k_size=k_size)

        self.pixel_shuffle = nn.PixelShuffle(upcale_factor) if upcale_factor >1 else nn.Identity()
    

    def forward(self, x):
        x, layer_outputs = self.encoder(x)
        x = self.decoder(x, layer_outputs)
        x = self.pixel_shuffle(x)
        return x


class core_ResUNet(psm.pyramid_step_model): 
    def __init__(self, model_settings, load_pretrained=False):
        super().__init__(model_settings, load_pretrained=load_pretrained)
        
        model_settings = self.model_settings

        input_dim = len(model_settings["variables_source"])
        output_dim = len(model_settings["variables_target"])
        dropout = model_settings['dropout']

        n_blocks = model_settings['n_blocks_core']
        model_dim_core = model_settings['model_dim_core']
        depth = model_settings["depth_core"]

        self.time_dim= False

        if model_settings['gauss']:
            output_dim *=2

        upcale_factor = model_settings['n_regular'][1] // model_settings['n_regular'][0]
        self.core_model = ResUNet(upcale_factor, depth, n_blocks, model_dim_core, input_dim, output_dim)

        if load_pretrained:
            self.check_pretrained(model_dir_check=self.model_settings['model_dir'])

        if "pretrained_path" in self.model_settings.keys():
            self.check_pretrained(model_dir_check=self.model_settings['pretrained_path'])