import torch.nn as nn
import torch
import climatereconstructionai.model.pyramid_step_model as psm

class res_net_block_reduction(nn.Module):
    def __init__(self, in_channels, out_channels,  k_size=3, batch_norm=True):
        super().__init__()

        padding = k_size // 2

        self.conv1 = nn.Conv2d(in_channels, out_channels, stride=2, padding=padding, kernel_size=k_size, padding_mode='replicate')
        self.conv2 = nn.Conv2d(out_channels, out_channels, stride=1, padding=padding, kernel_size=k_size, padding_mode='replicate')

        self.reduction = nn.Conv2d(in_channels, out_channels, stride=2, padding=0, kernel_size=1)

        self.bn = nn.BatchNorm2d(in_channels*2) if batch_norm else nn.Identity()
        self.activation2 = nn.LeakyReLU(negative_slope=-0.2)
        self.activation1 = nn.LeakyReLU(negative_slope=-0.2)

    def forward(self, x):
        x_res = self.reduction(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.activation1(x)
        x = self.activation2(x + x_res)
        return x
    


class res_net_block(nn.Module):
    def __init__(self, channels, k_size=3 ,batch_norm=True):
        super().__init__()

        padding = k_size // 2
        self.conv1 = nn.Conv2d(channels, channels, stride=1, padding=padding, kernel_size=k_size)
        self.conv2 = nn.Conv2d(channels, channels, stride=1, padding=padding, kernel_size=k_size)

        self.bn = nn.BatchNorm2d(channels) if batch_norm else nn.Identity()
        
        self.activation2 = nn.LeakyReLU(negative_slope=-0.2)
        self.activation1 = nn.LeakyReLU(negative_slope=-0.2)

    def forward(self, x):
        x_res = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.activation1(x)
        x = self.activation2(x + x_res)
        return x
    


class res_blocks(nn.Module): 
    def __init__(self,  n_blocks, in_channels, out_channels, k_size=3, with_reduction=True, batch_norm=True):
        super().__init__()

        self.res_net_blocks = nn.ModuleList()
        for n in range(n_blocks-1):
            self.res_net_blocks.append(res_net_block(in_channels, k_size=k_size, batch_norm=batch_norm))
        
        self.reduction_block = res_net_block_reduction(in_channels, out_channels, k_size=k_size, batch_norm=batch_norm) if with_reduction else nn.Identity()

    def forward(self,x):

        for layer in self.res_net_blocks:
            x = layer(x)

        return self.reduction_block(x)



class encoder(nn.Module): 
    def __init__(self, n_levels, n_blocks, in_channels, u_net_channels, k_size=3, k_size_in=5):
        super().__init__()

        padding = k_size_in // 2
        self.conv_in = nn.Conv2d(in_channels, u_net_channels, stride=1, padding=padding, kernel_size=k_size_in)

        self.layers = nn.ModuleList()
        for n in range(n_levels):
            in_channels_block = u_net_channels*(2**(n))
            out_channels_block = u_net_channels*(2**(n+1))
            self.layers.append(res_blocks(n_blocks, in_channels_block, out_channels_block, k_size=k_size))
        
    def forward(self, x):
        outputs = []
        x = self.conv_in(x)
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)
        return x, outputs


class conv_block(nn.Module): 
    def __init__(self,  in_channels, out_channels, k_size=3):
        super().__init__()

        padding = k_size // 2
        self.conv_conc = nn.Conv2d(in_channels, out_channels, stride=1, padding=padding, kernel_size=k_size)
        self.conv = nn.Conv2d(out_channels, out_channels, stride=1, padding=padding, kernel_size=k_size)

        self.activation1 = nn.LeakyReLU(negative_slope=-0.2)

    def forward(self, x):
        x = self.conv_conc(x)
        x = self.conv(x)
        x = self.activation1(x)
        return x


class decoder_block(nn.Module):
    def __init__(self, in_channels, out_channels, n_blocks, k_size=3, with_skip=True):
        super().__init__()

        self.trans_conv = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                                        nn.LeakyReLU(negative_slope=-0.2))

        self.with_skip=with_skip
        if not self.with_skip:
            in_channels = out_channels

        self.conv_block = conv_block(in_channels, out_channels, k_size=k_size)

    def forward(self, x, skip_channels=None):

        x = self.trans_conv(x)

        if self.with_skip:
            x = torch.concat((x, skip_channels), dim=1)
      
        x = self.conv_block(x)

        return x
    


class decoder(nn.Module):
    def __init__(self, n_blocks_unet, n_res_blocks, out_channels_unet, upscale_factor, k_size=3):
        super().__init__()

        self.N = n_blocks_unet
        self.decoder_blocks = nn.ModuleList()
        for n in range(n_blocks_unet):
      
            in_channels_block = int(out_channels_unet*(2**((self.N-n))))
            out_channels_block = in_channels_block // 2

            if n == n_blocks_unet -1:
                with_skip = False
                out_channels_block = out_channels_unet
            else:
                with_skip=True

            self.decoder_blocks.append(decoder_block(in_channels_block, out_channels_block, n_res_blocks, k_size=k_size, with_skip=with_skip))

        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        

    def forward(self, x, skip_channels):
        for k, layer in enumerate(self.decoder_blocks):
            
            skip_channel = skip_channels[(self.N-1)-(k+1)]  
            x = layer(x, skip_channel)

        x = self.pixel_shuffle(x)

        return x


class ResUNet(nn.Module): 
    def __init__(self, upcale_factor, n_blocks_unet, n_res_blocks, in_channels, out_channels, k_size=3):
        super().__init__()

        out_channels_unet = upcale_factor**2*out_channels

        self.encoder = encoder(n_blocks_unet, n_res_blocks, in_channels, out_channels_unet, k_size, 5)

        self.decoder = decoder(n_blocks_unet, n_res_blocks, out_channels_unet, upcale_factor, k_size=k_size)

        self.conv_out = conv_block(out_channels, out_channels, k_size=k_size)
                
        
    def forward(self, x):
        x, layer_outputs = self.encoder(x)
        x = self.decoder(x, layer_outputs)
        x = self.conv_out(x)
        return x


class core_ResUNet(psm.pyramid_step_model): 
    def __init__(self, model_settings, load_pretrained=False):
        super().__init__(model_settings, load_pretrained=load_pretrained)
        
        model_settings = self.model_settings

        input_dim = len(model_settings["variables_source"])
        output_dim = len(model_settings["variables_target"])
        dropout = model_settings['dropout']

        model_dim_core = model_settings['model_dim_core']
        depth = model_settings["depth_core"]

        self.time_dim= False

        if model_settings['gauss']:
            output_dim *=2

        upcale_factor = model_settings['n_regular'][1] // model_settings['n_regular'][0]
        self.core_model = ResUNet(upcale_factor, depth, model_dim_core, input_dim, output_dim)

        if load_pretrained:
            self.check_pretrained(model_dir_check=self.model_settings['model_dir'])

        if "pretrained_path" in self.model_settings.keys():
            self.check_pretrained(model_dir_check=self.model_settings['pretrained_path'])