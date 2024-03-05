import torch.nn as nn
import torch
import climatereconstructionai.model.pyramid_step_model as psm
from climatereconstructionai.model.core_model_resushuffle import encoder, decoder, res_blocks
import climatereconstructionai.model.transformer_helpers as helpers
import math



class DiagonalGaussianDistribution(object):
    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean, device=self.parameters.device, dtype=self.parameters.dtype
            )

    def sample(self) -> torch.FloatTensor:
        # make sure sample is on the same device as the parameters and has same dtype

        sample = torch.randn_like(self.mean, dtype=self.mean.dtype, device=self.mean.device)
        x = self.mean + self.std * sample
        return x

    def kl(self, other: "DiagonalGaussianDistribution" = None) -> torch.Tensor:
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=[1, 2, 3],
                )
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 2, 3],
                )


class mid(nn.Module):
    def __init__(self, hw, n_blocks, channels, laten_dim=None, k_size=3,  dropout=0, with_att=False, bias=True, global_padding=False):
        super().__init__()
        if laten_dim is None:
            laten_dim = channels
        self.res_blocks = res_blocks(hw, n_blocks, channels, laten_dim, k_size=k_size, batch_norm=False, groups=1,  dropout=dropout, with_att=with_att, bias=bias, global_padding=global_padding, factor=1)
        self.quant_conv = nn.Conv2d(laten_dim, 2*laten_dim, groups=2, kernel_size=k_size, padding=1)

    def forward(self, x):

        x = self.res_blocks(x)[0]
        moments = self.quant_conv(x)
        posterior = DiagonalGaussianDistribution(moments)
   
        return posterior

    

class out_net(nn.Module):
    def __init__(self, res_indices, hw_in, hw_out, res_mode=True, global_padding=False):
        super().__init__()

        self.res_indices_rhs = res_indices
        self.res_indices_lhs = torch.arange(len(res_indices))

        self.global_residual = True
        scale_factor = hw_out[0] / hw_in[1]
        self.res_mode = res_mode
        self.scale_factor = scale_factor

        self.res_interpolate = nn.Identity()

        self.global_residual = False
        
        
    def forward(self, x):
        return x


class ResVAE(nn.Module): 
    def __init__(self, hw_in, hw_out, phys_size_factor, n_levels, factor, n_res_blocks, model_dim_unet, in_channels, out_channels, res_mode, res_indices, latent_dim=None, channels_upscaling=None, batch_norm=True, k_size=3, min_att_dim=0, in_groups=1, out_groups=1, dropout=0, bias=True, global_padding=False, mid_att=False, initial_res=True, down_method="max"):
        super().__init__()

        global_upscale_factor = int(torch.tensor([(hw_out[0]) // hw_in[0], 1]).max())

        self.encoder = encoder(hw_in, factor, n_levels, n_res_blocks, model_dim_unet, in_channels, k_size, 7, min_att_dim=min_att_dim, batch_norm=batch_norm, n_groups=in_groups, dropout=dropout, bias=bias, global_padding=global_padding, initial_res=initial_res, down_method=down_method)
        
        self.encoder.layer_configs[-1]['out_channels']=latent_dim

        self.post_conv = nn.Conv2d(latent_dim, latent_dim, kernel_size=k_size, padding=1)
        self.decoder = decoder(self.encoder.layer_configs, phys_size_factor, factor, n_res_blocks, out_channels, global_upscale_factor=global_upscale_factor, k_size=k_size, dropout=dropout, n_groups=out_groups, bias=bias, global_padding=global_padding, channels_upscaling=channels_upscaling, skip_channels=False)

        self.out_net = out_net(res_indices, hw_in, hw_out, res_mode=res_mode, global_padding=global_padding)
  
        hw_mid = torch.tensor(hw_in)// (factor**(n_levels-1))
        self.mid = mid(hw_mid, n_res_blocks, model_dim_unet*(2**(n_levels-1)), latent_dim, with_att=mid_att, bias=bias, global_padding=global_padding)
        self.kl = 0

    def encode(self, x):
        x, _ = self.encoder(x)
        return self.mid(x)       

    def decode(self, x):
        x = self.post_conv(x)
        return self.decoder(x)

    def forward(self, x):

        posterior = self.encode(x)

        self.kl = posterior.kl()

        sample = posterior.sample()

        x = self.decode(sample)

        return x


class core_ResVAE(psm.pyramid_step_model): 
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
        mid_att = model_settings["mid_att"] if "mid_att" in model_settings.keys() else False
        latent_dim = model_settings["latent_dim"] if "latent_dim" in model_settings.keys() else model_dim_core*2**model_settings['depth_core']

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
        channels_upscaling = model_settings['channels_upscaling'] if "channels_upscaling" in model_settings.keys() else None
        min_att_dim = model_settings['min_att_dim'] if "min_att_dim" in model_settings.keys() else 0

        phys_size_factor = (1+2*model_settings["patches_overlap_source"])/(1+2*model_settings["patches_overlap_target"]) 

        self.core_model = ResVAE(hw_in, hw_out, phys_size_factor, depth, factor, n_blocks, model_dim_core, input_dim, output_dim, model_settings['res_mode'], res_indices, latent_dim=latent_dim, min_att_dim=min_att_dim, batch_norm=batch_norm, in_groups=in_groups, out_groups=out_groups, dropout=dropout, bias=bias, global_padding=global_padding, mid_att=mid_att, initial_res=initial_res, down_method=down_method, channels_upscaling=channels_upscaling)

        if "pretrained_path" in self.model_settings.keys():
            self.check_pretrained(log_dir_check=self.model_settings['pretrained_path'])
        elif self.eval_mode:
            self.check_pretrained(log_dir_check=self.log_dir)