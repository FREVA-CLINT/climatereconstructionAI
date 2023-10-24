import torch
import torch.nn as nn
import torch.nn.functional as F

import climatereconstructionai.model.transformer_helpers as helpers
import climatereconstructionai.model.pyramid_step_model as psm
from .. import config as cfg
from ..utils import grid_utils as gu
from climatereconstructionai.model.net import CRAINet


class CoreCRAI(psm.pyramid_step_model):
    def __init__(self, model_settings, load_pretrained=False) -> None: 
        super().__init__(model_settings,load_pretrained=load_pretrained)

        model_settings = self.model_settings

        input_dim = model_settings["input_dim"]
        ff_dim = model_settings["ff_dim"]
        dropout = model_settings['dropout']
        model_dim = model_settings['model_dim']
        nh = model_settings['model_dim']

        
        if model_settings["gauss"]:
            input_dim = input_dim*2

        output_dim = input_dim

        self.core_model = CRAINet(img_size_source=(self.n_lr, self.n_lr),
                        img_size_target=(self.n_hr, self.n_hr),
                        enc_dec_layers=4,
                        pool_layers=2,
                        in_channels=input_dim,
                        out_channels=output_dim,
                        bounds=None,
                        conv_factor=32,
                        upsampling_mode='bicubic',
                        predict_residual=True)
    
