import torch
import torch.nn as nn
import torch.nn.functional as F

import climatereconstructionai.model.transformer_helpers as helpers
import climatereconstructionai.model.pyramid_step_model as psm
from .. import config as cfg
from ..utils import grid_utils as gu
from climatereconstructionai.model.net import CRAINet


class CoreCRAI(psm.pyramid_step_model):
    def __init__(self, model_settings, model_dir=None) -> None: 
        super().__init__(model_settings, model_dir=model_dir)

        model_settings = self.model_settings

        input_dim = len(model_settings["variables_source"])
        output_dim = len(model_settings["variables_target"])
        dropout = model_settings['dropout']
        model_dim_core = model_settings['model_dim_core']

        if model_settings['gauss']:
            output_dim *=2

        self.time_dim= True

        self.core_model = CRAINet(img_size_source=(self.n_in, self.n_in),
                        img_size_target=(self.n_out, self.n_out),
                        enc_dec_layers=model_settings["depth_core"],
                        pool_layers=model_settings["n_pool_core"],
                        in_channels=input_dim,
                        out_channels=output_dim,
                        bounds=None,
                        conv_factor=model_dim_core,
                        upsampling_mode='bicubic',
                        predict_residual=False,
                        dropout=dropout)

        if "pretrained_path" in self.model_settings.keys():
            self.check_pretrained(model_dir_check=self.model_settings['pretrained_path'])