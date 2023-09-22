import torch
import json
import torch.nn as nn
from .. import transformer_training as trainer
from ..utils.io import load_ckpt

class transformer_model(nn.Module):
    def __init__(self, model_settings):
        super().__init__()
        self.model_settings = load_settings(model_settings)
        

    def forward(self):
        pass

    def train_(self, train_settings, pretrain=False):
        self.train_settings = load_settings(train_settings)

        if self.model_settings['use_gauss']:
            self.train_settings['gauss_loss'] = True
        else:
            self.train_settings['gauss_loss'] = False

        trainer.train(self, self.train_settings, self.model_settings)


    def load(self, ckpt_path:str, device=None):
        ckpt_dict = load_ckpt(ckpt_path, device=device)
        self.load_state_dict(ckpt_dict["labels"][-1]["model"])

    def load_pretrained_interpolator(self, ckpt_path:str, device=None):
        ckpt_dict = load_ckpt(ckpt_path, device=device)
        self.load_state_dict(ckpt_dict[ckpt_dict["labels"][-1]]["model"], strict=False)

def load_settings(dict_or_file):
    if isinstance(dict_or_file, dict):
        return dict_or_file

    elif isinstance(dict_or_file, str):
        with open(dict_or_file,'r') as file:
            dict_or_file = json.load(file)

        return dict_or_file