import json
import os
import copy
import torch.nn as nn
from .. import transformer_training as trainer
from .. import transformer_infer as inference
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

        if self.train_settings['pretrain_interpolator']:

            pretrain_settings = copy.deepcopy(self.train_settings)
            model_settings = copy.deepcopy(self.model_settings)
            pretrain_model_setting = copy.deepcopy(self.model_settings)

            pretrain_settings["T_warmup"]=2000
            pretrain_settings["max_iter"]=5000
            pretrain_settings["batch_size"]=32
            pretrain_settings["log_interval"]=500
            pretrain_settings["save_model_interval"]=1000

            pretrain_settings["log_dir"] = os.path.join(pretrain_settings["log_dir"],'pretrain')
            pretrain_model_setting['encoder']['n_layers']=0
            pretrain_model_setting['decoder']['n_layers']=0

            self.__init__(pretrain_model_setting)
            trainer.train(self, pretrain_settings, pretrain_model_setting)
            self.model_settings["pretrained"] = os.path.join(pretrain_settings["log_dir"],'ckpts','best.pth')

            self.__init__(model_settings)

        trainer.train(self, self.train_settings, self.model_settings)

    def infer(self, settings):
        self.inference_settings = load_settings(settings)
        inference.infer(self, self.inference_settings)

    def load(self, ckpt_path:str, device=None):
        ckpt_dict = load_ckpt(ckpt_path, device=device)
        self.load_state_dict(ckpt_dict["labels"][-1]["model"])

    def check_pretrained(self):
        if len(self.model_settings["pretrained"]) >0:
            self.load_pretrained(self.model_settings["pretrained"], encoder_only=False)

        elif len(self.model_settings["encoder"]["pretrained"]) >0:
            self.load_pretrained(self.model_settings["encoder"]["pretrained"], encoder_only=True)

    def load_pretrained(self, ckpt_path:str, device=None, encoder_only=True):
        ckpt_dict = load_ckpt(ckpt_path, device=device)
        model_state_dict = ckpt_dict[ckpt_dict["labels"][-1]]["model"]
        if encoder_only:
            load_state_dict = {}
            for key, value in model_state_dict.items():
                if (key.split(".")[0] == "Encoder"):
                    load_state_dict[key] = value
        else:
            load_state_dict = model_state_dict
        self.load_state_dict(load_state_dict, strict=False)

def load_settings(dict_or_file):
    if isinstance(dict_or_file, dict):
        return dict_or_file

    elif isinstance(dict_or_file, str):
        with open(dict_or_file,'r') as file:
            dict_or_file = json.load(file)

        return dict_or_file
    

