import json
import os
import copy
import torch
import torch.nn as nn
from .. import transformer_training as trainer
from .. import transformer_infer as inference

import climatereconstructionai.model.transformer_helpers as helpers
import climatereconstructionai.model.pyramid_step_model as pysm
import climatereconstructionai.model.core_model_crai as cmc

from ..utils.io import load_ckpt
from ..utils import grid_utils as gu

class pyramid_model(nn.Module):
    def __init__(self, model_settings):
        super().__init__()

        self.fusion_modules = None
        self.pre_computed_relations = False
    
        self.model_settings = load_settings(model_settings)
        self.model_dir = self.model_settings['model_dir']

        self.check_model_dir()

        self.load_step_models()

        self.check_load_relations()

    def check_model_dir(self):

        self.relation_fp = os.path.join(self.model_dir,'relations.pt')
        self.step_dir = os.path.join(self.model_dir,'step_models')
        model_settings = os.path.join(self.model_dir,'model_settings.json')

        if not os.path.isdir(self.model_dir):
            os.mkdir(self.model_dir)
        else:
            if os.path.isfile(self.relation_fp):
                self.pre_computed_relations = True 

        with open(model_settings, 'w') as f:
            json.dump(self.model_settings, f)

        if not os.path.isdir(self.step_dir):
            os.mkdir(self.step_dir)

    def check_load_relations(self):
        if self.pre_computed_relations:
            self.relations = torch.load(self.relation_fp)
        else:
            grids = [self.model_settings['region_grid'], self.pysm_models[0].model_settings['input_grid'], self.pysm_models[0].model_settings['output_grid']]
            coord_dict = self.pysm_models[0].model_settings['coord_dict']
            radius_region =  self.pysm_models[0].model_settings['radius_region_km']
            self.relations = gu.get_grid_relations(grids, coord_dict, save_file_path=self.relation_fp, radius_regions_km=radius_region, resolutions=self.model_settings['resolutions'])

    def load_step_models(self):
        self.pysm_models = nn.ModuleList()

        for pys_model_dir in self.model_settings["step_models"]:
            pys_model = cmc.CoreCRAI(pys_model_dir)
            self.pysm_models.append(pys_model)

    def forward(self):
        pass

    def get_parents(self):
        pass

    def get_children(self):
        pass
    
    def load_grid_relations(self):
        pass
    
    # -> high-level models first, cache results, then fusion
    def apply_serial(self):
        pass

    # feed data from all levels into the model at once
    def apply_parallel(self):
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
    
