import json
import os
import copy
import torch
import torch.nn as nn
import xarray as xr
from .. import transformer_training as trainer

import climatereconstructionai.model.transformer_helpers as helpers
import climatereconstructionai.model.pyramid_step_model as pysm
import climatereconstructionai.model.core_model_crai as cmc
import climatereconstructionai.model.core_model_resushuffle as cms

from ..utils.io import load_ckpt
from ..utils import grid_utils as gu

class pyramid_model(nn.Module):
    def __init__(self, model_settings):
        super().__init__()

        self.fusion_modules = None
        self.pre_computed_relations = False
    
        self.model_settings = load_settings(model_settings, id='model')
        self.model_dir = self.model_settings['model_dir']

        self.check_model_dir()

        self.check_load_relations()

        self.save_local_domain_parameters()

        if "local_model" in self.model_settings.keys():
            self.load_step_models()

        # for outputting the patch file
        #self.check_grid_file()

    def check_model_dir(self):

        self.relation_fp_source = os.path.join(self.model_dir,'relations_source.pt')
        self.relation_fp_target = os.path.join(self.model_dir,'relations_target.pt')

        self.local_model_dir = os.path.join(self.model_dir,'local_model')
        model_settings = os.path.join(self.model_dir,'model_settings.json')

        if not os.path.isdir(self.model_dir):
            os.mkdir(self.model_dir)
        else:
            if os.path.isfile(self.relation_fp_source):
                self.pre_computed_relations = True 

        with open(model_settings, 'w') as f:
            json.dump(self.model_settings, f, indent=4)

        if not os.path.isdir(self.local_model_dir):
            os.mkdir(self.local_model_dir)


    def check_load_relations(self):
        if self.pre_computed_relations:
            self.relations_source = torch.load(self.relation_fp_source)
            self.relations_target = torch.load(self.relation_fp_target)
        else:
            
            region_grid = xr.load_dataset(self.model_settings['region_grid'])
            parent_coords = gu.get_coords_as_tensor(region_grid, lon='clon', lat='clat')

            if "radius_region_source_km" not in self.model_settings.keys():
                radius_region_source_km = radius_region_target_km =  2*xr.load_dataset(self.model_settings['region_grid']).mean_dual_edge_length/1000
            else:
                radius_region_source_km = self.model_settings["radius_region_source_km"]
                radius_region_target_km = self.model_settings["radius_region_target_km"]
            
            if "radius_inc" not in self.model_settings.keys():
                radius_inc = 100
            else:
                radius_inc = self.model_settings["radius_inc"]

            ds_source = xr.load_dataset(self.model_settings["data_file_source"])
            self.relations_source, self.radius_region_source_km = self.get_relations(ds_source, self.model_settings['variables_source'], parent_coords, radius_region_source_km, radius_inc)

            if "data_file_target" in self.model_settings.keys() and self.model_settings['data_file_target']!=self.model_settings["data_file_source"]:
                ds_target = xr.load_dataset(self.model_settings["data_file_target"])
                self.relations_target, self.radius_region_target_km = self.get_relations(ds_target, self.model_settings['variables_target'], parent_coords, radius_region_target_km, radius_inc)
            else:
                self.relations_target = self.relations_source
                self.radius_region_target_km = radius_region_target_km

            torch.save(self.relations_source, self.relation_fp_source)
            torch.save(self.relations_target, self.relation_fp_target)
    


    def get_relations(self, ds, variables, parent_coords, radius_region_km, radius_inc):

        ds_dict = gu.prepare_coordinates_ds_dict({0: {'ds': ds}},[0], variables)
        child_coords_spatial_dims = ds_dict[0]['spatial_dims']

        child_var_spatial_dims = ds_dict[0]['var_spatial_dims']
        child_spatial_dims_var = gu.invert_dict(child_var_spatial_dims)

        relations = {}
        for spatial_dim, coords in child_coords_spatial_dims.items():
            child_coords = torch.stack(tuple(coords['coords'].values()),dim=0)

            relation_dict, radius_region_km = gu.get_parent_child_indices(parent_coords, child_coords, radius_region_km, radius_inc, min_overlap=self.model_settings["min_overlap_regions"], batch_size=self.model_settings['relation_batch_size'])

            rel_coords_children = gu.get_relative_coordinates_grids(parent_coords, child_coords, relation_dict, relative_to='parents', batch_size=self.model_settings['relation_batch_size'])
            rel_coords_parents = gu.get_relative_coordinates_grids(parent_coords, child_coords, relation_dict, relative_to='children', batch_size=self.model_settings['relation_batch_size'])

            relations[spatial_dim] = {
                'indices': relation_dict,
                'rel_coords_children': rel_coords_children,
                'rel_coords_parents': rel_coords_parents
            }

        relations['spatial_dims_var'] = child_spatial_dims_var
        relations['var_spatial_dims'] = child_var_spatial_dims
        relations['radius_region_km'] = radius_region_km

        return relations, radius_region_km
    
    def save_local_domain_parameters(self):

        local_model_settings = {}
        
        local_model_settings['variables_source'] = self.model_settings['variables_source']
        local_model_settings['variables_target'] = self.model_settings['variables_target']

        local_model_settings['spatial_dims_var_source'] = self.relations_source['spatial_dims_var']
        local_model_settings['spatial_dims_var_target'] = self.relations_target['spatial_dims_var']

        local_model_settings['radius_region_source_km'] = self.relations_source['radius_region_km']
        local_model_settings['radius_region_target_km'] = self.relations_target['radius_region_km']

        with open(os.path.join(self.local_model_dir,"domain.json"),"w+") as f:
            json.dump(local_model_settings, f, indent=4)   

    def load_step_models(self):
     
        local_model_specs = self.model_settings["local_model"]
        local_model_specs_is_file = local_model_specs.endswith('.json')

        step_model_settings = pysm.load_settings(local_model_specs)

        if not local_model_specs_is_file:
            step_model_settings['model_dir_pretrained'] = self.model_settings["local_model"]

        if step_model_settings['model'] =='crai':
            self.local_model = cmc.CoreCRAI(step_model_settings)
        
        elif step_model_settings['model'] =='shuffle':
            self.local_model = cms.core_ResUNet(step_model_settings)
        
        if not local_model_specs_is_file:
            self.norm_stats_file = os.path.join(local_model_specs,'norm_stats.json')
            norm_stats_file_local = os.path.join(self.local_model_dir, 'norm_stats.json')
            
            if os.path.isfile(norm_stats_file_local):
                with open(norm_stats_file_local,'r') as file:
                    self.norm_stats = json.load(file)
            else:
                if os.path.isfile(self.norm_stats_file):

                    with open(self.norm_stats_file,'r') as file:
                        self.norm_stats = json.load(file)

                    with open(os.path.join(self.local_model_dir,"norm_stats.json"),"w+") as f:
                        json.dump(self.norm_stats,f, indent=4)    
        

    def check_grid_file(self):
        if 'grid_file_target' in self.model_settings.keys():
            grid = xr.load_dataset(self.model_settings['grid_file_target'])

            self.meridional_normal_dual_edge = grid.meridional_normal_dual_edge.values
            self.zonal_normal_dual_edge = grid.zonal_normal_dual_edge.values
        

    def forward(self):
        pass

    def get_parents(self):
        pass

    def get_children(self):
        pass
    
    def load_grid_relations(self):
        pass
    
    def get_batches_coords(self, relations_dict, batch_size, device='cpu'):
        
        spatial_dims = list(relations_dict['spatial_dims_var'].keys())
        n_regions = relations_dict[spatial_dims[0]]['rel_coords_children'].shape[0]

        if batch_size != -1 and batch_size < n_regions:
            n_chunks = n_regions // batch_size
        else:
            n_chunks = 1

        batch_indices = torch.stack(torch.arange(n_regions).chunk(n_chunks, dim=0),dim=0)

        spatial_dim_coords_batched = {}
        spatial_dim_indices_batched = {}
        spatial_dim_indices_rel = {}
        spatial_dim_indices = {}
        spatial_dim_p_indices = {}

        for spatial_dim in spatial_dims:
            coords = relations_dict[spatial_dim]['rel_coords_children'][:,:,-2:].transpose(-2,-1).float()
            indices = relations_dict[spatial_dim]['indices']['children']
            c_indices_rel = relations_dict[spatial_dim]['indices']['children_idx'].to(device)
            p_indices = relations_dict[spatial_dim]['indices']['parents'].to(device)

            spatial_dim_indices[spatial_dim] = indices[batch_indices].to(device)

            spatial_dim_indices_batched[spatial_dim] = indices[batch_indices].to(device)
            spatial_dim_coords_batched[spatial_dim] = coords[batch_indices].to(device)

            spatial_dim_indices_rel[spatial_dim] = c_indices_rel[spatial_dim_indices[spatial_dim]].to(device)
            spatial_dim_p_indices[spatial_dim] = p_indices[spatial_dim_indices[spatial_dim]].to(device)

        spatial_dim_coords_batched = dl_to_ld(spatial_dim_coords_batched)
        spatial_dim_indices_batched = dl_to_ld(spatial_dim_indices)

        return spatial_dim_indices, spatial_dim_coords_batched, spatial_dim_indices_batched, spatial_dim_indices_rel, spatial_dim_p_indices
    

    def apply(self, x, ts=-1, batch_size=-1, device='cpu'):

        indices_source, coords_source_batches, indices_source_batches,_,_ = self.get_batches_coords(self.relations_source, batch_size, device=device)
        indices_target, coords_target_batches, _, _, _ = self.get_batches_coords(self.relations_target, batch_size, device=device)

        spatial_dims_source = list(self.relations_source['spatial_dims_var'].keys())
        spatial_dims_target = list(self.relations_target['spatial_dims_var'].keys())

        data_source = {}
        for spatial_dim in spatial_dims_source:
            for variable in self.relations_source['spatial_dims_var'][spatial_dim]:
                data = torch.tensor(x[variable].values[[ts]]).squeeze().to(device)[indices_source[spatial_dim]].unsqueeze(dim=-1)
                data_source[variable] = (data - self.norm_stats[variable]['q_05'])/(self.norm_stats[variable]['q_95'] - self.norm_stats[variable]['q_05'])

        data_source_batched = dl_to_ld(data_source)

        data_output_regions = {}
        for spatial_dim in spatial_dims_target:
            for variable in self.relations_target['spatial_dims_var'][spatial_dim]:
                data_output_regions[variable] = torch.zeros_like(indices_target[spatial_dim], dtype=torch.float)

        for batch_idx in range(len(indices_source_batches)):
            coords_source_batch = coords_source_batches[batch_idx]
            coords_target_batch = coords_target_batches[batch_idx]
            data_source_batch = data_source_batched[batch_idx]
            output_batch = self.local_model(data_source_batch, coords_source_batch, coords_target_batch)[0]
            
            for variable, data in output_batch.items():
                
                data = (data).detach().cpu()*(self.norm_stats[variable]['q_95'] - self.norm_stats[variable]['q_05']) + self.norm_stats[variable]['q_05']
                data_output_regions[variable][batch_idx] = data[:,:,0,0]


        for variable, spatial_dim in  self.relations_target['var_spatial_dims'].items():
            which_regions = self.relations_target[spatial_dim]['indices']['parents']
            which_idx_in_region = self.relations_target[spatial_dim]['indices']['children_idx']

            data = data_output_regions[variable]
            b,n,c = data.shape
            data = data.view(b*n,c)
            data_output_regions[variable]=data[which_regions, which_idx_in_region].mean(dim=-1).numpy()
            
        return data_output_regions

        # work through global data and then average - parameters?
    def train_(self, train_settings, pretrain=False):
        self.train_settings = load_settings(train_settings, id='train')

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

def load_settings(dict_or_file, id='model'):
    if isinstance(dict_or_file, dict):
        return dict_or_file

    elif isinstance(dict_or_file, str):
        if os.path.isfile(dict_or_file):
            with open(dict_or_file,'r') as file:
                dict_or_file = json.load(file)
        else:
            dict_or_file = os.path.join(dict_or_file, f'{id}_settings.json')
            with open(dict_or_file,'r') as file:
                dict_or_file = json.load(file)

        return dict_or_file


def dl_to_ld(dl):
    return [dict(zip(dl,t)) for t in zip(*dl.values())]