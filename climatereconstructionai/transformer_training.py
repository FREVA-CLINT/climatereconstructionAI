import os
import numpy as np
import torch
import torch.nn as nn
import json
import math
import torch.multiprocessing
from torch.utils.data import DataLoader
from tqdm import tqdm
import netCDF4 as netcdf

from .utils import twriter_t, early_stopping, optimization
from .utils.io import save_ckpt
from .utils.netcdfloader_patches import NetCDFLoader_lazy, InfiniteSampler, SampleLoader

def arclen(p1,p2):
  length = 2*torch.arcsin(torch.linalg.norm(p2-p1,axis=-1)/2)
  return length


class AscentFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_input):
        return -grad_input

def make_ascent(loss):
    return AscentFunction.apply(loss)

class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor

def data_to_device(d, device):
    if isinstance(d, dict):
        return dict_to_device(d, device)
    
    elif torch.is_tensor(d):
        return d.to(device)
    else:
        return None

def dict_to_device(d, device):
    for key, value in d.items():
        d[key] = value.to(device)
    return d

def check_get_data_files(list_or_path, root_path = '', train_or_val='train'):

    if isinstance(list_or_path, list):
        data_paths = list_or_path
        if not os.path.isfile(data_paths[0]):
            root_file = os.path.join(root_path, data_paths[0])
            if os.path.isfile(root_file):
                data_paths = [os.path.join(root_path, name) for name in data_paths]
            else:
                data_paths = [os.path.join(root_path, train_or_val, name) for name in data_paths]

    elif isinstance(list_or_path,str):
        if os.path.isfile(list_or_path):
            data_paths = np.genfromtxt(list_or_path, dtype=str)

    return data_paths


def train(model, training_settings, model_settings={}):
 
    torch.multiprocessing.set_sharing_strategy('file_system')

    print("* Number of GPUs: ", torch.cuda.device_count())


    log_dir = training_settings['log_dir']
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    ckpt_dir = os.path.join(log_dir,'ckpts')
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)

    writer = twriter_t.writer(log_dir)

    device = training_settings['device']
    writer.set_hparams(model_settings)

    if 'random_region' not in training_settings.keys():
        random_region = None
    else:
        random_region = training_settings['random_region']
    
    batch_size = training_settings['batch_size']

    source_files_train = check_get_data_files(training_settings['train_data']['data_names_source'], 
                                              root_path = training_settings['root_dir'], 
                                              train_or_val='train')
    target_files_train = check_get_data_files(training_settings['train_data']['data_names_target'], 
                                              root_path = training_settings['root_dir'], 
                                              train_or_val='train')        
    
    source_files_val = check_get_data_files(training_settings['val_data']['data_names_source'],
                                             root_path = training_settings['root_dir'], 
                                             train_or_val='val')
    target_files_val = check_get_data_files(training_settings['val_data']['data_names_target'], 
                                            root_path = training_settings['root_dir'], 
                                            train_or_val='val')      


    if 'save_tensor_samples_path' in training_settings and len(training_settings['save_tensor_samples_path'])>0:
        sample_dir_train = os.path.join(training_settings['save_tensor_samples_path'], 'train')
        sample_dir_val = os.path.join(training_settings['save_tensor_samples_path'], 'val')

        if not os.path.exists(sample_dir_train):
            os.makedirs(sample_dir_train)

        if not os.path.exists(sample_dir_val):
            os.makedirs(sample_dir_val)
    else:
        sample_dir_train=''
        sample_dir_val=''

    if 'train_on_samples' in training_settings.keys() and training_settings['train_on_samples']:

        with open(os.path.join(sample_dir_train, 'dims_var_target.json'), 'r') as f:
            dims_variables_target = json.load(f)

        with open(os.path.join(sample_dir_train, 'dims_var_source.json'), 'r') as f:
            dims_variables_source = json.load(f)

        dataset_train = SampleLoader(sample_dir_train, dims_variables_source, dims_variables_target, training_settings['variables_source'], training_settings['variables_target'])
        dataset_val = SampleLoader(sample_dir_val, dims_variables_source, dims_variables_target, training_settings['variables_source'], training_settings['variables_target'])
       
        with open(os.path.join(sample_dir_train,'norm_dict.json'), 'r') as f:
            norm_dict = json.load(f)

        model_settings['normalization'] = norm_dict
    else:

        dataset_train = NetCDFLoader_lazy(source_files_train, 
                                    target_files_train,
                                    training_settings['variables_source'],
                                    training_settings['variables_target'],
                                    model_settings['normalization'],
                                    model_settings["grid_spacing_equator_km"],
                                    model_settings["pix_size_patch"],
                                    model_settings["patches_overlap_source"],
                                    p_dropout_source=training_settings['p_dropout_source'],
                                    p_dropout_target=training_settings['p_dropout_target'],
                                    n_pts_min = training_settings["n_pts_min"] if 'n_pts_min' in training_settings else True,
                                    save_nc_sample_path='',
                                    save_tensor_sample_path=sample_dir_train,
                                    index_range_source=training_settings['index_range_source'] if 'index_range_source' in training_settings else None,
                                    index_offset_target=training_settings['index_offset_target'] if 'index_offset_target' in training_settings else 0,
                                    rel_coords=training_settings['rel_coords'] if 'rel_coords' in training_settings else False,
                                    sample_for_norm=training_settings['sample_for_norm'] if 'sample_for_norm' in training_settings else None,
                                    lazy_load=training_settings['lazy_load'] if 'lazy_load' in training_settings else False,
                                    rotate_cs=training_settings['rotate_cs'] if 'rotate_cs' in training_settings else False,
                                    interpolation_dict=training_settings['interpolation'],
                                    sample_patch_range_lat=training_settings['sample_patch_range_lat'] if 'sample_patch_range_lat' in training_settings else [-math.pi,math.pi],
                                    sample_condition_dict=training_settings['sample_condition_dict'] if 'sample_condition_dict' in training_settings else {})
        
        dataset_val = NetCDFLoader_lazy(source_files_val, 
                                    target_files_val,
                                    training_settings['variables_source'],
                                    training_settings['variables_target'],
                                    model_settings['normalization'],
                                    model_settings["grid_spacing_equator_km"],
                                    model_settings["pix_size_patch"],
                                    model_settings["patches_overlap_source"],
                                    p_dropout_source=training_settings['p_dropout_source'],
                                    p_dropout_target=training_settings['p_dropout_target'],
                                    n_pts_min = training_settings["n_pts_min"] if 'n_pts_min' in training_settings else True,
                                    save_nc_sample_path='',
                                    save_tensor_sample_path=sample_dir_val,
                                    index_range_source=training_settings['index_range_source'] if 'index_range_source' in training_settings else None,
                                    index_offset_target=training_settings['index_offset_target'] if 'index_offset_target' in training_settings else 0,
                                    rel_coords=training_settings['rel_coords'] if 'rel_coords' in training_settings else False,
                                    sample_for_norm=training_settings['sample_for_norm'] if 'sample_for_norm' in training_settings else None,
                                    lazy_load=training_settings['lazy_load'] if 'lazy_load' in training_settings else False,
                                    rotate_cs=training_settings['rotate_cs'] if 'rotate_cs' in training_settings else False,
                                    interpolation_dict=training_settings['interpolation'],
                                    sample_patch_range_lat=training_settings['sample_patch_range_lat'] if 'sample_patch_range_lat' in training_settings else [-math.pi,math.pi],
                                    sample_condition_dict=training_settings['sample_condition_dict'] if 'sample_condition_dict' in training_settings else {})

        model_settings['normalization'] = norm_dict = dataset_train.norm_dict

        if len(sample_dir_train)>0:
            with open(os.path.join(sample_dir_train,'norm_dict.json'), 'w') as f:
                json.dump(norm_dict, f, indent=4)

    model_settings_path = os.path.join(model_settings['model_dir'],'model_settings.json')
    with open(model_settings_path, 'w') as f:
        json.dump(model_settings, f, indent=4)

    iterator_train = iter(DataLoader(dataset_train,
                                    batch_size=batch_size,
                                    sampler=InfiniteSampler(len(dataset_train)),
                                    num_workers=training_settings['n_workers'], 
                                    pin_memory=True if device == 'cuda' else False,
                                    pin_memory_device=device))

    iterator_val = iter(DataLoader(dataset_val,
                                    batch_size=batch_size,
                                    sampler=InfiniteSampler(len(dataset_val)),
                                    num_workers=training_settings['n_workers'] if 'n_workers_val' not in training_settings.keys() else training_settings['n_workers_val'], 
                                    pin_memory=True if device == 'cuda' else False,
                                    pin_memory_device=device))

    dw_train = False
    if 'dw_train' in training_settings.keys() and training_settings['dw_train']:
        dw_train = True

    model = model.to(device)

    early_stop = early_stopping.early_stopping(training_settings['early_stopping_delta'], training_settings['early_stopping_patience'])

    loss_calculator = optimization.loss_calculator(training_settings, model.model_settings['spatial_dims_var_target'])    

    lambdas_var = training_settings['lambdas_var']
    lambdas_stat = training_settings['lambdas']

    lambdas_optim_keys = [key for key, val in lambdas_stat.items() if val>0]
    lambdas_optim_vals = [torch.nn.Parameter(torch.tensor(1, dtype=float), requires_grad=dw_train) for _ in lambdas_optim_keys]
    lambdas_optim = dict(zip(lambdas_optim_keys, lambdas_optim_vals))

    if 'l1_relv' in lambdas_optim.keys():
        inital_k = 0. if 'initial_k' not in training_settings.keys() else training_settings['initial_k']
        lambdas_optim['k_l1_relv'] = torch.nn.Parameter(torch.tensor(inital_k, dtype=float), requires_grad=dw_train)


    if dw_train:
        optimizer2 = torch.optim.Adam(filter(lambda p: p.requires_grad, lambdas_optim.values()), lr=training_settings['lr_w'])
        lr_scheduler2 = CosineWarmupScheduler(optimizer2, training_settings["T_warmup"], training_settings['max_iter'])
        dw_patience = 4 if 'dw_patience' not in training_settings.keys() else training_settings['dw_patience'] 

    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=training_settings['lr'])
    lr_scheduler = CosineWarmupScheduler(optimizer, training_settings["T_warmup"], training_settings['max_iter'])
  
    
    if training_settings['multi_gpus']:
        model = torch.nn.DataParallel(model)

    pbar = tqdm(range(0, training_settings['max_iter']))

    train_losses_hist = []
    val_losses_hist = []
    lrs = []
    lrs = []
    for i in pbar:
     
        n_iter = i + 1
        lr_val = optimizer.param_groups[0]['lr']
        pbar.set_description("lr = {:.1e}".format(lr_val))

        lrs.append(lr_val)

        model.train()

        data = [data_to_device(x, device) for x in next(iterator_train)]
        source, target, coords_source, coords_target = data[:4] # for backward compability
        target_indices = data[-1]

        if 'k_l1_relv' in lambdas_optim.keys():
            train_total_loss, train_loss_dict = loss_calculator(lambdas_optim, target, model, source, coords_target, target_indices, coords_source=coords_source, k=lambdas_optim['k_l1_relv'])
        else:
            train_total_loss, train_loss_dict = loss_calculator(lambdas_optim, target, model, source, coords_target, target_indices, coords_source=coords_source, k=None)

        train_losses_hist.append(train_loss_dict['total_loss'])

        optimizer.zero_grad()
        if dw_train and len(train_losses_hist)>=dw_patience and (torch.tensor(train_losses_hist)).diff()[-dw_patience:].sum()<0:
            
            train_total_loss.backward(retain_graph=True) 

            dw_loss = make_ascent(train_total_loss)

            optimizer2.zero_grad()
            dw_loss.backward()
            optimizer2.step()
            lr_scheduler2.step()

            lambda_keys = [f'lambda_{key}' for key in lambdas_optim.keys()]
            lambda_vals = [lambdas_stat[key] * val_optim.item() if key!='k_l1_relv' else val_optim.item() for key, val_optim in lambdas_optim.items() ]
            writer.update_scalars(dict(zip(lambda_keys, lambda_vals)), n_iter, 'train')
        
        else:
            train_total_loss.backward() 

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        lr_scheduler.step()


        if n_iter % training_settings['log_interval'] == 0:
            writer.update_scalars(train_loss_dict, n_iter, 'train')

            model.eval()
            val_losses = []

            for _ in range(training_settings['n_iters_val']):

                data = [data_to_device(x, device) for x in next(iterator_val)]
                source, target, coords_source, coords_target = data[:4] # for backward compability
                target_indices = data[-1]               

                if 'k_l1_relv' in lambdas_optim.keys():
                    _, val_loss_dict, output, target, output_reg_hr, output_reg_lr, non_valid_mask = loss_calculator(lambdas_optim, target, model, source, coords_target, target_indices, coords_source=coords_source, val=True, k=lambdas_optim['k_l1_relv'])
                else:
                    _, val_loss_dict, output, target, output_reg_hr, output_reg_lr, non_valid_mask = loss_calculator(lambdas_optim, target, model, source, coords_target, target_indices, coords_source=coords_source, val=True, k=None)

                val_losses.append(list(val_loss_dict.values()))
            
            val_loss = torch.tensor(val_losses).mean(dim=0)
            val_loss = dict(zip(val_loss_dict.keys(), val_loss))
            
            val_losses_hist.append(val_loss['total_loss'])
            debug_dict = {}
            if training_settings['save_debug']:
                torch.save(debug_dict, os.path.join(log_dir,'debug_dict.pt'))
                torch.save(coords_source,os.path.join(log_dir,'coords_source.pt'))
                torch.save(coords_target,os.path.join(log_dir,'coords_target.pt'))
                torch.save(output, os.path.join(log_dir,'output.pt'))
                torch.save(output_reg_hr, os.path.join(log_dir,'output_reg_hr.pt'))
                torch.save(output_reg_lr, os.path.join(log_dir,'output_reg_lr.pt'))
                torch.save(target, os.path.join(log_dir,'target.pt'))
                torch.save(source, os.path.join(log_dir,'source.pt'))
                if "vort" in non_valid_mask.keys():
                    torch.save(non_valid_mask["vort"], os.path.join(log_dir,'non_valid_mask_vort.pt'))
                np.savetxt(os.path.join(log_dir,'losses_val.txt'),np.array(val_losses_hist))
                np.savetxt(os.path.join(log_dir,'losses_train.txt'),np.array(train_losses_hist))
                np.savetxt(os.path.join(log_dir,'lrs.txt'),np.array(lrs))
       
            early_stop.update(val_loss['total_loss'], n_iter, model_save=model)

            writer.update_scalars(val_loss, n_iter, 'val')

            if training_settings['early_stopping']:
                writer.update_scalar('val', 'loss_gradient', early_stop.criterion_diff, n_iter)


        if n_iter % training_settings['save_model_interval'] == 0:
            save_ckpt('{:s}/{:d}.pth'.format(ckpt_dir, n_iter), norm_dict,
                      [(str(n_iter), n_iter, model, optimizer)])

        if training_settings['early_stopping'] and early_stop.terminate:
            model = early_stop.best_model
            break

    save_ckpt('{:s}/best.pth'.format(ckpt_dir), norm_dict,
              [(str(n_iter), n_iter, early_stop.best_model, optimizer)])

    writer.close()