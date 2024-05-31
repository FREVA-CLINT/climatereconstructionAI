import os
import numpy as np
import torch
import torch.nn as nn
import json
import math
import torch.multiprocessing
import torch.utils
from torch.utils.data import DataLoader
import torch.utils.data
import torch.utils.data.distributed
from tqdm import tqdm
import netCDF4 as netcdf

from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from .utils import twriter_t, early_stopping, optimization
from .utils.io import save_ckpt
from .utils.netcdfloader_trans import NetCDFLoader_lazy, InfiniteSampler

def arclen(p1,p2):
  length = 2*torch.arcsin(torch.linalg.norm(p2-p1,axis=-1)/2)
  return length


def set_device_and_init_torch_dist():
    
    # check out https://gist.github.com/TengdaHan/1dd10d335c7ca6f13810fff41e809904

    world_size = int(os.environ.get('WORLD_SIZE'))

    #rank = int(os.environ.get('RANK', os.environ.get('SLURM_PROCID')))
    rank = int(os.environ.get('SLURM_PROCID'))

    dist_url = 'env://'
    backend = 'nccl'

    dist.init_process_group(backend=backend, init_method=dist_url,
                                world_size=world_size, rank=rank)
       
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)

    return local_rank


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

    def __init__(self, optimizer, warmup, max_iters, iter_start=0):
        self.warmup = warmup
        self.max_num_iters = max_iters
        self.iter_start = iter_start
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        epoch += self.iter_start
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
    
    if training_settings['distributed']:
        local_rank = set_device_and_init_torch_dist()
    else:
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
    
    if 'data_names_target_past' in training_settings['train_data'].keys():
        target_files_past_train = check_get_data_files(training_settings['train_data']['data_names_target_past'], 
                                                root_path = training_settings['root_dir'], 
                                                train_or_val='train')
        target_files_past_val = check_get_data_files(training_settings['val_data']['data_names_target_past'], 
                                                root_path = training_settings['root_dir'], 
                                                train_or_val='val')
    else:
        target_files_past_train = target_files_past_val = None


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


    dataset_train = NetCDFLoader_lazy(source_files_train, 
                                target_files_train,
                                training_settings['variables_source'],
                                training_settings['variables_target'],
                                model_settings['normalization'],
                                training_settings['coarsen_sample_level'],
                                files_target_past=target_files_past_train,
                                index_range_source=training_settings['index_range_source'] if 'index_range_source' in training_settings else None,
                                index_offset_target=training_settings['index_offset_target'] if 'index_offset_target' in training_settings else 0,
                                sample_for_norm=training_settings['sample_for_norm'] if 'sample_for_norm' in training_settings else None,
                                lazy_load=training_settings['lazy_load'] if 'lazy_load' in training_settings else False,
                                sample_condition_dict=training_settings['sample_condition_dict'],
                                model_settings=model_settings,
                                p_dropout = training_settings['p_dropout'] if 'p_dropout' in training_settings else 0)
    
    dataset_val = NetCDFLoader_lazy(source_files_val, 
                                target_files_val,
                                training_settings['variables_source'],
                                training_settings['variables_target'],
                                model_settings['normalization'],
                                training_settings['coarsen_sample_level'],
                                files_target_past=target_files_past_val,
                                index_range_source=training_settings['index_range_source'] if 'index_range_source' in training_settings else None,
                                index_offset_target=training_settings['index_offset_target'] if 'index_offset_target' in training_settings else 0,
                                sample_for_norm=training_settings['sample_for_norm'] if 'sample_for_norm' in training_settings else None,
                                lazy_load=training_settings['lazy_load'] if 'lazy_load' in training_settings else False,
                                sample_condition_dict=training_settings['sample_condition_dict'],
                                model_settings=model_settings,
                                p_dropout = training_settings['p_dropout'] if 'p_dropout' in training_settings else 0)

    model_settings['normalization'] = norm_dict = dataset_train.norm_dict

    if len(sample_dir_train)>0:
        with open(os.path.join(sample_dir_train,'norm_dict.json'), 'w') as f:
            json.dump(norm_dict, f, indent=4)

    model_settings_path = os.path.join(model_settings['model_dir'],'model_settings.json')
    with open(model_settings_path, 'w') as f:
        json.dump(model_settings, f, indent=4)
    
    if training_settings['distributed']:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset_train,
                shuffle=True
            )
        val_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset_val,
                shuffle=False
            )
    else:
        train_sampler = InfiniteSampler(len(dataset_train))
        val_sampler = InfiniteSampler(len(dataset_val))    

    iterator_train = iter(DataLoader(dataset_train,
                                    batch_size=batch_size,
                                    sampler=train_sampler,
                                    num_workers=training_settings['n_workers'], 
                                    pin_memory=True if device == 'cuda' or training_settings['distributed'] else False))

    iterator_val = iter(DataLoader(dataset_val,
                                    batch_size=batch_size,
                                    sampler=val_sampler,
                                    num_workers=training_settings['n_workers'] if 'n_workers_val' not in training_settings.keys() else training_settings['n_workers_val'], 
                                    pin_memory=True if device == 'cuda' or training_settings['distributed'] else False))

    dw_train = False
    if 'dw_train' in training_settings.keys() and training_settings['dw_train']:
        dw_train = True

    early_stop = early_stopping.early_stopping(training_settings['early_stopping_delta'], training_settings['early_stopping_patience'])

    loss_calculator = optimization.loss_calculator(training_settings, model_settings['variables_target'], model_settings)    

    lambdas_var = training_settings['lambdas_var']
    lambdas_stat = training_settings['lambdas']

    lambdas_optim_keys = [key for key, val in lambdas_stat.items() if val>0]
    lambdas_optim_vals = [torch.nn.Parameter(torch.tensor(1, dtype=float), requires_grad=dw_train) for _ in lambdas_optim_keys]
    lambdas_optim = dict(zip(lambdas_optim_keys, lambdas_optim_vals))

    if 'l1_relv' in lambdas_optim.keys():
        inital_k = 0. if 'initial_k' not in training_settings.keys() else training_settings['initial_k']
        lambdas_optim['k_l1_relv'] = torch.nn.Parameter(torch.tensor(inital_k, dtype=float), requires_grad=dw_train)


    if "continue_training" in training_settings.keys() and training_settings["continue_training"]:
        iter_start = model.trained_iterations
    else:
        iter_start = 0

    if training_settings['distributed']:
        model = model.cuda(local_rank)
        model = DDP(model, device_ids=[local_rank])

    elif training_settings['multi_gpus']:
        model = model.to(device)
        model = torch.nn.DataParallel(model)      
    
    else:
        model = model.to(device)

    if dw_train:
        optimizer2 = torch.optim.Adam(filter(lambda p: p.requires_grad, lambdas_optim.values()), lr=training_settings['lr_w'])
        lr_scheduler2 = CosineWarmupScheduler(optimizer2, training_settings["T_warmup"], training_settings['max_iter'])
        dw_patience = 4 if 'dw_patience' not in training_settings.keys() else training_settings['dw_patience'] 

    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=training_settings['lr'])
    lr_scheduler = CosineWarmupScheduler(optimizer, training_settings["T_warmup"], training_settings['max_iter'], iter_start=iter_start)
    
    if training_settings['mixed_precision']:
        scaler = GradScaler()   

    pbar = tqdm(range(0, training_settings['max_iter']))

    train_losses_hist = []
    val_losses_hist = []
    lrs = []
    lrs = []
    for i in pbar:
     
        n_iter = i + 1 + iter_start
        lr_val = optimizer.param_groups[0]['lr']
        pbar.set_description("lr = {:.1e}".format(lr_val))

        lrs.append(lr_val)

        model.train()

        data = [data_to_device(x, device) for x in next(iterator_train)]
        source, target, indices = data

        if training_settings['mixed_precision']:
            with torch.autocast(device_type=training_settings['device'], dtype=torch.bfloat16):
                train_total_loss, train_loss_dict = loss_calculator(lambdas_optim, 
                                                                        target, 
                                                                        model, 
                                                                        source, 
                                                                        source_indices=indices, 
                                                                        k=lambdas_optim['k_l1_relv'] if 'k_l1_relv' in lambdas_optim.keys() else None, 
                                                                        val=False, 
                                                                        model_type='transformer')
        else:
            train_total_loss, train_loss_dict = loss_calculator(lambdas_optim, 
                                                                        target, 
                                                                        model, 
                                                                        source, 
                                                                        source_indices=indices, 
                                                                        k=lambdas_optim['k_l1_relv'] if 'k_l1_relv' in lambdas_optim.keys() else None, 
                                                                        val=False, 
                                                                        model_type='transformer')
        
        train_losses_hist.append(train_loss_dict['total_loss'])


        optimizer.zero_grad()
        if dw_train and len(train_losses_hist)>=dw_patience and (torch.tensor(train_losses_hist)).diff()[-dw_patience:].sum()<0:
            
            if training_settings['mixed_precision']:
                scaler.scale(train_total_loss).backward(retain_graph=True)
            else:
                train_total_loss.backward() 
      
            dw_loss = make_ascent(train_total_loss)

            optimizer2.zero_grad()
            if training_settings['mixed_precision']:
                scaler.scale(dw_loss).backward(retain_graph=True)
            else:
                dw_loss.backward()

            optimizer2.step()
            lr_scheduler2.step()

            lambda_keys = [f'lambda_{key}' for key in lambdas_optim.keys()]
            lambda_vals = [lambdas_stat[key] * val_optim.item() if key!='k_l1_relv' else val_optim.item() for key, val_optim in lambdas_optim.items() ]
            writer.update_scalars(dict(zip(lambda_keys, lambda_vals)), n_iter, 'train')
        
        else:
            if training_settings['mixed_precision']:
                scaler.scale(train_total_loss).backward()
                scaler.unscale_(optimizer)
            else:
                train_total_loss.backward() 

        if "clip_grad_norm" in training_settings.keys() and training_settings["clip_grad_norm"]>0:
            nn.utils.clip_grad_norm_(model.parameters(), training_settings["clip_grad_norm"])

        if training_settings['mixed_precision']:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        lr_scheduler.step()


        if n_iter % training_settings['log_interval'] == 0:
            writer.update_scalars(train_loss_dict, n_iter, 'train')

            model.eval()
            val_losses = []

            for _ in range(training_settings['n_iters_val']):

                data = [data_to_device(x, device) for x in next(iterator_val)]
                source, target, indices = data

                if training_settings['mixed_precision']:
                    with torch.autocast(device_type=training_settings['device'], dtype=torch.bfloat16):
                        val_total_loss, val_loss_dict, output, target, _, debug_dict = loss_calculator(lambdas_optim, 
                                                                                                target, 
                                                                                                model, 
                                                                                                source, 
                                                                                                source_indices=indices, 
                                                                                                k=lambdas_optim['k_l1_relv'] if 'k_l1_relv' in lambdas_optim.keys() else None, 
                                                                                                val=True, 
                                                                                                model_type='transformer')
                else:
                    val_total_loss, val_loss_dict, output, target, _, debug_dict = loss_calculator(lambdas_optim, 
                                                                                                target, 
                                                                                                model, 
                                                                                                source, 
                                                                                                source_indices=indices, 
                                                                                                k=lambdas_optim['k_l1_relv'] if 'k_l1_relv' in lambdas_optim.keys() else None, 
                                                                                                val=True, 
                                                                                                model_type='transformer')

                
                val_losses.append(list(val_loss_dict.values()))
            
            val_loss = torch.tensor(val_losses).mean(dim=0)
            val_loss = dict(zip(val_loss_dict.keys(), val_loss))
            
            val_losses_hist.append(val_loss['total_loss'])
            
            if training_settings['save_debug']:
                torch.save(debug_dict, os.path.join(log_dir,'debug_dict.pt'))
                torch.save(output, os.path.join(log_dir,'output.pt'))
                torch.save(target, os.path.join(log_dir,'target.pt'))
                torch.save(source, os.path.join(log_dir,'source.pt'))
                #input_mapping = {}
                #for key, layer in model.input_layers.items():
                #    input_mapping[key] = layer.input_mapping

                #input_coordinates = {}
                #for key, layer in model.input_layers.items():
                #    input_coordinates[key] = layer.input_coordinates

                #torch.save(input_mapping, os.path.join(log_dir,'input_mapping.pt'))
                #torch.save(input_coordinates, os.path.join(log_dir,'input_coordinates.pt'))
                torch.save(indices, os.path.join(log_dir,'indices.pt'))
                np.savetxt(os.path.join(log_dir,'losses_val.txt'),np.array(val_losses_hist))
                np.savetxt(os.path.join(log_dir,'losses_train.txt'),np.array(train_losses_hist))
                np.savetxt(os.path.join(log_dir,'lrs.txt'),np.array(lrs))
       
            early_stop.update(val_loss['total_loss'], n_iter, model_save=model)

            writer.update_scalars(val_loss, n_iter, 'val')

            if training_settings['early_stopping']:
                writer.update_scalar('val', 'loss_gradient', early_stop.criterion_diff, n_iter)


        if n_iter % training_settings['save_model_interval'] == 0:
            save_ckpt('{:s}/{:d}.pth'.format(ckpt_dir, n_iter), norm_dict,
                      [(str(n_iter), n_iter, model, optimizer)], model_settings=model_settings)

        if training_settings['early_stopping'] and early_stop.terminate:
            model = early_stop.best_model
            break

    save_ckpt('{:s}/best.pth'.format(ckpt_dir), norm_dict,
              [(str(n_iter), n_iter, early_stop.best_model, optimizer)], model_settings=model_settings)

    writer.close()