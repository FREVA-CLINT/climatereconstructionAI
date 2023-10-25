import os
import numpy as np
import torch
import torch.nn as nn
import json

import torch.multiprocessing
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils import twriter_t, early_stopping
from .utils.io import save_ckpt
from .utils.netcdfloader_trans import NetCDFLoader, InfiniteSampler


class GaussLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.Gauss = torch.nn.GaussianNLLLoss()

    def forward(self, output, target):
        loss =  self.Gauss(output[:,:,:,0],target,output[:,:,:,1])
        return loss

class L1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.L1Loss()

    def forward(self, output, target):
        loss =  self.loss(output[:,:,:,0],target)
        return loss


class Loss_w_source(nn.Module):
    def __init__(self, target_loss, p=0.5):
        super().__init__()
        self.source_loss = torch.nn.L1Loss()
        self.target_loss = target_loss
        self.p = p

    def forward(self, output, target, source_output, source):
        target_loss = self.target_loss(output, target)
        source_loss = self.source_loss(source_output, source)

        total_loss = self.p*target_loss + (1-self.p)*source_loss
        return total_loss

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


def train(model, training_settings, model_hparams={}):
 
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
    writer.set_hparams(model_hparams)

    if 'random_region' not in training_settings.keys():
        random_region = None
    else:
        random_region = training_settings['random_region']
    
    batch_size = training_settings['batch_size']

    source_files_train = check_get_data_files(training_settings['train_data']['data_names_source'], root_path = training_settings['root_dir'], train_or_val='train')
    target_files_train = check_get_data_files(training_settings['train_data']['data_names_target'], root_path = training_settings['root_dir'], train_or_val='train')        
    
    source_files_val = check_get_data_files(training_settings['val_data']['data_names_source'], root_path = training_settings['root_dir'], train_or_val='val')
    target_files_val = check_get_data_files(training_settings['val_data']['data_names_target'], root_path = training_settings['root_dir'], train_or_val='val')      

    if len(training_settings["norm_stats"]) > 0:
        with open(training_settings["norm_stats"],'r') as file:
            stat_dict = json.load(file)
    else:
        stat_dict = None

    dataset_train = NetCDFLoader(source_files_train, 
                                 target_files_train,
                                 training_settings['variables'],
                                 training_settings['coord_dict'],
                                 random_region=random_region,
                                 apply_img_norm=training_settings['apply_img_norm'],
                                 normalize_data=training_settings['normalize_data'],
                                 stat_dict=stat_dict,
                                 p_input_dropout=training_settings['input_dropout'],
                                 sampling_mode=training_settings['sampling_mode'],
                                 coordinate_pert=training_settings['coordinate_pertubation'],
                                 index_range=training_settings['index_range'] if 'index_range' in training_settings else None,
                                 rel_coords=training_settings['rel_coords'] if 'rel_coords' in training_settings else False,
                                 lazy_load=training_settings['lazy_load'] if 'lazy_load' in training_settings else False,
                                 sample_for_norm=training_settings['sample_for_norm'] if 'sample_for_norm' in training_settings else -1,
                                 norm_stats_save_path=training_settings['model_dir'])
    
    dataset_val = NetCDFLoader(  source_files_val, 
                                 target_files_val,
                                 training_settings['variables'],
                                 training_settings['coord_dict'],
                                 random_region=random_region,
                                 apply_img_norm=training_settings['apply_img_norm'],
                                 normalize_data=training_settings['normalize_data'],
                                 stat_dict=dataset_train.stat_dict if stat_dict is None else stat_dict,
                                 p_input_dropout=training_settings['input_dropout'],
                                 sampling_mode=training_settings['sampling_mode'],
                                 coordinate_pert=0,
                                 index_range=training_settings['index_range'] if 'index_range' in training_settings else None,
                                 rel_coords=training_settings['rel_coords'] if 'rel_coords' in training_settings else False,
                                 lazy_load=training_settings['lazy_load'] if 'lazy_load' in training_settings else False,
                                 sample_for_norm=training_settings['sample_for_norm'] if 'sample_for_norm' in training_settings else -1)
    
    iterator_train = iter(DataLoader(dataset_train,
                                     batch_size=batch_size,
                                     sampler=InfiniteSampler(len(dataset_train)),
                                     num_workers=training_settings['n_workers'], 
                                     pin_memory=True if device == 'cuda' else False,
                                     pin_memory_device=device))

    iterator_val = iter(DataLoader(dataset_val,
                                    batch_size=batch_size,
                                    sampler=InfiniteSampler(len(dataset_val)),
                                    num_workers=training_settings['n_workers'], 
                                    pin_memory=True if device == 'cuda' else False,
                                    pin_memory_device=device))
    
    train_stats = dataset_train.stat_dict

    model = model.to(device)


    if training_settings["gauss_loss"]:
        loss_fcn = GaussLoss()
    else:
        loss_fcn = L1Loss()

    if training_settings["source_loss"]:
        loss_fcn = Loss_w_source(loss_fcn, p=0.5)

   
    early_stop = early_stopping.early_stopping(training_settings['early_stopping_delta'], training_settings['early_stopping_patience'])
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=training_settings['lr'], weight_decay=0.05)

    lr_scheduler = CosineWarmupScheduler(optimizer, training_settings["T_warmup"], training_settings['max_iter'])

    start_iter = 0

    if training_settings['multi_gpus']:
        model = torch.nn.DataParallel(model)

    pbar = tqdm(range(start_iter, training_settings['max_iter']))

    train_losses_save = []
    val_losses_save = []
    lrs = []
    for i in pbar:
     
        n_iter = i + 1
        lr_val = optimizer.param_groups[0]['lr']
        pbar.set_description("lr = {:.1e}".format(lr_val))

        lrs.append(lr_val)

        model.train()

        
        source, target, coord_dict = next(iterator_train)

        coord_dict['rel'] = dict_to_device(coord_dict['rel'], device)

        source = source.to(device)
        target = target.to(device)

        output, source_output = model(source, coord_dict)

        optimizer.zero_grad()

        if training_settings['source_loss']:
            loss = loss_fcn(output, target, source_output, source)
        else:
            loss = loss_fcn(output, target)

        loss.backward()

        train_losses_save.append(loss.item())
        train_loss = {'total': loss.item(), 'valid': loss.item()}
        
        optimizer.step()
        lr_scheduler.step()

        if n_iter % training_settings['log_interval'] == 0:
            writer.update_scalars(train_loss, n_iter, 'train')

            model.eval()
            val_losses = []

            for _ in range(training_settings['n_iters_val']):

                source, target, coord_dict = next(iterator_val)

                coord_dict['rel'] = dict_to_device(coord_dict['rel'], device)
                
                source = source.to(device)
                target = target.to(device)

                with torch.no_grad():
                    output, source_output = model(source, coord_dict)

                    if training_settings['source_loss']:
                        loss = loss_fcn(output, target, source_output, source)
                    else:
                        loss = loss_fcn(output, target)

                    val_loss = {'total': loss.item(), 'valid': loss.item()}

                val_losses.append(list(val_loss.values()))
            
            #output, debug_dict = model(source, coord_dict, return_debug=True)

            val_loss = torch.tensor(val_losses).mean(dim=0)
            val_loss = dict(zip(train_loss.keys(), val_loss))

            val_losses_save.append(val_loss['total'])
            debug_dict = {}
            if training_settings['save_debug']:
                torch.save(debug_dict, os.path.join(log_dir,'debug_dict.pt'))
                torch.save(coord_dict,os.path.join(log_dir,'coord_dict.pt'))
                torch.save(output, os.path.join(log_dir,'output.pt'))
                torch.save(target, os.path.join(log_dir,'target.pt'))
                torch.save(source, os.path.join(log_dir,'source.pt'))
                np.savetxt(os.path.join(log_dir,'losses_val.txt'),np.array(val_losses_save))
                np.savetxt(os.path.join(log_dir,'losses_train.txt'),np.array(train_losses_save))
                np.savetxt(os.path.join(log_dir,'lrs.txt'),np.array(lrs))
       
            early_stop.update(val_loss['total'], n_iter, model_save=model)

            writer.update_scalars(val_loss, n_iter, 'val')

            if training_settings['early_stopping']:
                writer.update_scalar('val', 'loss_gradient', early_stop.criterion_diff, n_iter)


        if n_iter % training_settings['save_model_interval'] == 0:
            save_ckpt('{:s}/{:d}.pth'.format(ckpt_dir, n_iter), train_stats,
                      [(str(n_iter), n_iter, model, optimizer)])

        if training_settings['early_stopping'] and early_stop.terminate:
            model = early_stop.best_model
            break

    save_ckpt('{:s}/best.pth'.format(ckpt_dir), train_stats,
              [(str(n_iter), n_iter, early_stop.best_model, optimizer)])

    writer.close()


def create_samples(sample_settings):

    print("* Number of GPUs: ", torch.cuda.device_count())

    sample_dir = sample_settings['sample_dir']
    sample_dir_train = os.path.join(sample_settings['sample_dir'], 'train')
    sample_dir_val = os.path.join(sample_settings['sample_dir'], 'val')

    if not os.path.exists(sample_dir_train):
        os.makedirs(sample_dir_train)

    if not os.path.exists(sample_dir_val):
        os.makedirs(sample_dir_val)

    if 'random_region' not in sample_settings.keys():
        random_region = None
    else:
        random_region = sample_settings['random_region']
    
    batch_size = sample_settings['batch_size']

    source_files_train = check_get_data_files(sample_settings['train_data']['data_names_source'], root_path = sample_settings['root_dir'], train_or_val='train')
    target_files_train = check_get_data_files(sample_settings['train_data']['data_names_target'], root_path = sample_settings['root_dir'], train_or_val='train')        
    
    source_files_val = check_get_data_files(sample_settings['val_data']['data_names_source'], root_path = sample_settings['root_dir'], train_or_val='val')
    target_files_val = check_get_data_files(sample_settings['val_data']['data_names_target'], root_path = sample_settings['root_dir'], train_or_val='val')      

    stat_dict = None

    dataset_train = NetCDFLoader(source_files_train, 
                                 target_files_train,
                                 sample_settings['variables'],
                                 sample_settings['coord_dict'],
                                 random_region=random_region,
                                 apply_img_norm=False,
                                 normalize_data=False,
                                 stat_dict=None,
                                 p_input_dropout=0,
                                 sampling_mode=sample_settings['sampling_mode'],
                                 coordinate_pert=0,
                                 save_sample_path=sample_dir_train,
                                 index_range=sample_settings['index_range'] if 'index_range' in sample_settings else None,
                                 lazy_load=sample_settings['lazy_load'] if 'lazy_load' in sample_settings else False,
                                 sample_for_norm=sample_settings['sample_for_norm'] if 'sample_for_norm' in sample_settings else -1)
    
    dataset_val = NetCDFLoader(  source_files_val, 
                                 target_files_val,
                                 sample_settings['variables'],
                                 sample_settings['coord_dict'],
                                 random_region=random_region,
                                 apply_img_norm=False,
                                 normalize_data=False,
                                 stat_dict=dataset_train.stat_dict if stat_dict is None else stat_dict,
                                 p_input_dropout=0,
                                 sampling_mode=sample_settings['sampling_mode'],
                                 coordinate_pert=0,
                                 save_sample_path=sample_dir_val,
                                 index_range=sample_settings['index_range'] if 'index_range' in sample_settings else None,
                                 lazy_load=sample_settings['lazy_load'] if 'lazy_load' in sample_settings else False,
                                 sample_for_norm=sample_settings['sample_for_norm'] if 'sample_for_norm' in sample_settings else -1)
    
    iterator_train = iter(DataLoader(dataset_train,
                                     batch_size=batch_size,
                                     sampler=InfiniteSampler(len(dataset_train)),
                                     num_workers=sample_settings['n_workers']))

    iterator_val = iter(DataLoader(dataset_val,
                                    batch_size=batch_size,
                                    sampler=InfiniteSampler(len(dataset_val)),
                                    num_workers=sample_settings['n_workers']))
    

    start_iter = 0

    pbar = tqdm(range(start_iter, sample_settings['n_samples_train']))

    for i in pbar:      
        next(iterator_train)


    pbar = tqdm(range(start_iter, sample_settings['n_samples_val']))

    for i in pbar:
        next(iterator_val)