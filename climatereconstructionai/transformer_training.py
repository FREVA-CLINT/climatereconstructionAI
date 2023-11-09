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
from .utils.netcdfloader_samples import NetCDFLoader_lazy, InfiniteSampler

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
        loss = self.loss(output[:,:,:,0],target)
        return loss

class DictLoss(nn.Module):
    def __init__(self, loss_fcn):
        super().__init__()
        self.loss_fcn = loss_fcn

    def forward(self, output, target):
        loss_dict = {}
        total_loss = 0

        for var in output.keys():
            loss = self.loss_fcn(output[var], target[var])
            loss_dict[var] = loss.item()
            total_loss+=loss

        return total_loss, loss_dict

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

    if len(training_settings["norm_stats"]) > 0:
        with open(training_settings["norm_stats"],'r') as file:
            stat_dict = json.load(file)
    else:
        stat_dict = None

    if 'save_samples_path' in training_settings and len(training_settings['save_samples_path'])>0:
        sample_dir_train = os.path.join(training_settings['save_samples_path'], 'train')
        sample_dir_val = os.path.join(training_settings['save_samples_path'], 'val')

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
                                random_region=random_region,
                                apply_img_norm=training_settings['apply_img_norm'],
                                normalize_data=training_settings['normalize_data'],
                                stat_dict=stat_dict,
                                p_dropout_source=training_settings['p_dropout_source'],
                                p_dropout_target=training_settings['p_dropout_target'],
                                sampling_mode=training_settings['sampling_mode'],
                                save_sample_path=sample_dir_train,
                                coordinate_pert=training_settings['coordinate_pertubation'],
                                index_range=training_settings['index_range'] if 'index_range' in training_settings else None,
                                rel_coords=training_settings['rel_coords'] if 'rel_coords' in training_settings else False,
                                sample_for_norm=training_settings['sample_for_norm'] if 'sample_for_norm' in training_settings else None,
                                norm_stats_save_path=training_settings['model_dir'],
                                lazy_load=training_settings['lazy_load'] if 'lazy_load' in training_settings else False)
    
    dataset_val = NetCDFLoader_lazy(source_files_val, 
                                target_files_val,
                                training_settings['variables_source'],
                                training_settings['variables_target'],
                                random_region=random_region,
                                apply_img_norm=training_settings['apply_img_norm'],
                                normalize_data=training_settings['normalize_data'],
                                stat_dict=dataset_train.stat_dict if stat_dict is None else stat_dict,
                                p_dropout_source=training_settings['p_dropout_source'],
                                p_dropout_target=training_settings['p_dropout_target'],
                                sampling_mode=training_settings['sampling_mode'],
                                save_sample_path=sample_dir_val,
                                coordinate_pert=0,
                                index_range=training_settings['index_range'] if 'index_range' in training_settings else None,
                                rel_coords=training_settings['rel_coords'] if 'rel_coords' in training_settings else False,
                                sample_for_norm=training_settings['sample_for_norm'] if 'sample_for_norm' in training_settings else None,
                                lazy_load=training_settings['lazy_load'] if 'lazy_load' in training_settings else False)
    

    
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

    dict_loss_fcn = DictLoss(loss_fcn)

   
    early_stop = early_stopping.early_stopping(training_settings['early_stopping_delta'], training_settings['early_stopping_patience'])
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=training_settings['lr'])

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

        source, target, coords_source, coords_target = [dict_to_device(x, device) for x in next(iterator_train)]

        output = model(source, coords_source, coords_target)[0]

        optimizer.zero_grad()

        loss, train_loss_dict = dict_loss_fcn(output, target)

        loss.backward()

        train_losses_save.append(loss.item())
        train_loss_dict['total'] = loss.item()

        optimizer.step()
        lr_scheduler.step()

        if n_iter % training_settings['log_interval'] == 0:
            writer.update_scalars(train_loss_dict, n_iter, 'train')

            model.eval()
            val_losses = []

            for _ in range(training_settings['n_iters_val']):

                source, target, coords_source, coords_target = [dict_to_device(x, device) for x in next(iterator_val)]

                with torch.no_grad():
                    output, output_reg = model(source, coords_source, coords_target)

                    loss, val_loss_dict = dict_loss_fcn(output, target)

                    val_loss_dict['total'] = loss.item()

                val_losses.append(list(val_loss_dict.values()))
            
            val_loss = torch.tensor(val_losses).mean(dim=0)
            val_loss = dict(zip(train_loss_dict.keys(), val_loss))

            val_losses_save.append(val_loss['total'])
            debug_dict = {}
            if training_settings['save_debug']:
                torch.save(debug_dict, os.path.join(log_dir,'debug_dict.pt'))
                torch.save(coords_source,os.path.join(log_dir,'coords_source.pt'))
                torch.save(coords_target,os.path.join(log_dir,'coords_target.pt'))
                torch.save(output, os.path.join(log_dir,'output.pt'))
                torch.save(output_reg, os.path.join(log_dir,'output_reg.pt'))
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