import os
import numpy as np
import torch
import torch.nn as nn

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
        loss =  self.Gauss(output[:,:,0],target.squeeze(),output[:,:,1])
        return loss

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

def train(model, training_settings, model_hparams={}):
 
    print("* Number of GPUs: ", torch.cuda.device_count())

    torch.multiprocessing.set_sharing_strategy('file_system')

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
    
    dataset_train = NetCDFLoader(training_settings['data_root_dir'],
                                 training_settings['data_names_source'], 
                                 training_settings['data_names_target'],
                                 'train',
                                 training_settings['variables'],
                                 training_settings['coord_names'],
                                 random_region=random_region,
                                 apply_img_norm=training_settings['apply_img_norm'],
                                 normalize_data=training_settings['normalize_data'],
                                 p_input_dropout=training_settings['input_dropout'],
                                 sampling_mode=training_settings['sampling_mode'],
                                 n_points=training_settings['n_points'])
    
    dataset_val = NetCDFLoader(training_settings['data_root_dir'],
                                 training_settings['data_names_source'], 
                                 training_settings['data_names_target'],
                                 'val',
                                 training_settings['variables'],
                                 training_settings['coord_names'],
                                 random_region=random_region,
                                 apply_img_norm=training_settings['apply_img_norm'],
                                 normalize_data=dataset_train.normalizer.moments,
                                 p_input_dropout=training_settings['input_dropout'],
                                 sampling_mode=training_settings['sampling_mode'],
                                 n_points=training_settings['n_points'])
    
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
    
    train_stats = {"mean": dataset_train.normalizer.moments[0], "std": dataset_train.normalizer.moments[1]}

    model = model.to(device)


    if training_settings["gauss_loss"]:
        loss_fcn = GaussLoss()
    else:
        loss_fcn = torch.nn.L1Loss()
   
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

        output = model(source, coord_dict)

        optimizer.zero_grad()
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
                    output = model(source, coord_dict)
                    loss = loss_fcn(output, target)
                    val_loss = {'total': loss.item(), 'valid': loss.item()}

                val_losses.append(list(val_loss.values()))
            
            output, debug_dict = model(source, coord_dict, return_debug=True)

            val_loss = torch.tensor(val_losses).mean(dim=0)
            val_loss = dict(zip(train_loss.keys(), val_loss))

            val_losses_save.append(val_loss['total'])

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


