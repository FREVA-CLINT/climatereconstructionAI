import os
import numpy as np
import torch
import torch.multiprocessing
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils import twriter_t, early_stopping
from .utils.io import save_ckpt
from .utils.netcdfloader_trans import NetCDFLoader, InfiniteSampler


def gather_rel_coords(c, indices):
    c = c[indices].squeeze()
    c = torch.gather(c, dim=2, index=indices.permute(0,2,1).repeat(1,indices.shape[1],1))
    return c


def input_dropout(x, coord_dict, perc=0.2, device='cpu'):

    num_rows = x.shape[1]

    keep = torch.rand((num_rows)) > perc

    indices = [torch.randperm(num_rows)[keep].unsqueeze(dim=0) for k in range(x.shape[0])]
    indices = torch.cat(indices).unsqueeze(dim=-1).to(device)

    x = torch.gather(x, dim=1, index=indices)


    coord_dict['rel']['source'][0] = gather_rel_coords(coord_dict['rel']['source'][0], indices)
    coord_dict['rel']['source'][1] = gather_rel_coords(coord_dict['rel']['source'][1], indices)


    coord_dict['abs']['source'][0] = coord_dict['abs']['source'][0].squeeze()[indices]
    coord_dict['abs']['source'][1] = coord_dict['abs']['source'][1].squeeze()[indices]


    return x, coord_dict

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

    # create data sets
        # create data sets
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
                                 device=training_settings['device'])
    
    dataset_val = NetCDFLoader(training_settings['data_root_dir'],
                                 training_settings['data_names_source'], 
                                 training_settings['data_names_target'],
                                 'val',
                                 training_settings['variables'],
                                 training_settings['coord_names'],
                                 random_region=random_region,
                                 apply_img_norm=training_settings['apply_img_norm'],
                                 normalize_data=training_settings['normalize_data'],
                                 device=training_settings['device'])
    
    iterator_train = iter(DataLoader(dataset_train,
                                     batch_size=batch_size,
                                     sampler=InfiniteSampler(len(dataset_train)),
                                     num_workers=training_settings['n_workers'], 
                                     pin_memory=True if device == 'cuda' else False,
                                     pin_memory_device=device))

    iterator_val = iter(DataLoader(dataset_val,
                                    batch_size=batch_size,
                                    sampler=InfiniteSampler(len(dataset_train)),
                                    num_workers=training_settings['n_workers'], 
                                    pin_memory=True if device == 'cuda' else False,
                                    pin_memory_device=device))
    
    train_stats = {"mean": dataset_train.normalizer.moments[0], "std": dataset_train.normalizer.moments[1]}

    model = model.to(device)

    loss_fcn = torch.nn.L1Loss()
   
  
    early_stop = early_stopping.early_stopping(training_settings['early_stopping_delta'], training_settings['early_stopping_patience'])
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=training_settings['lr'],weight_decay=0.05)

    lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1/training_settings["T_warmup"], end_factor=1, total_iters=training_settings["T_warmup"])

    start_iter = 0

    if training_settings['multi_gpus']:
        model = torch.nn.DataParallel(model)

    pbar = tqdm(range(start_iter, training_settings['max_iter']))

    train_losses_save = []
    val_losses_save = []
    for i in pbar:
     
        n_iter = i + 1
        lr_val = optimizer.param_groups[0]['lr']
        pbar.set_description("lr = {:.1e}".format(lr_val))

        # train model
        model.train()

        if 'random_region' in training_settings.keys():
            if i % training_settings['random_region']['generate_interval']==0:
                dataset_train.generate_region()

        source, target = next(iterator_train)

        source, coord_dict = input_dropout(source.to(device), dataset_train.coord_dict, training_settings['input_dropout'], device=device)

        output = model(source, coord_dict)

        optimizer.zero_grad()
        loss = loss_fcn(output.view(batch_size,-1), target.view(batch_size,-1).to(device))
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

                if 'random_region' in training_settings.keys():
                    if i % training_settings['random_region']['generate_interval']==0:
                        dataset_val.generate_region()

                source, target =  next(iterator_val)

                source, coord_dict = input_dropout(source.to(device), dataset_val.coord_dict, training_settings['input_dropout'], device=device)

                with torch.no_grad():
                    output = model(source, coord_dict)
                    loss = loss_fcn(output.view(batch_size,-1), target.view(batch_size,-1).to(device))
                    val_loss = {'total': loss.item(), 'valid': loss.item()}

                val_losses.append(list(val_loss.values()))

            val_losses_save.append(val_loss['total'])
            output, debug_dict = model(source, coord_dict, return_debug=True)

            if training_settings['save_debug']:
                torch.save(debug_dict, os.path.join(log_dir,'debug_dict.pt'))
                torch.save(coord_dict,os.path.join(log_dir,'coord_dict.pt'))
                torch.save(output, os.path.join(log_dir,'output.pt'))
                torch.save(target, os.path.join(log_dir,'target.pt'))
                torch.save(source, os.path.join(log_dir,'source.pt'))
                np.savetxt(os.path.join(log_dir,'losses_val.txt'),np.array(val_losses_save))
                np.savetxt(os.path.join(log_dir,'losses_train.txt'),np.array(train_losses_save))

            val_loss = torch.tensor(val_losses).mean(dim=0)
            val_loss = dict(zip(train_loss.keys(), val_loss))

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

