import copy
import os

import json
import numpy as np
import torch
import torch.multiprocessing
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import config as cfg

from .utils import twriter, early_stopping
from .utils.evaluation import create_snapshot_image
from .utils.io import load_ckpt, load_model, save_ckpt
from .utils.netcdfloader_trans import NetCDFLoader, InfiniteSampler

import climatereconstructionai.model.transformer_net_2 as nt
    

def train_transformer(arg_file=None):
    cfg.set_train_args(arg_file)

    with open(cfg.transformer_settings,'r') as file:
        settings = json.load(file)

    print("* Number of GPUs: ", torch.cuda.device_count())

    torch.multiprocessing.set_sharing_strategy('file_system')

    np.random.seed(cfg.loop_random_seed)
    if cfg.cuda_random_seed is not None:
        torch.manual_seed(cfg.cuda_random_seed)

    if cfg.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False

    for subdir in ("", "/images", "/ckpt"):
        outdir = cfg.snapshot_dir + subdir
        if not os.path.exists(outdir):
            os.makedirs(outdir)

    writer = twriter.writer()

    writer.set_hparams(cfg.passed_args)

    # create data sets
        # create data sets
    if 'random_region' not in settings.keys():
        random_region = None
    else:
        random_region = settings['random_region']

    dataset_train = NetCDFLoader(cfg.data_root_dir, [cfg.data_names[0]], [cfg.data_names[1]], 'train',
                                 cfg.data_types, settings['coord_names'], random_region=random_region)
    
    dataset_val = NetCDFLoader(cfg.data_root_dir, [cfg.data_names[0]], [cfg.data_names[1]], 'val',
                                 cfg.data_types, settings['coord_names'], random_region=random_region)
    
    iterator_train = iter(DataLoader(dataset_train, batch_size=cfg.batch_size,
                                     sampler=InfiniteSampler(len(dataset_train)),
                                     num_workers=cfg.n_threads, 
                                     pin_memory=True if cfg.device != 'cpu' else False,
                                     pin_memory_device=cfg.device))

    iterator_val = iter(DataLoader(dataset_val, batch_size=cfg.batch_size,
                                   sampler=InfiniteSampler(len(dataset_val)),
                                   num_workers=cfg.n_threads,
                                   pin_memory=True if cfg.device != 'cpu' else False,
                                   pin_memory_device=cfg.device))

    model = nt.CRTransNet(settings['model']).to(cfg.device)

    loss_fcn = torch.nn.L1Loss()
   
    # define learning rate
    if cfg.finetune:
        lr = cfg.lr_finetune
        model.freeze_enc_bn = True
    else:
        lr = cfg.lr

    early_stop = early_stopping.early_stopping()
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=settings['optimization']['lr'],weight_decay=0.05)

    lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1/settings['optimization']["T_warmup"], end_factor=1, total_iters=settings['optimization']["T_warmup"])

    start_iter = 0
    if cfg.resume_iter:
        ckpt_dict = load_ckpt('{}/ckpt/{}.pth'.format(cfg.snapshot_dir, cfg.resume_iter), cfg.device)
        start_iter = load_model(ckpt_dict, model, optimizer)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Starting from iter ', start_iter)

    if cfg.multi_gpus:
        model = torch.nn.DataParallel(model)

    i = cfg.max_iter - (cfg.n_final_models - 1) * cfg.final_models_interval
    final_models = range(i, cfg.max_iter + 1, cfg.final_models_interval)

    savelist = []
    pbar = tqdm(range(start_iter, cfg.max_iter))

    train_losses_save = []
    val_losses_save = []
    for i in pbar:
     
        n_iter = i + 1
        lr_val = optimizer.param_groups[0]['lr']
        pbar.set_description("lr = {:.1e}".format(lr_val))

        # train model
        model.train()

        if 'random_region' in settings.keys():
            if i % settings['random_region']['generate_interval']==0:
                dataset_train.generate_region()
        source, target =  next(iterator_train)

        output = model(source.to(cfg.device), dataset_train.coord_dict)

        optimizer.zero_grad()
        loss = loss_fcn(output.view(cfg.batch_size,-1), target.view(cfg.batch_size,-1).to(cfg.device))
        loss.backward()

        train_losses_save.append(loss.item())
        train_loss = {'total': loss.item(), 'valid': loss.item()}
        
        optimizer.step()

        lr_scheduler.step()

        if (cfg.log_interval and n_iter % cfg.log_interval == 0):
            writer.update_scalars(train_loss, n_iter, 'train')

            model.eval()
            val_losses = []

            if 'random_region' in settings.keys():
                dataset_val.generate_region()
            for _ in range(cfg.n_iters_val):
                source, target =  next(iterator_val)
                with torch.no_grad():
                    output = model(source.to(cfg.device), dataset_val.coord_dict)
                    loss = loss_fcn(output.view(cfg.batch_size,-1), target.view(cfg.batch_size,-1).to(cfg.device))
                    val_loss = {'total': loss.item(), 'valid': loss.item()}

                val_losses.append(list(val_loss.values()))

            val_losses_save.append(val_loss['total'])
            output, debug_dict = model(source.to(cfg.device), dataset_val.coord_dict, return_debug=True)

            if True:
                torch.save(debug_dict, os.path.join(cfg.snapshot_dir,'debug_dict.pt'))
                torch.save(dataset_val.coord_dict,os.path.join(cfg.snapshot_dir,'coord_dict.pt'))
                torch.save(output, os.path.join(cfg.snapshot_dir,'output.pt'))
                torch.save(target, os.path.join(cfg.snapshot_dir,'target.pt'))
                np.savetxt(os.path.join(cfg.snapshot_dir,'losses_val.txt'),np.array(val_losses_save))
                np.savetxt(os.path.join(cfg.snapshot_dir,'losses_train.txt'),np.array(train_losses_save))

            val_loss = torch.tensor(val_losses).mean(dim=0)
            val_loss = dict(zip(train_loss.keys(), val_loss))

            early_stop.update(val_loss['total'], n_iter, model_save=model)

            writer.update_scalars(val_loss, n_iter, 'val')

            if cfg.early_stopping:
                writer.update_scalar('val', 'loss_gradient', early_stop.criterion_diff, n_iter)

            if cfg.eval_timesteps:
                model.eval()
                create_snapshot_image(model, dataset_val, '{:s}/images/iter_{:d}'.format(cfg.snapshot_dir, n_iter))

        if n_iter % cfg.save_model_interval == 0:
            save_ckpt('{:s}/ckpt/{:d}.pth'.format(cfg.snapshot_dir, n_iter), [],
                      [(str(n_iter), n_iter, model, optimizer)])

        if n_iter in final_models:
            savelist.append((str(n_iter), n_iter, copy.deepcopy(model), copy.deepcopy(optimizer)))


        if cfg.early_stopping and early_stop.terminate:
            model = early_stop.best_model
            break



    save_ckpt('{:s}/ckpt/best.pth'.format(cfg.snapshot_dir), [],
              [(str(n_iter), n_iter, early_stop.best_model, optimizer)])

    save_ckpt('{:s}/ckpt/final.pth'.format(cfg.snapshot_dir), [], savelist)

    writer.close()


if __name__ == "__main__":
    train_transformer()
