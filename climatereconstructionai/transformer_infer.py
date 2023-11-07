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

def infer(model, settings, model_hparams={}):
 
    print("* Number of GPUs: ", torch.cuda.device_count())

    log_dir = settings['log_dir']
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    ckpt_dir = os.path.join(log_dir,'ckpts')
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)

    writer = twriter_t.writer(log_dir)

    device = settings['device']
    writer.set_hparams(model_hparams)

    if 'random_region' not in settings.keys():
        random_region = None
    else:
        random_region = settings['random_region']
    
    batch_size = settings['batch_size']

    source_files = check_get_data_files(settings['data']['data_names_source'], root_path = settings['root_dir'])
    target_files = check_get_data_files(settings['data']['data_names_target'], root_path = settings['root_dir'])        
    


    with open(settings["norm_stats"],'r') as file:
        stat_dict = json.load(file)


    dataset = NetCDFLoader(source_files, 
                                 target_files,
                                 settings['variables'],
                                 settings['coord_names'],
                                 random_region=random_region,
                                 apply_img_norm=settings['apply_img_norm'],
                                 normalize_data=settings['normalize_data'],
                                 stat_dict=stat_dict,
                                 p_input_dropout=settings['input_dropout'],
                                 sampling_mode=settings['sampling_mode'],
                                 n_points=settings['n_points'],
                                 coordinate_pert=settings['coordinate_pertubation'])
    
    iterator = iter(DataLoader(dataset,
                                batch_size=batch_size,
                                num_workers=settings['n_workers'], 
                                pin_memory=True if device == 'cuda' else False,
                                pin_memory_device=device))


    model = model.to(device)

    start_iter = 0

    pbar = tqdm(range(start_iter, 1))

    train_losses_save = []
    val_losses_save = []
    lrs = []
    with torch.no_grad():
        for i in pbar:
        
            n_iter = i + 1

            model.eval()

            source, target, coord_dict = next(iterator)

            coord_dict['rel'] = dict_to_device(coord_dict['rel'], device)
            source = source.to(device)
            target = target.to(device)

            output, debug_dict = model(source, coord_dict, return_debug=True)

      
            if settings['save_debug']:
                torch.save(debug_dict, os.path.join(log_dir,'debug_dic.pt'))
                torch.save(coord_dict,os.path.join(log_dir,'coord_dict.pt'))
                torch.save(output, os.path.join(log_dir,'output.pt'))
                torch.save(target, os.path.join(log_dir,'target.pt'))
                torch.save(source, os.path.join(log_dir,'source.pt'))
                np.savetxt(os.path.join(log_dir,'losses_val.txt'),np.array(val_losses_save))
                np.savetxt(os.path.join(log_dir,'losses_train.txt'),np.array(train_losses_save))
                np.savetxt(os.path.join(log_dir,'lrs.txt'),np.array(lrs))
        

