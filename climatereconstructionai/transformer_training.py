import os
import numpy as np
import torch
import torch.nn as nn
import json

import torch.multiprocessing
from torch.utils.data import DataLoader
from tqdm import tqdm
import netCDF4 as netcdf

from .utils import twriter_t, early_stopping
from .utils.io import save_ckpt
from .utils.netcdfloader_samples import NetCDFLoader_lazy, InfiniteSampler

class vorticity_calculator():
    def __init__(self, grid_file_path, device='cpu') -> None:
        
        self.device=device
        dset = netcdf.Dataset(grid_file_path)

        self.zonal_normal_primal_edge = torch.from_numpy(dset['zonal_normal_primal_edge'][:].data).to(device)
        self.meridional_normal_primal_edge = torch.from_numpy(dset['meridional_normal_primal_edge'][:].data).to(device)
        self.edges_of_vertex = torch.from_numpy(dset['edges_of_vertex'][:].data.T-1).clamp(min=0).long().to(device)
        self.dual_edge_length = torch.from_numpy(dset['dual_edge_length'][:].data).to(device)
        self.edge_vertices = torch.from_numpy(dset['edge_vertices'][:].data).long().to(device)-1

        self.dual_area = torch.tensor(dset['dual_area'][:].data, device=device)

        edge_dual_normal_cartesian_x = dset['edge_dual_normal_cartesian_x'][:].data
        edge_dual_normal_cartesian_y = dset['edge_dual_normal_cartesian_y'][:].data
        edge_dual_normal_cartesian_z = dset['edge_dual_normal_cartesian_z'][:].data

        edge_middle_cartesian_x = dset['edge_middle_cartesian_x'][:].data
        edge_middle_cartesian_y = dset['edge_middle_cartesian_y'][:].data
        edge_middle_cartesian_z = dset['edge_middle_cartesian_z'][:].data

        cartesian_x_vertices = dset['cartesian_x_vertices'][:].data
        cartesian_y_vertices = dset['cartesian_y_vertices'][:].data
        cartesian_z_vertices = dset['cartesian_z_vertices'][:].data

        cartesian_vertices = torch.tensor([cartesian_x_vertices,cartesian_y_vertices,
                                            cartesian_z_vertices],device=device).T

        edge_dual_normal_cartesian = torch.tensor([edge_dual_normal_cartesian_x,edge_dual_normal_cartesian_y,
                                            edge_dual_normal_cartesian_z],device=device).T

        edge_middle_cartesian = torch.tensor([edge_middle_cartesian_x,edge_middle_cartesian_y,
                                        edge_middle_cartesian_z],device=device).T

        
        self.nout = edge_middle_cartesian[self.edges_of_vertex]-cartesian_vertices.unsqueeze(dim=1)
        self.signorient = torch.sign((self.nout*edge_dual_normal_cartesian[self.edges_of_vertex]).sum(axis=-1))

        self.nout = self.nout.to(device)
        self.signorient = self.signorient.to(device)
        self.coords_v = torch.stack([torch.tensor(dset['vlon'][:].data, device=device), torch.tensor(dset['vlat'][:].data, device=device)])
        dset.close()

    def get_vorticity(self, normalVelocity):
        vort = ((normalVelocity*self.dual_edge_length)[self.edges_of_vertex]*self.signorient).sum(axis=-1)/self.dual_area
        return vort
    

    def get_vorticity_from_uv(self, u, v):
        normalVectorx = self.zonal_normal_primal_edge
        normalVectory = self.meridional_normal_primal_edge

        normalVelocity = u * normalVectorx + v * normalVectory

        return self.get_vorticity(normalVelocity)
    

    def get_vorticity_from_ds(self, ds, ts):
        u = torch.from_numpy(ds['u'][:].data[ts,0].astype(np.float32))
        v = torch.from_numpy(ds['v'][:].data[ts,0].astype(np.float32))

        return self.get_vorticity_from_uv(u,v)
    

    def get_vorticity_from_edge_indices(self, global_edge_indices, u, v):
        b,n = global_edge_indices.shape
        n_edges_global = self.zonal_normal_primal_edge.shape[-1]
        n_vertices_global = self.edges_of_vertex.shape[0]

        global_edge_indices_b = global_edge_indices + (torch.arange(b, device=self.device)*n_edges_global).view(b,1)
        global_edge_indices_b_red = global_edge_indices_b.unique(return_counts=True)[1].max()>1

        v_indices = self.edge_vertices[:,global_edge_indices].transpose(0,1).reshape(b,-1)
        v_indices_b = v_indices + (torch.arange(b, device=self.device)*n_vertices_global).view(b,1)

        u_global = torch.zeros((b*n_edges_global),device=self.device)
        v_global = torch.zeros((b*n_edges_global),device=self.device)
        u_global[global_edge_indices_b.view(-1)] = u.view(-1)
        v_global[global_edge_indices_b.view(-1)] = v.view(-1)
        u_global = u_global.view(b,n_edges_global)
        v_global = v_global.view(b,n_edges_global)

        unique, inverse, counts = torch.unique(v_indices_b, return_counts=True, return_inverse=True)
        non_valid = unique[(counts!=6)]
        mask = torch.isin(v_indices_b, non_valid)
       # valid_b = torch.bucketize(valid, (torch.arange(b)*n_vertices_global))
       # n_valid = valid_b.unique(return_counts=True)[1]

        v_indices_fov = v_indices_b
        self.v_indices_fov = v_indices_fov

        normalVelocity = u_global*self.zonal_normal_primal_edge + v_global*self.meridional_normal_primal_edge

        edges_of_vertex = self.edges_of_vertex[v_indices]
        normalVelocity_con = torch.gather(normalVelocity, dim=1, index=edges_of_vertex.view(b,-1))
        normalVelocity_con = normalVelocity_con.view(b,-1,6)
        #normalVelocity_con = normalVelocity[torch.from_numpy(self.edges_of_vertex[v_indices_fov])]

        edge_length_con = self.dual_edge_length[edges_of_vertex]

        signorient_con = self.signorient[v_indices]

        dual_area_con = self.dual_area[v_indices]

        vort = ((normalVelocity_con*edge_length_con)*signorient_con).sum(axis=-1)/dual_area_con

        vort = vort.view(b,-1,1,1)

        coords_v = self.coords_v[:,v_indices].transpose(1,0)
        return vort, coords_v, mask

def add_vorticity(vort_cal, tensor_dict, global_edge_indices):
    u = tensor_dict['u']
    v = tensor_dict['v']

    u = u[:,:,:,0] if u.dim()==4 else u
    v = v[:,:,:,0] if v.dim()==4 else v
    tensor_dict['vort'], coords_vort, non_valid_mask = vort_cal.get_vorticity_from_edge_indices(global_edge_indices, u, v)
    return tensor_dict, non_valid_mask, coords_vort

class GaussLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.Gauss = torch.nn.GaussianNLLLoss()

    def forward(self, output, target, non_valid_mask):
        output_valid = output[~non_valid_mask]
        target_valid = target[~non_valid_mask]
        loss =  self.Gauss(output_valid[:,:,0],target_valid,output_valid[:,:,1])
        return loss

class L1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.L1Loss()

    def forward(self, output, target, non_valid_mask):
        output_valid = output[~non_valid_mask,:,0].squeeze()
        target_valid = target[~non_valid_mask].squeeze()
        loss = self.loss(output_valid,target_valid)
        return loss

class L1Loss_rel(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.L1Loss()

    def forward(self, output, target):
        abs_loss = ((output[:,:,:,0] - target)/target).abs()
        loss = abs_loss.clamp(max=1)
        loss = loss.mean()
        return loss
    
class TVLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output_hr):
        
        loss = (output_hr[:,1:] - output_hr[:,:-1]).abs().mean() + (output_hr[:,:,1:] - output_hr[:,:,:-1]).abs().mean()

        return loss

class TVLoss_rel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output_hr):
        
        rel_diff1 = ((output_hr[:,1:] - output_hr[:,:-1])/output_hr[:,1:]).abs()
        rel_diff1 = rel_diff1.clamp(max=1)

        rel_diff2 = ((output_hr[:,:,1:] - output_hr[:,:,:-1])/output_hr[:,:,1:]).abs()
        rel_diff2 = rel_diff2.clamp(max=1)

        loss = (rel_diff1.mean() + rel_diff2.mean())

        return loss

class DictLoss(nn.Module):
    def __init__(self, loss_fcn_list, factor_list):
        super().__init__()
        self.loss_fcns = loss_fcn_list
        self.factor_list = factor_list


    def forward(self, output, target, non_valid_mask, lambdas):
        loss_dict = {}
        total_loss = 0

        for k, loss_fcn in enumerate(self.loss_fcns):
            f = self.factor_list[k]
            for var in output.keys():
                lambda_var = lambdas[var]
                loss = lambda_var*f*loss_fcn(output[var], target[var], non_valid_mask[var])
                loss_dict[f'{var}_{str(loss_fcn._get_name())}'] = loss.item()
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
                                model_settings['normalization'],
                                random_region=random_region,
                                p_dropout_source=training_settings['p_dropout_source'],
                                p_dropout_target=training_settings['p_dropout_target'],
                                sampling_mode=training_settings['sampling_mode'],
                                save_sample_path=sample_dir_train,
                                coordinate_pert=training_settings['coordinate_pertubation'],
                                index_range_source=training_settings['index_range_source'] if 'index_range_source' in training_settings else None,
                                index_offset_target=training_settings['index_offset_target'] if 'index_offset_target' in training_settings else 0,
                                rel_coords=training_settings['rel_coords'] if 'rel_coords' in training_settings else False,
                                sample_for_norm=training_settings['sample_for_norm'] if 'sample_for_norm' in training_settings else None,
                                lazy_load=training_settings['lazy_load'] if 'lazy_load' in training_settings else False,
                                rotate_cs=training_settings['rotate_cs'] if 'rotate_cs' in training_settings else False)
    
    dataset_val = NetCDFLoader_lazy(source_files_val, 
                                target_files_val,
                                training_settings['variables_source'],
                                training_settings['variables_target'],
                                dataset_train.norm_dict,
                                random_region=random_region,
                                p_dropout_source=training_settings['p_dropout_source'],
                                p_dropout_target=training_settings['p_dropout_target'],
                                sampling_mode=training_settings['sampling_mode'],
                                save_sample_path=sample_dir_val,
                                coordinate_pert=0,
                                index_range_source=training_settings['index_range_source'] if 'index_range_source' in training_settings else None,
                                index_offset_target=training_settings['index_offset_target'] if 'index_offset_target' in training_settings else 0,
                                rel_coords=training_settings['rel_coords'] if 'rel_coords' in training_settings else False,
                                sample_for_norm=training_settings['sample_for_norm'] if 'sample_for_norm' in training_settings else None,
                                lazy_load=training_settings['lazy_load'] if 'lazy_load' in training_settings else False,
                                rotate_cs=training_settings['rotate_cs'] if 'rotate_cs' in training_settings else False)
    

    
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
    

    model_settings['normalization'] = dataset_train.norm_dict
    model_settings_path = os.path.join(model_settings['model_dir'],'model_settings.json')
    with open(model_settings_path, 'w') as f:
        json.dump(model_settings, f, indent=4)

    model = model.to(device)
    
    calc_vort=False
    if 'vort_loss' in training_settings.keys() and training_settings['vort_loss']:
        if 'grid_file' in training_settings.keys() and len(training_settings['grid_file']):
            calc_vort=True
            vort_calc = vorticity_calculator(training_settings['grid_file'], device=device)


    loss_fcns = []
    factors = []
    if training_settings["gauss_loss"]:
        loss_fcns.append(GaussLoss())
        factors.append(1)
    else:
        loss_fcns.append(L1Loss())
        factors.append(1)

    if "lambda_l1_rel" in training_settings.keys() and training_settings["lambda_l1_rel"]>0:
        factors.append(training_settings["lambda_l1_rel"])
        loss_fcns.append(L1Loss_rel())

    calc_reg_loss=False
    if "lambda_tv_loss_rel" in training_settings.keys() and training_settings["lambda_tv_loss_rel"]>0:
        f_tv = training_settings["lambda_tv_loss_rel"]
        loss_fcn_reg = TVLoss_rel()
        calc_reg_loss=True

    elif "lambda_tv_loss" in training_settings.keys() and training_settings["lambda_tv_loss"]>0:
        f_tv = training_settings["lambda_tv_loss"]
        loss_fcn_reg = TVLoss()
        calc_reg_loss=True

    dict_loss_fcn = DictLoss(loss_fcns, factors)

   
    early_stop = early_stopping.early_stopping(training_settings['early_stopping_delta'], training_settings['early_stopping_patience'])
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=training_settings['lr'])

    lr_scheduler = CosineWarmupScheduler(optimizer, training_settings["T_warmup"], training_settings['max_iter'])

    start_iter = 0
    
    spatial_dim_var_target = model.model_settings['spatial_dims_var_target']

    if 'lambdas' in training_settings.keys():
        lambdas = training_settings['lambdas']
    else:
        vars = model.model_settings['variables_target']
        lambdas = dict(zip(vars, [1]*len(vars)))

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

        source, target, coords_source, coords_target, target_indices = [dict_to_device(x, device) for x in next(iterator_train)]

        output,_, output_reg_hr, non_valid_mask = model(source, coords_source, coords_target)

        optimizer.zero_grad()

        if calc_vort:
            spatial_dim_uv = [k for k,v in spatial_dim_var_target.items() if 'u' in v][0]
            uv_dim_indices = target_indices[spatial_dim_uv]
            output, non_valid_mask_vort, _ = add_vorticity(vort_calc, output, uv_dim_indices)
            non_valid_mask['vort'] = non_valid_mask_vort

            if 'vort' not in target.keys():
                target = add_vorticity(vort_calc, target, uv_dim_indices)[0]

        loss, train_loss_dict = dict_loss_fcn(output, target, non_valid_mask, lambdas)

        if calc_reg_loss:
            reg_loss = f_tv*loss_fcn_reg(output_reg_hr)
            loss += reg_loss
            train_loss_dict['reg_loss'] = reg_loss.item()

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

                source, target, coords_source, coords_target, target_indices = [dict_to_device(x, device) for x in next(iterator_val)]

                with torch.no_grad():
                    output, output_reg_lr, output_reg_hr, non_valid_mask = model(source, coords_source, coords_target)

                    if calc_vort:
                        uv_dim_indices = target_indices[spatial_dim_uv]
                        output, non_valid_mask_vort, coords_vort = add_vorticity(vort_calc, output, uv_dim_indices)
                        non_valid_mask['vort'] = non_valid_mask_vort

                        if 'vort' not in target.keys():
                            target = add_vorticity(vort_calc, target, uv_dim_indices)[0]
                    else:
                        coords_vort = None
                    loss, val_loss_dict = dict_loss_fcn(output, target, non_valid_mask, lambdas)

                    if calc_reg_loss:
                        reg_loss = f_tv*loss_fcn_reg(output_reg_hr)
                        loss += reg_loss
                        val_loss_dict['reg_loss'] = reg_loss.item()

                    val_loss_dict['total'] = loss.item()

                val_losses.append(list(val_loss_dict.values()))
            
            val_loss = torch.tensor(val_losses).mean(dim=0)
            val_loss = dict(zip(train_loss_dict.keys(), val_loss))

            val_losses_save.append(val_loss_dict['total'])
            debug_dict = {}
            if training_settings['save_debug']:
                torch.save(debug_dict, os.path.join(log_dir,'debug_dict.pt'))
                torch.save(coords_source,os.path.join(log_dir,'coords_source.pt'))
                torch.save(coords_target,os.path.join(log_dir,'coords_target.pt'))
                torch.save(coords_vort,os.path.join(log_dir,'coords_vort.pt'))
                torch.save(output, os.path.join(log_dir,'output.pt'))
                torch.save(output_reg_hr, os.path.join(log_dir,'output_reg_hr.pt'))
                torch.save(output_reg_lr, os.path.join(log_dir,'output_reg_lr.pt'))
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
            save_ckpt('{:s}/{:d}.pth'.format(ckpt_dir, n_iter), dataset_train.norm_dict,
                      [(str(n_iter), n_iter, model, optimizer)])

        if training_settings['early_stopping'] and early_stop.terminate:
            model = early_stop.best_model
            break

    save_ckpt('{:s}/best.pth'.format(ckpt_dir), dataset_train.norm_dict,
              [(str(n_iter), n_iter, early_stop.best_model, optimizer)])

    writer.close()