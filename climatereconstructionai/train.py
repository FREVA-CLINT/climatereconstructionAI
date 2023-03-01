import os

import torch
import torch.multiprocessing

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy
import numpy as np

from . import config as cfg
from .loss import get_loss
from .metrics.get_metrics import get_metrics
from .model.net import CRAINet

from .utils.io import load_ckpt, load_model, save_ckpt
from .utils.netcdfloader import NetCDFLoader, InfiniteSampler, load_steadymask
from .utils.profiler import load_profiler
from .utils.io import read_input_file_as_dict, get_parameters_as_dict
from .utils import twriter, early_stopping, evaluation


def train(arg_file=None):
    
    arg_parser = cfg.set_train_args(arg_file)
    
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
    if cfg.input_file is not None:
        input_dict = read_input_file_as_dict(cfg.input_file)
        writer.set_hparams(get_parameters_as_dict(input_dict, arg_parser))

    if cfg.lstm_steps:
        time_steps = cfg.lstm_steps
    elif cfg.gru_steps:
        time_steps = cfg.gru_steps
    elif cfg.channel_steps:
        time_steps = cfg.channel_steps
    else:
        time_steps = 0

    # create data sets
    dataset_train = NetCDFLoader(cfg.data_root_dir, cfg.data_names, cfg.mask_dir, cfg.mask_names, 'train',
                                 cfg.data_types, time_steps)
    dataset_val = NetCDFLoader(cfg.data_root_dir, cfg.val_names, cfg.mask_dir, cfg.mask_names, 'val', cfg.data_types,
                               time_steps)
    iterator_train = iter(DataLoader(dataset_train, batch_size=cfg.batch_size,
                                     sampler=InfiniteSampler(len(dataset_train)),
                                     num_workers=cfg.n_threads, multiprocessing_context='fork'))
    iterator_val = iter(DataLoader(dataset_val, batch_size=cfg.batch_size,
                                   sampler=InfiniteSampler(len(dataset_val)),
                                   num_workers=cfg.n_threads, multiprocessing_context='fork'))

    steady_mask = load_steadymask(cfg.mask_dir, cfg.steady_masks, cfg.data_types, cfg.device)

    if cfg.n_target_data == 0:
        stat_target = None
    else:
        stat_target = {"mean": dataset_train.img_mean[-cfg.n_target_data:],
                       "std": dataset_train.img_std[-cfg.n_target_data:]}

    # define network model
    if len(cfg.image_sizes) - cfg.n_target_data > 1:
        model = CRAINet(img_size=cfg.image_sizes[0],
                        enc_dec_layers=cfg.encoding_layers[0],
                        pool_layers=cfg.pooling_layers[0],
                        in_channels=2 * cfg.channel_steps + 1,
                        out_channels=cfg.out_channels,
                        fusion_img_size=cfg.image_sizes[1],
                        fusion_enc_layers=cfg.encoding_layers[1],
                        fusion_pool_layers=cfg.pooling_layers[1],
                        fusion_in_channels=(len(cfg.image_sizes) - 1 - cfg.n_target_data
                                            ) * (2 * cfg.channel_steps + 1),
                        bounds=dataset_train.bounds).to(cfg.device)
    else:
        model = CRAINet(img_size=cfg.image_sizes[0],
                        enc_dec_layers=cfg.encoding_layers[0],
                        pool_layers=cfg.pooling_layers[0],
                        in_channels=2 * cfg.channel_steps + 1,
                        out_channels=cfg.out_channels,
                        bounds=dataset_train.bounds).to(cfg.device)

    # define learning rate
    if cfg.finetune:
        lr = cfg.lr_finetune
        model.freeze_enc_bn = True
    else:
        lr = cfg.lr

    
    early_stop = early_stopping.early_stopping(cfg.early_stopping_dict)

    loss_comp = get_loss.LossComputation()

    # define optimizer and loss functions
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    if cfg.lr_scheduler_patience is not None:
        lr_scheduler = ReduceLROnPlateau(optimizer, 'min', patience=cfg.lr_scheduler_patience)

    # define start point
    start_iter = 0
    if cfg.resume_iter:
        ckpt_dict = load_ckpt('{}/ckpt/{}.pth'.format(cfg.snapshot_dir, cfg.resume_iter), cfg.device)
        start_iter = load_model(ckpt_dict, model, optimizer)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Starting from iter ', start_iter)
    
    elif cfg.pretrained_model:
        ckpt_dict = load_ckpt(cfg.pretrained_model, cfg.device)
        load_model(ckpt_dict, model, optimizer)
 
    prof = load_profiler(start_iter)

    if cfg.multi_gpus:
        model = torch.nn.DataParallel(model)

    i = cfg.max_iter - (cfg.n_final_models - 1) * cfg.final_models_interval
    final_models = range(i, cfg.max_iter + 1, cfg.final_models_interval)

    n_iter_val = 1
    savelist = []
    pbar = tqdm(range(start_iter, cfg.max_iter))
    prof.start()
    for i in pbar:

        n_iter = i + 1
        lr_val = optimizer.param_groups[0]['lr']
        pbar.set_description("lr = {:.1e}".format(lr_val))

        # train model
        model.train()
        image, mask, gt = [x.to(cfg.device) for x in next(iterator_train)]
        output = model(image, mask)

        train_loss = loss_comp.get_loss(mask, steady_mask, output, gt)

        optimizer.zero_grad()
        train_loss['total'].backward()
        optimizer.step()
        if (cfg.log_interval and n_iter % cfg.log_interval == 0):
            writer.update_scalars(train_loss, n_iter, 'train')

            if cfg.train_metrics is not None and cfg.input_file is not None:
                metric_dict = get_metrics(mask, steady_mask, output, gt, 'train')
                writer.update_hparams(metric_dict, n_iter)


        if (cfg.val_interval and n_iter % cfg.val_interval == 0):
            
            model.eval()
            val_losses = []
            for _ in range(cfg.n_iters_val): 
                image, mask, gt = [x.to(cfg.device) for x in next(iterator_val)]
                with torch.no_grad():
                    output = model(image, mask)
                val_losses.append(list(loss_comp.get_loss(mask, steady_mask, output, gt).values()))

            val_loss = torch.tensor(val_losses).mean(dim=0)
            val_loss = dict(zip(train_loss.keys(),val_loss))

            early_stop.update(val_loss['total'].item() , n_iter, model_save=model)
            writer.update_scalar('val', 'loss_gradient', early_stop.criterion_diff , n_iter)
            
            if (n_iter_val % cfg.log_val_interval == 0):
                
                writer.update_scalars(val_loss, n_iter, 'val')

                if cfg.save_snapshot_image:
                    fig = evaluation.create_snapshot_image(model, dataset_val, '{:s}/images/iter_{:d}'.format(cfg.snapshot_dir, n_iter))
                    writer.add_figure(fig, n_iter, 'snapshot')
            
                if cfg.val_metrics is not None and cfg.input_file is not None:
                    metric_dict = get_metrics(mask, steady_mask, output, gt, 'val')
                    writer.update_hparams(metric_dict, n_iter)

                if cfg.plot_plots:
                    fig = evaluation.create_correlation_plot(mask, steady_mask, output, gt)
                    writer.add_figure(fig, n_iter, name_tag='plot/val/correlation')

                    fig = evaluation.create_error_dist_plot(mask, steady_mask, output, gt)
                    writer.add_figure(fig, n_iter, name_tag='plot/val/error_dist')

                if cfg.plot_distributions:

                    errors_dists = evaluation.get_all_error_distributions(mask, steady_mask, output, gt, domain="valid", num_samples=1000)
                    [writer.add_distributions(error_dist, n_iter,f'dist/test/{name}') for error_dist, name in zip(errors_dists,['error','relative error','abs error','relative abs error'])]

                if cfg.plot_maps:
                    error_maps= evaluation.get_all_error_maps(mask, steady_mask, output, gt, domain="valid", num_samples=3)
                    [writer.add_figure(error_map, n_iter,f'map/val/{name}') for error_map, name in zip(error_maps,['error','relative error','abs error','relative abs error'])]

                    fig = evaluation.create_map(mask, steady_mask, output, gt, num_samples=3)
                    writer.add_figure(fig, n_iter, name_tag='map/val/values')

            if cfg.lr_scheduler_patience is not None:
                lr_scheduler.step(val_loss['total'])
        
            n_iter_val+=1
            
        if n_iter % cfg.save_model_interval == 0:
            save_ckpt('{:s}/ckpt/{:d}.pth'.format(cfg.snapshot_dir, n_iter), stat_target,
                      [(str(n_iter), n_iter, model, optimizer)])

        if n_iter in final_models:
            savelist.append((str(n_iter), n_iter, copy.deepcopy(model), copy.deepcopy(optimizer)))

        prof.step()

        if cfg.early_stopping and early_stop.terminate:
            metric_dict = {'iterations': n_iter, 'iterations_best_model': early_stop.global_iter_best}
            writer.update_hparams(metric_dict, n_iter)
            prof.stop()
            break
        
    prof.stop()

    model = early_stop.best_model
    save_ckpt('{:s}/ckpt/best.pth'.format(cfg.snapshot_dir, early_stop.global_iter_best), stat_target,
                      [(str(n_iter), n_iter, early_stop.best_model, optimizer)])

    if cfg.test_names:
        model.eval()
        dataset_test = NetCDFLoader(cfg.data_root_dir, cfg.test_names, cfg.mask_dir, cfg.mask_names, 'infill', cfg.data_types,
                                time_steps)
        batch_size_eval = 100
        iterator_test = iter(DataLoader(dataset_test, batch_size=batch_size_eval,
                                    sampler=InfiniteSampler(len(dataset_val)),
                                    num_workers=cfg.n_threads, multiprocessing_context='fork'))
        
        image, mask, gt = [x.to(cfg.device) for x in next(iterator_test)]

        with torch.no_grad():
            output = model(image, mask)

        metric_dict = get_metrics(mask, steady_mask, output, gt, 'test')
        writer.update_hparams(metric_dict, n_iter)

        errors_dists = evaluation.get_all_error_distributions(mask, steady_mask, output, gt, domain="valid", num_samples=1000)
        [writer.add_distributions(error_dist, n_iter,f'dist/test/{name}') for error_dist, name in zip(errors_dists,['error','relative error','abs error','relative abs error'])]

        error_maps= evaluation.get_all_error_maps(mask, steady_mask, output, gt, domain="valid", num_samples=3)
        [writer.add_figure(error_map, n_iter,f'map/test/{name}') for error_map, name in zip(error_maps,['error','relative error','abs error','relative abs error'])]
       
        fig = evaluation.create_correlation_plot(mask, steady_mask, output, gt)
        writer.add_figure(fig, n_iter, name_tag='plot/test/correlation')

        fig = evaluation.create_error_dist_plot(mask, steady_mask, output, gt)
        writer.add_figure(fig, n_iter, name_tag='plot/test/error_dist')

        fig = evaluation.create_map(mask, steady_mask, output, gt, num_samples=3)
        writer.add_figure(fig, n_iter, name_tag='map/test/values')

    writer.close()

    save_ckpt('{:s}/ckpt/final.pth'.format(cfg.snapshot_dir), stat_target, savelist)


if __name__ == "__main__":
    train()
