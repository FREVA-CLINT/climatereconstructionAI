import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import os.path
from tensorboardX import SummaryWriter
from .netcdfloader import SteadyMaskLoader
from .netcdfchecker import reformat_dataset
from .plotdata import plot_data
from .normalizer import renormalize

from .. import config as cfg

def create_snapshot_image(model, dataset, filename):
    image, mask, gt, rea_images, rea_masks, rea_gts = zip(*[dataset[int(i)] for i in cfg.eval_timesteps])

    image = torch.stack(image).to(cfg.device)
    mask = torch.stack(mask).to(cfg.device)
    gt = torch.stack(gt).to(cfg.device)
    if rea_images:
        rea_images = torch.stack(rea_images).to(cfg.device)
        rea_masks = torch.stack(rea_masks).to(cfg.device)
        rea_gts = torch.stack(rea_gts).to(cfg.device)

    with torch.no_grad():
        output = model(image, mask, rea_images, rea_masks)

    # select last element of lstm sequence as evaluation element
    image = image[:, cfg.lstm_steps, cfg.gt_channels, :, :].to(torch.device('cpu'))
    gt = gt[:, cfg.lstm_steps, cfg.gt_channels, :, :].to(torch.device('cpu'))
    mask = mask[:, cfg.lstm_steps, cfg.gt_channels, :, :].to(torch.device('cpu'))
    output = output[:, cfg.lstm_steps, :, :, :].to(torch.device('cpu'))

    output_comp = mask * image + (1 - mask) * output

    # set mask
    mask = 1 - mask
    image = np.ma.masked_array(image, mask)

    mask = np.ma.masked_array(mask, mask)

    for c in range(output.shape[1]):
        if cfg.data_types[c] == 'pr':
            vmin, vmax = (0, 5)
        elif cfg.data_types[c] == 'tas':
            vmin, vmax = (-10, 35)
        elif cfg.data_types[c] == 'TXx':
            vmin, vmax = (-50, 50)
        data_list = [image[:, c, :, :], mask[:, c, :, :], output[:, c, :, :], output_comp[:, c, :, :], gt[:, c, :, :]]

        # plot and save data
        fig, axes = plt.subplots(nrows=len(data_list), ncols=image.shape[0], figsize=(20, 20))
        fig.patch.set_facecolor('black')
        for i in range(len(data_list)):
            for j in range(image.shape[0]):
                axes[i, j].axis("off")
                axes[i, j].imshow(np.squeeze(data_list[i][j]), vmin=vmin, vmax=vmax)
        plt.subplots_adjust(wspace=0.012, hspace=0.012)
        plt.savefig(filename + '_' + str(c) + '.jpg', bbox_inches='tight', pad_inches=0)
    plt.clf()
    plt.close('all')

def get_partitions(parameters,length):

    if cfg.maxmem is None:
        partitions = cfg.partitions
    else:
        model_size = 0
        for parameter in parameters:
            model_size += sys.getsizeof(parameter.storage())
        model_size = model_size*length/1e6
        partitions = int(np.ceil(model_size*5/cfg.maxmem))

    if partitions > length:
        partitions = length

    return partitions

def infill(model, dataset, eval_path):
    if not os.path.exists(cfg.evaluation_dirs[0]):
        os.makedirs('{:s}'.format(cfg.evaluation_dirs[0]))
    image = []
    mask = []
    gt = []
    output = []

    partitions = get_partitions(model.parameters(),dataset.img_length)

    if partitions != 1:
        print("The data will be split in {} partitions...".format(partitions))

    n_elements = dataset.__len__() // partitions
    for split in range(partitions):
        data_part = []
        i_start = split*n_elements
        if split == partitions-1:
            i_end = dataset.__len__()
        else:
            i_end = i_start+n_elements
        for i in range(6):
            data_part.append(torch.stack([dataset[j][i] for j in range(i_start,i_end)]))

        # Tensors in data_part: image_part, mask_part, gt_part, rea_images_part, rea_masks_part, rea_gts_part

        if split == 0 and cfg.create_graph:
            writer = SummaryWriter(log_dir=cfg.log_dir)
            writer.add_graph(model, (data_part[0],data_part[1],data_part[3],data_part[4]))
            writer.close()

        # get results from trained network
        with torch.no_grad():
            output_part = model(data_part[0].to(cfg.device), data_part[1].to(cfg.device),
                                data_part[3].to(cfg.device), data_part[4].to(cfg.device))

        # image_part, mask_part, gt_part
        for i in range(3):
            data_part[i] = data_part[i][:, cfg.lstm_steps, :, :, :].to(torch.device('cpu'))
            # only select first channel
            data_part[i] = torch.unsqueeze(data_part[i][:, cfg.prev_next_steps, :, :], dim=1)
        output_part = output_part[:, cfg.lstm_steps, :, :, :].to(torch.device('cpu'))

        image.append(data_part[0])
        mask.append(data_part[1])
        gt.append(data_part[2])
        output.append(output_part)

    image = torch.cat(image)
    mask = torch.cat(mask)
    gt = torch.cat(gt)
    output = torch.cat(output)

    steady_mask = SteadyMaskLoader(cfg.mask_dir, cfg.steady_mask, cfg.data_types[0], cfg.device)
    if not steady_mask is None:
        steady_mask = 1-steady_mask
        image /= steady_mask
        gt /= steady_mask
        output /= steady_mask

    # create output_comp
    output_comp = mask * image + (1 - mask) * output
    image[np.where(mask==0)] = np.nan

    cvar = {'gt': gt, 'mask': mask, 'image': image, 'output': output, 'infilled': output_comp}
    create_outputs(cvar, dataset, 0, eval_path)

def create_outputs(cvar, dataset, ind_data, eval_path):

    data_type = cfg.data_types[ind_data]

    for cname in cvar:
        output_name = '{}_{}'.format(eval_path,cname)

        ds = dataset.xr_dss[1].copy()

        if cfg.normalize_images:
            cvar[cname] = renormalize(cvar[cname], dataset.img_mean[ind_data], dataset.img_std[ind_data])
        ds[data_type].values = cvar[cname].to(torch.device('cpu')).detach().numpy()[:,0,:,:]

        ds = reformat_dataset(dataset.xr_dss[0],ds,data_type)
        ds.attrs["history"] = "Infilled using CRAI (Climate Reconstruction AI: https://github.com/FREVA-CLINT/climatereconstructionAI)\n"+ds.attrs["history"]
        ds.to_netcdf(output_name+".nc")

    output_name = '{}_{}'.format(eval_path,"combined")
    plot_data(dataset.xr_dss[1].coords,[cvar["image"],cvar["infilled"]],["Original","Reconstructed"],output_name,data_type,cfg.plot_results,*cfg.dataset_format["scale"])

