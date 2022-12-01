import os.path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from tensorboardX import SummaryWriter
import xarray as xr

from .netcdfchecker import reformat_dataset
from .netcdfloader import load_steadymask
from .normalizer import renormalize
from .plotdata import plot_data
from .. import config as cfg


def create_snapshot_image(model, dataset, filename):
    images, masks, gts = zip(*[dataset[int(i)] for i in cfg.eval_timesteps])

    images = torch.stack(images).to(cfg.device)
    masks = torch.stack(masks).to(cfg.device)
    gt = torch.stack(gt).to(cfg.device)

    with torch.no_grad():
        output = model(images, masks)

    # select last element of lstm sequence as evaluation element
    images = images[:, cfg.lstm_steps, cfg.gt_channels, :, :].to(torch.device('cpu'))
    gt = gt[:, cfg.lstm_steps, cfg.gt_channels, :, :].to(torch.device('cpu'))
    masks = masks[:, cfg.lstm_steps, cfg.gt_channels, :, :].to(torch.device('cpu'))
    output = output[:, cfg.lstm_steps, :, :, :].to(torch.device('cpu'))

    output_comp = masks * images + (1 - masks) * output

    # set mask
    masks = 1 - masks
    images = np.ma.masked_array(images, masks)
    masks = np.ma.masked_array(masks, masks)

    for c in range(output.shape[1]):

        if cfg.vlim is None:
            vmin = gt[:, c, :, :].min().item()
            vmax = gt[:, c, :, :].max().item()
        else:
            vmin = cfg.vlim[0]
            vmax = cfg.vlim[1]
        data_list = [images[:, c, :, :], masks[:, c, :, :], output[:, c, :, :], output_comp[:, c, :, :], gt[:, c, :, :]]

        # plot and save data
        fig, axes = plt.subplots(nrows=len(data_list), ncols=images.shape[0], figsize=(20, 20))
        fig.patch.set_facecolor('black')
        for i in range(len(data_list)):
            for j in range(images.shape[0]):
                axes[i, j].axis("off")
                axes[i, j].imshow(np.squeeze(data_list[i][j]), vmin=vmin, vmax=vmax)
        plt.subplots_adjust(wspace=0.012, hspace=0.012)
        plt.savefig(filename + '_' + str(c) + '.jpg', bbox_inches='tight', pad_inches=0)
    plt.clf()
    plt.close('all')


def get_partitions(parameters, length):
    if cfg.maxmem is None:
        partitions = cfg.partitions
    else:
        model_size = 0
        for parameter in parameters:
            model_size += sys.getsizeof(parameter.storage())
        model_size = model_size * length / 1e6
        partitions = int(np.ceil(model_size * 5 / cfg.maxmem))

    if partitions > length:
        partitions = length

    return partitions


def infill(model, dataset):
    if not os.path.exists(cfg.evaluation_dirs[0]):
        os.makedirs('{:s}'.format(cfg.evaluation_dirs[0]))
    # images = []
    # masks = []
    # gt = []
    # output = []

    data_dict = {'image': [], 'mask': [], 'gt': [], 'output': [], 'infilled': []}
    keys = list(data_dict.keys())

    partitions = get_partitions(model.parameters(), dataset.img_length)

    if partitions != 1:
        print("The data will be split in {} partitions...".format(partitions))

    n_elements = dataset.__len__() // partitions
    for split in range(partitions):
        # data_part = []
        i_start = split * n_elements
        if split == partitions - 1:
            i_end = dataset.__len__()
        else:
            i_end = i_start + n_elements
        # print(len(dataset[0]))
        # for i in range(3):
        #     data_part.append(torch.stack([dataset[j][i] for j in range(i_start, i_end)]))

        for i in range(3):
            data_dict[keys[i]].append(torch.stack([dataset[j][i] for j in range(i_start, i_end)]))

        # Tensors in data_part: image_part, mask_part, gt_part, rea_images_part, rea_masks_part, rea_gts_part

        if split == 0 and cfg.create_graph:
            writer = SummaryWriter(log_dir=cfg.log_dir)
            writer.add_graph(model, [data_dict["image"][-1], data_dict["mask"][-1], data_dict["gt"][-1]])
            writer.close()

        # get results from trained network
        with torch.no_grad():
            data_dict["output"].append(model(data_dict["image"][-1].to(cfg.device), data_dict["mask"][-1].to(cfg.device)))

        # image_part, mask_part, gt_part
        # for i in range(3):
        #     data_part[i] = data_part[i][:, cfg.lstm_steps, :, :, :].to(torch.device('cpu'))
            # only select first channel
            #data_part[i] = torch.unsqueeze(data_part[i][:, cfg.prev_next_steps, :, :], dim=1)
        # output_part = output_part[:, cfg.lstm_steps, :, :, :].to(torch.device('cpu'))
        for key in keys[:4]:
            data_dict[key][-1] = data_dict[key][-1][:, cfg.lstm_steps, :, :, :].to(torch.device('cpu'))

        #
        #
        # images.append(data_part[0])
        # masks.append(data_part[1])
        # gt.append(data_part[2])
        # output.append(output_part)
    for key in keys[:4]:
        data_dict[key] = torch.cat(data_dict[key])
    # images = torch.cat(images)
    # masks = torch.cat(mask)
    # gt = torch.cat(gt)
    # output = torch.cat(output)

    steady_mask = load_steadymask(cfg.mask_dir, cfg.steady_masks, cfg.data_types, cfg.device)
    if steady_mask is not None:
        steady_mask = 1 - steady_mask
        for key in ('gt', 'image', 'output'):
            data_dict[key] /= steady_mask

    # create output_comp
    data_dict["infilled"] = data_dict["mask"] * data_dict["image"] + (1 - data_dict["mask"]) * data_dict["output"]
    data_dict["image"][np.where(data_dict["mask"] == 0)] = np.nan

    return data_dict


def create_outputs(outputs, dataset, eval_path, suffix=""):

    if suffix != "":
        suffix = "." + str(suffix)

    n_out = len(outputs)

    for j in range(len(eval_path)):

        data_type = cfg.data_types[j]

        for cname in outputs[0]:

            output_name = '{}_{}'.format(eval_path[j], cname)

            dss = []
            for i in range(n_out):
                dss.append(dataset.xr_dss[j][1].copy())

                if cfg.normalize_data and cname != "masks":
                    outputs[i][cname][:,j,:,:] = renormalize(outputs[i][cname][:,j,:,:],
                                                    dataset.img_mean[j], dataset.img_std[j])
                dss[-1][data_type].values = outputs[i][cname].to(torch.device('cpu')).detach().numpy()[:, j, :, :]

                dss[-1] = reformat_dataset(dataset.xr_dss[j][0], dss[-1], data_type)

            ds = xr.concat(dss, dim="time", data_vars="minimal").sortby('time')
            ds.attrs["history"] = "Infilled using CRAI " \
                                  "(Climate Reconstruction AI: https://github.com/FREVA-CLINT/climatereconstructionAI)\n" \
                                  + ds.attrs["history"]
            ds.to_netcdf(output_name + suffix + ".nc")

        for i in range(n_out):
            output_name = '{}_{}.{}'.format(eval_path[j], "combined", i + 1)
            plot_data(dataset.xr_dss[j][1].coords, [outputs[i]["image"][:,j,:,:], outputs[i]["infilled"][:,j,:,:]],
                      ["Original", "Reconstructed"], output_name, data_type, cfg.plot_results, *cfg.dataset_format["scale"])
