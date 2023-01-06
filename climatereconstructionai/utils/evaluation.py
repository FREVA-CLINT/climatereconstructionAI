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
    data_dict = {}
    data_dict["image"], data_dict["mask"], data_dict["gt"] = zip(*[dataset[int(i)] for i in cfg.eval_timesteps])

    for key in data_dict.keys():
        data_dict[key] = torch.stack(data_dict[key]).to(cfg.device)

    with torch.no_grad():
        data_dict["output"] = model(data_dict["image"], data_dict["mask"])

    data_dict["infilled"] = data_dict["mask"] * data_dict["image"] + (1 - data_dict["mask"]) * data_dict["output"]

    keys = list(data_dict.keys())
    for key in keys:
        data_dict[key] = data_dict[key].to(torch.device('cpu'))

    # set mask
    data_dict["mask"] = 1 - data_dict["mask"]
    data_dict["image"] = np.ma.masked_array(data_dict["image"], data_dict["mask"])
    data_dict["mask"] = np.ma.masked_array(data_dict["mask"], data_dict["mask"])

    n_rows = sum([data_dict[key].shape[2] for key in keys])
    n_cols = data_dict["image"].shape[0]

    # plot and save data
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(4 * n_cols, 4 * n_rows))
    fig.patch.set_facecolor('black')
    k = 0
    for key in keys:
        for c in range(data_dict[key].shape[2]):

            if cfg.vlim is None:
                vmin = data_dict[key][:, :, c, :, :].min().item()
                vmax = data_dict[key][:, :, c, :, :].max().item()
            else:
                vmin = cfg.vlim[0]
                vmax = cfg.vlim[1]

            axes[k, 0].text(-0.5, 0.5, key + " " + str(c) + "\n" + str(vmin) + "\n" + str(vmax), size=24, va="center",
                            transform=axes[k, 0].transAxes, color="white")

            for j in range(n_cols):
                axes[k, j].axis("off")
                axes[k, j].imshow(np.squeeze(data_dict[key][j][:, c, :, :]), vmin=vmin, vmax=vmax)

            k += 1

    plt.subplots_adjust(wspace=0.012, hspace=0.012)
    plt.savefig(filename + '.jpg', bbox_inches='tight', pad_inches=0)
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

    data_dict = {'image': [], 'mask': [], 'gt': [], 'output': [], 'infilled': []}
    keys = list(data_dict.keys())

    partitions = get_partitions(model.parameters(), dataset.img_length)

    if partitions != 1:
        print("The data will be split in {} partitions...".format(partitions))

    n_elements = dataset.__len__() // partitions
    for split in range(partitions):
        i_start = split * n_elements
        if split == partitions - 1:
            i_end = dataset.__len__()
        else:
            i_end = i_start + n_elements

        for i in range(3):
            data_dict[keys[i]].append(torch.stack([dataset[j][i] for j in range(i_start, i_end)]))

        if split == 0 and cfg.create_graph:
            writer = SummaryWriter(log_dir=cfg.log_dir)
            writer.add_graph(model, [data_dict["image"][-1], data_dict["mask"][-1], data_dict["gt"][-1]])
            writer.close()

        # get results from trained network
        with torch.no_grad():
            data_dict["output"].append(model(data_dict["image"][-1].to(cfg.device),
                                             data_dict["mask"][-1].to(cfg.device)))

        for key in keys[:4]:
            data_dict[key][-1] = data_dict[key][-1][:, cfg.lstm_steps, :, :, :].to(torch.device('cpu'))

    for key in keys[:4]:
        data_dict[key] = torch.cat(data_dict[key])

    steady_mask = load_steadymask(cfg.mask_dir, cfg.steady_masks, cfg.data_types, cfg.device)
    if steady_mask is not None:
        steady_mask = 1 - steady_mask
        for key in ('gt', 'image', 'output'):
            data_dict[key] /= steady_mask

    data_dict["infilled"] = data_dict["mask"] * data_dict["image"] + (1 - data_dict["mask"]) * data_dict["output"]
    data_dict["image"][np.where(data_dict["mask"] == 0)] = np.nan

    return data_dict


def create_outputs(outputs, dataset, eval_path, stat_target, suffix=""):

    if suffix != "":
        suffix = "." + str(suffix)

    if cfg.n_target_data == 0:
        mean_val, std_val = dataset.img_mean[:cfg.out_channels], dataset.img_std[:cfg.out_channels]
        cnames = ["gt", "mask", "image", "output", "infilled"]
        pnames = ["image", "infilled"]
    else:
        mean_val = dataset.img_mean[-cfg.n_target_data:]
        std_val = dataset.img_std[-cfg.n_target_data:]
        cnames = ["gt", "mask", "output"]
        pnames = ["gt", "output"]

    n_out = len(outputs)

    for j in range(len(eval_path)):

        ind = -cfg.n_target_data + j
        data_type = cfg.data_types[ind]

        for cname in cnames:

            output_name = '{}_{}'.format(eval_path[j], cname)

            dss = []
            for i in range(n_out):
                dss.append(dataset.xr_dss[ind][1].copy())

                if cfg.normalize_data and cname != "mask":
                    if cname == "output" and stat_target is not None:
                        outputs[i][cname][:, j, :, :] = renormalize(outputs[i][cname][:, j, :, :],
                                                                    stat_target["mean"][j], stat_target["std"][j])
                    else:
                        outputs[i][cname][:, j, :, :] = renormalize(outputs[i][cname][:, j, :, :],
                                                                    mean_val[j], std_val[j])

                dss[-1][data_type].values = outputs[i][cname].to(torch.device('cpu')).detach().numpy()[:, j, :, :]

                dss[-1] = reformat_dataset(dataset.xr_dss[j][0], dss[-1], data_type)

            ds = xr.concat(dss, dim="time", data_vars="minimal").sortby('time')
            ds.attrs["history"] = "Infilled using CRAI (Climate Reconstruction AI: " \
                                  "https://github.com/FREVA-CLINT/climatereconstructionAI)\n" + ds.attrs["history"]
            ds.to_netcdf(output_name + suffix + ".nc")

        for i in range(n_out):
            output_name = '{}_{}.{}'.format(eval_path[j], "combined", i + 1)
            plot_data(dataset.xr_dss[j][1].coords,
                      [outputs[i][pnames[0]][:, j, :, :], outputs[i][pnames[1]][:, j, :, :]],
                      ["Original", "Reconstructed"], output_name, data_type, cfg.plot_results,
                      *cfg.dataset_format["scale"])
