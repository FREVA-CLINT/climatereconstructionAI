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
from tqdm import tqdm

plt.rcParams.update({'font.size': 16})


def create_snapshot_image(model, dataset, filename):
    data_dict = {}
    data_dict["image"], data_dict["mask"], data_dict["gt"], index = zip(*[dataset[int(i)] for i in cfg.eval_timesteps])

    for key in data_dict.keys():
        data_dict[key] = torch.stack(data_dict[key]).to(cfg.device)

    with torch.no_grad():
        data_dict["output"] = model(data_dict["image"], data_dict["mask"])

    if cfg.predict_diff:
        data_dict["output"] += data_dict["image"]
        data_dict["gt"] += data_dict["image"]

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

    for j in range(n_cols):
        axes[0, j].text(0.4, 1, index[j], size=24, transform=axes[0, j].transAxes, color="white")

    k = 0
    for key in keys:
        for c in range(data_dict[key].shape[2]):

            if cfg.vlim is None:
                vmin = data_dict[key][:, :, c, :, :].min().item()
                vmax = data_dict[key][:, :, c, :, :].max().item()
            else:
                vmin = cfg.vlim[0]
                vmax = cfg.vlim[1]

            axes[k, 0].text(-0.8, 0.5, key + " " + str(c) + "\n" + "{:.3e}".format(vmin) + "\n" + "{:.3e}".format(vmax),
                            size=24, va="center", transform=axes[k, 0].transAxes, color="white")

            for j in range(n_cols):
                axes[k, j].axis("off")
                axes[k, j].imshow(np.squeeze(data_dict[key][j][cfg.recurrent_steps, c, :, :]), vmin=vmin, vmax=vmax)

            k += 1

    plt.subplots_adjust(wspace=0.012, hspace=0.012)
    plt.savefig(filename + '.jpg', bbox_inches='tight', pad_inches=0)
    plt.clf()
    plt.close('all')


def get_batch_size(parameters, n_samples, image_sizes):
    if cfg.maxmem is None:
        partitions = cfg.partitions
    else:
        model_size = 0
        for parameter in parameters:
            model_size += sys.getsizeof(parameter.storage())
        model_size = 3.5 * n_samples * model_size / 1e6
        data_size = 4 * n_samples * np.sum([np.prod(size) for size in image_sizes]) * 5 / 1e6
        partitions = int(np.ceil((model_size + data_size) / cfg.maxmem))

    if partitions > n_samples:
        partitions = n_samples

    if partitions != 1:
        print("The data will be split in {} partitions...".format(partitions))

    return int(np.ceil(n_samples / partitions))


def infill(model, dataset, eval_path, output_names, data_stats, xr_dss, i_model, data_types):
    if not os.path.exists(cfg.evaluation_dirs[0]):
        os.makedirs('{:s}'.format(cfg.evaluation_dirs[0]))

    steady_mask = load_steadymask(cfg.steady_mask_data_dict, cfg.device)

    data_dict = {'image': [], 'mask': [], 'gt': [], 'output': [], 'infilled': []}

    for split in tqdm(range(dataset.__len__())):

        data_dict["image"], data_dict["mask"], data_dict["gt"], index = next(dataset)

        if split == 0 and cfg.create_graph:
            writer = SummaryWriter(log_dir=cfg.log_dir)
            writer.add_graph(model, [data_dict["image"], data_dict["mask"], data_dict["gt"]])
            writer.close()

        # get results from trained network
        with torch.no_grad():
            data_dict["output"] = model(data_dict["image"].to(cfg.device), data_dict["mask"].to(cfg.device))

        if cfg.predict_diff:
            data_dict["output"] += data_dict["image"]
            data_dict["gt"] += data_dict["image"]

        for key in ('image', 'mask', 'gt', 'output'):
            data_dict[key] = data_dict[key][:, cfg.recurrent_steps, :, :, :].to(torch.device('cpu'))

        for key in ('image', 'mask', 'gt'):
            data_dict[key] = data_dict[key][:, cfg.gt_channels, :, :]

        if steady_mask is not None:
            for key in ('image', 'gt', 'output'):
                data_dict[key][:, :, steady_mask.type(torch.bool)] = np.nan

        data_dict["infilled"] = (1 - data_dict["mask"])
        data_dict["infilled"] *= data_dict["output"]
        data_dict["infilled"] += data_dict["mask"] * data_dict["image"]

        data_dict["image"] /= data_dict["mask"]

        create_outputs(data_dict, eval_path, output_names, data_stats, xr_dss, i_model, split, index, data_types)

        if cfg.progress_fwd is not None:
            cfg.progress_fwd[0]('Infilling...',
                                int(cfg.progress_fwd[2] * (cfg.progress_fwd[1] + (split + 1) / dataset.__len__())))

    return output_names


def create_outputs(data_dict, eval_path, output_names, data_stats, xr_dss, i_model, split, index, data_types):

    m_label = "." + str(i_model)
    suffix = m_label + "-" + str(split + 1)

    if cfg.n_target_data == 0:
        cnames = ["gt", "mask", "image", "output", "infilled"]
        pnames = ["image", "infilled"]
    else:
        cnames = ["gt", "output"]
        pnames = ["gt", "output"]

    for j in range(len(eval_path)):

        i_data = -cfg.n_target_data + j
        data_type = data_types[i_data]

        for cname in cnames:

            rootname = '{}_{}'.format(eval_path[j], cname)
            if rootname not in output_names:
                output_names[rootname] = []
            output_names[rootname] += [rootname + suffix + ".nc"]
            ds = xr_dss[i_data][1].copy()

            if cfg.normalize_data and cname != "mask":
                data_dict[cname][:, j, :, :] = renormalize(data_dict[cname][:, j, :, :],
                                                           data_stats["mean"][i_data], data_stats["std"][i_data])

            ds[data_type] = xr.DataArray(data_dict[cname].to(torch.device('cpu')).detach().numpy()[:, j, :, :],
                                         dims=xr_dss[i_data][2], coords=xr_dss[i_data][3])
            ds["time"] = xr_dss[i_data][0]["time"].values[index]

            ds = reformat_dataset(xr_dss[i_data][0], ds, data_type)

            for var in xr_dss[i_data][0].keys():
                if "time" in xr_dss[i_data][0][var].dims:
                    ds[var] = xr_dss[i_data][0][var].isel(time=index)
                else:
                    ds[var] = xr_dss[i_data][0][var]

            ds.attrs["history"] = "Infilled using CRAI (Climate Reconstruction AI: " \
                                  "https://github.com/FREVA-CLINT/climatereconstructionAI)\n" + ds.attrs["history"]
            ds.to_netcdf(output_names[rootname][-1])

        for time_step in cfg.plot_results:
            if time_step in index:
                output_name = '{}_{}{}_{}.png'.format(eval_path[j], "combined", m_label, time_step)
                plot_data(xr_dss[i_data][1].coords,
                          [data_dict[p][time_step - index[0], j, :, :].squeeze() for p in pnames],
                          ["Original", "Reconstructed"], output_name, data_type,
                          str(xr_dss[i_data][0]["time"][time_step].values),
                          *cfg.dataset_format["scale"])
