import os.path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from tensorboardX import SummaryWriter
import xarray as xr
import pandas as pd

from .netcdfchecker import reformat_dataset
from .normalizer import renormalize
from .plotdata import plot_data
from .. import config as cfg
from tqdm import tqdm

plt.rcParams.update({'font.size': 16})


def create_snapshot_image(model, dataset, filename):
    data_dict = {}
    data_dict["image"], data_dict["in_mask"], data_dict["out_mask"], data_dict["gt"], index = zip(*[dataset[int(i)] for i in cfg.eval_timesteps])

    for key in data_dict.keys():
        data_dict[key] = torch.stack(data_dict[key]).to(cfg.device)

    with torch.no_grad():
        data_dict["output"] = model(data_dict["image"], data_dict["in_mask"])

    # data_dict["infilled"] = data_dict["mask"] * data_dict["image"] + (1 - data_dict["mask"]) * data_dict["output"]

    keys = list(data_dict.keys())
    for key in keys:
        data_dict[key] = data_dict[key].to(torch.device('cpu'))

    for key in ('image', 'in_mask', 'output'):
        data_dict[key] = data_dict[key][:, cfg.recurrent_steps, :, :, :]

    for key in ('gt', 'out_mask'):
        data_dict[key] = data_dict[key][:, 0, :, :, :]

    # set mask
    data_dict["in_mask"] = 1 - data_dict["in_mask"]
    data_dict["image"] = np.ma.masked_array(data_dict["image"], data_dict["in_mask"])
    data_dict["in_mask"] = np.ma.masked_array(data_dict["in_mask"], data_dict["in_mask"])

    n_rows = sum([data_dict[key].shape[1] for key in keys])
    n_cols = data_dict["image"].shape[0]

    # plot and save data
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(4 * n_cols, 4 * n_rows))
    fig.patch.set_facecolor('black')

    for j in range(n_cols):
        axes[0, j].text(0.4, 1, index[j], size=24, transform=axes[0, j].transAxes, color="white")

    k = 0
    for key in keys:
        for c in range(data_dict[key].shape[1]):

            if cfg.vlim is None:
                vmin = data_dict[key][:, c, :, :].min().item()
                vmax = data_dict[key][:, c, :, :].max().item()
            else:
                vmin = cfg.vlim[0]
                vmax = cfg.vlim[1]

            axes[k, 0].text(-0.8, 0.5, key + " " + str(c) + "\n" + "{:.3e}".format(vmin) + "\n" + "{:.3e}".format(vmax),
                            size=24, va="center", transform=axes[k, 0].transAxes, color="white")

            for j in range(n_cols):
                axes[k, j].axis("off")
                axes[k, j].imshow(np.squeeze(data_dict[key][j][c, :, :]), vmin=vmin, vmax=vmax)

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


def infill(model, dataset, eval_path, output_names, steady_mask, data_stats, xr_dss, i_model):
    if not os.path.exists(cfg.evaluation_dirs[0]):
        os.makedirs('{:s}'.format(cfg.evaluation_dirs[0]))

    data_dict = {'image': [], 'mask': [], 'gt': [], 'output': []}

    for split in tqdm(range(dataset.__len__())):

        data_dict["image"], data_dict["mask"], _, data_dict["gt"], index = next(dataset)

        if split == 0 and cfg.create_graph:
            writer = SummaryWriter(log_dir=cfg.log_dir)
            writer.add_graph(model, [data_dict["image"], data_dict["mask"]])
            writer.close()

        # get results from trained network
        with torch.no_grad():
            data_dict["output"] = model(data_dict["image"].to(cfg.device), data_dict["mask"].to(cfg.device))

        for key in ('image', 'mask', 'output'):
            data_dict[key] = data_dict[key][:, cfg.recurrent_steps, :, :, :].to(torch.device('cpu'))
        data_dict['gt'] = data_dict['gt'][:, 0, :, :, :].to(torch.device('cpu'))

        for key in ('image', 'mask'):
            data_dict[key] = data_dict[key][:, cfg.gt_channels, :, :]

        if steady_mask is not None:
            for key in ('gt', 'image'):
                data_dict[key][:, ~steady_mask.type(torch.bool)] = np.nan
            data_dict['output'][:, ~np.repeat(steady_mask, cfg.n_pred_steps, axis=0).type(torch.bool)] = np.nan

        data_dict["image"] /= data_dict["mask"]

        if cfg.n_target_data == 0 and cfg.n_pred_steps == 1:
            data_dict["infilled"] = (1 - data_dict["mask"])
            data_dict["infilled"] *= data_dict["output"]
            data_dict["infilled"] += data_dict["mask"] * data_dict["gt"]

        create_outputs(data_dict, eval_path, output_names, data_stats, xr_dss, i_model, split, index)

        if cfg.progress_fwd is not None:
            cfg.progress_fwd[0]('Infilling...',
                                int(cfg.progress_fwd[2] * (cfg.progress_fwd[1] + (split + 1) / dataset.__len__())))

    return output_names


def create_outputs(data_dict, eval_path, output_names, data_stats, xr_dss, i_model, split, index):

    m_label = "." + str(i_model)
    suffix = m_label + "-" + str(split + 1)

    if cfg.n_target_data == 0:
        if cfg.n_pred_steps == 1:
            cnames = ["gt", "mask", "image", "output", "infilled"]
            pnames = ["image", "infilled"]
        else:
            cnames = ["gt", "mask", "image", "output"]
            pnames = ["image", "output"]
    else:
        cnames = ["gt", "output"]
        pnames = ["gt", "output"]

    for j in range(len(eval_path)):

        i_data = -cfg.n_target_data + j
        data_type = cfg.data_types[i_data]
        i_plot = {}

        for cname in cnames:

            rootname = '{}_{}'.format(eval_path[j], cname)
            if rootname not in output_names:
                output_names[rootname] = {}

            if i_model not in output_names[rootname]:
                output_names[rootname][i_model] = []

            output_names[rootname][i_model] += [rootname + suffix + ".nc"]

            ds = xr_dss[i_data]["ds1"].copy()
            dims = xr_dss[i_data]["dims"].copy()
            coords = xr_dss[i_data]["coords"].copy()
            coords["time"] = xr_dss[i_data]["ds"]["time"].values[index]
            if cfg.n_pred_steps > 1 and cname == "output":
                dims = [dims[0]] + ["pred_time"] + dims[1:]
                coords["pred_time"] = np.array(cfg.pred_timestep)
                if cfg.time_freq:
                    coords["pred_time"] = pd.to_timedelta(coords["pred_time"], unit=cfg.time_freq)
                    coords['times'] = (["pred_time", "time"], np.add.outer(coords["pred_time"], coords["time"]))
                i_pred = range(j * cfg.n_pred_steps, (j + 1) * cfg.n_pred_steps)
                i_plot[cname] = i_pred[0]
            else:
                i_pred, i_plot[cname] = j, j

            if cfg.normalize_data and cname != "mask":
                data_dict[cname][:, i_pred] = renormalize(data_dict[cname][:, i_pred],
                                                          data_stats["mean"][i_data], data_stats["std"][i_data])

            ds[data_type] = xr.DataArray(data_dict[cname].detach().numpy()[:, i_pred], dims=dims, coords=coords)
            ds = reformat_dataset(xr_dss[i_data]["ds"], ds, data_type)

            for var in xr_dss[i_data]["ds"].keys():
                if "time" in xr_dss[i_data]["ds"][var].dims:
                    ds[var] = xr_dss[i_data]["ds"][var].isel(time=index)
                else:
                    ds[var] = xr_dss[i_data]["ds"][var]
            
            if "history" in ds.attrs:
                history = "\n" + ds.attrs["history"]
            else:
                history = ""
            ds.attrs["history"] = "Infilled using CRAI (Climate Reconstruction AI: " \
                                  "https://github.com/FREVA-CLINT/climatereconstructionAI)" + history
            ds.to_netcdf(output_names[rootname][i_model][-1])

        for time_step in cfg.plot_results:
            if time_step in index:
                output_name = '{}_{}{}_{}.png'.format(eval_path[j], "combined", m_label, time_step)

                plot_data(xr_dss[i_data]["ds1"].coords,
                          [data_dict[p][time_step - index[0], i_plot[p], :, :].squeeze() for p in pnames],
                          ["Original", "Reconstructed"], output_name, data_type,
                          str(xr_dss[i_data]["ds"]["time"][time_step].values),
                          *cfg.dataset_format["scale"])
