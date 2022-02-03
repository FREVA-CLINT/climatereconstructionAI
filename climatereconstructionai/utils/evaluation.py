import h5py
import torch
import numpy as np
import pandas as pd
import imageio
import matplotlib.pyplot as plt
from dateutil import parser
from fpdf import FPDF
from numpy import ma
import os.path

from .. import config as cfg
from . import metrics as metrics


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
    image = ma.masked_array(image, mask)

    mask = ma.masked_array(mask, mask)

    for c in range(output.shape[1]):
        if cfg.data_types[c] == 'pr':
            vmin, vmax = (0, 5)
        elif cfg.data_types[c] == 'tas':
            vmin, vmax = (-10, 35)
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


def plot_data(time_series_dict, subplot, plot=False):
    for name, time_series in time_series_dict.items():
        subplot.plot([i for i in range(0, time_series.shape[0])], time_series, label=name)
    if plot:
        subplot.xlabel("Timesteps")
        subplot.ylabel("Precipitation")
    else:
        subplot.set_xlabel("Timesteps")
        subplot.set_ylabel("Precipitation")
    subplot.legend(prop={'size': 6})


def infill(model, dataset, partitions):
    if not os.path.exists(cfg.evaluation_dirs[0]):
        os.makedirs('{:s}'.format(cfg.evaluation_dirs[0]))
    image = []
    mask = []
    gt = []
    output = []

    if partitions > dataset.__len__():
        partitions = dataset.__len__()
    if dataset.__len__() % partitions != 0:
        print("WARNING: The size of the dataset should be dividable by the number of partitions. The last "
              + str(dataset.__len__() % partitions) + " time steps will not be infilled.")
    for split in range(partitions):
        image_part, mask_part, gt_part, rea_images_part, rea_masks_part, rea_gts_part = zip(
            *[dataset[i + split * (dataset.__len__() // partitions)] for i in
              range(dataset.__len__() // partitions)])
        image_part = torch.stack(image_part)
        mask_part = torch.stack(mask_part)
        gt_part = torch.stack(gt_part)
        rea_images_part = torch.stack(rea_images_part)
        rea_masks_part = torch.stack(rea_masks_part)
        rea_gts_part = torch.stack(rea_gts_part)
        # get results from trained network
        with torch.no_grad():
            output_part = model(image_part.to(cfg.device), mask_part.to(cfg.device),
                                rea_images_part.to(cfg.device), rea_masks_part.to(cfg.device))

        image_part = image_part[:, cfg.lstm_steps, :, :, :].to(torch.device('cpu'))
        mask_part = mask_part[:, cfg.lstm_steps, :, :, :].to(torch.device('cpu'))
        gt_part = gt_part[:, cfg.lstm_steps, :, :, :].to(torch.device('cpu'))
        output_part = output_part[:, cfg.lstm_steps, :, :, :].to(torch.device('cpu'))

        # only select first channel
        image_part = torch.unsqueeze(image_part[:, cfg.prev_next_steps, :, :], dim=1)
        gt_part = torch.unsqueeze(gt_part[:, cfg.prev_next_steps, :, :], dim=1)
        mask_part = torch.unsqueeze(mask_part[:, cfg.prev_next_steps, :, :], dim=1)

        image.append(image_part)
        mask.append(mask_part)
        gt.append(gt_part)
        output.append(output_part)

    image = torch.cat(image)
    mask = torch.cat(mask)
    gt = torch.cat(gt)
    output = torch.cat(output)

    # create output_comp
    output_comp = mask * image + (1 - mask) * output

    cvar = {'image': image, 'mask': mask, 'output': output, 'output_comp': output_comp, 'gt': gt}
    write_output_h5(cvar, dataset.data_path, to_netcdf=cfg.convert_to_netcdf)

    return ma.masked_array(gt, mask)[:, 0, :, :], ma.masked_array(output_comp, mask)[:, 0, :, :]

def write_output_h5(cvar, data_path, to_netcdf=False):

    data_type = cfg.data_types[0]

    if to_netcdf:
        import xarray as xr
        ds_src = xr.open_dataset('{}{}'.format(data_path, cfg.img_names[0]))

    for cname in cvar:
        output_name = '{}/{}_{}'.format(cfg.evaluation_dirs[0],cfg.eval_names[0],cname)
        data = cvar[cname].to(torch.device('cpu')).squeeze()

        if to_netcdf:
            ds_dest = ds_src.copy(data={data_type: data})
            ds_dest.to_netcdf(output_name+".nc")

        else:
            dname = ['time', 'lat', 'lon']
            h5 = h5py.File(output_name+".h5", 'w')
            h5.create_dataset(data_type,data=data)
            for dim in range(3):
                h5[data_type].dims[dim].label = dname[dim]
            h5.close()


def create_evaluation_images(name, data_set, create_video=False, save_dir='images/', vmin=0, vmax=5, axis='off'):
    if not os.path.exists(save_dir):
        os.makedirs('{:s}'.format(save_dir))
    for i in range(data_set.shape[0]):
        plt.imshow(np.squeeze(data_set[i, :, :]), vmin=vmin, vmax=vmax)
        plt.axis(axis)
        plt.savefig('{}/{}{}.jpg'.format(save_dir, name, str(i)), bbox_inches='tight', pad_inches=0)
        plt.clf()

    if create_video:
        with imageio.get_writer('{}/movie.gif'.format(save_dir), mode='I', fps=cfg.fps) as writer:
            for i in range(data_set.shape[0]):
                image = imageio.imread('{}/{}{}.jpg'.format(save_dir, name, str(i)))
                writer.append_data(image)


def create_evaluation_maps(map_list, map_names, vmin, vmax):
    # plot and save data
    if len(map_list) > 1:
        fig, axes = plt.subplots(nrows=((len(map_list) // 2) + (len(map_list) % 2)), ncols=2,  figsize=(2 * 4, 3 * ((len(map_list) // 2) + (len(map_list) % 2))))
        fig.patch.set_facecolor('white')
        for i in range((len(map_list) // 2) + (len(map_list) % 2)):
            for j in range(2):
                try:
                    img = axes[i,j].imshow(np.squeeze(map_list[2*i + j]), vmin=vmin, vmax=vmax, cmap='jet', aspect='auto')
                    axes[i,j].set_title(map_names[2*i + j])
                    plt.colorbar(img, ax=axes[i,j])
                except IndexError:
                    if (len(map_list) // 2) + (len(map_list) % 2) == 1:
                        img = axes[j].imshow(np.squeeze(map_list[j]), vmin=vmin, vmax=vmax, cmap='jet',
                                                aspect='auto')
                        axes[j].set_title(map_names[j])
                        plt.colorbar(img, ax=axes[j])
    else:
        img = plt.imshow(np.squeeze(map_list[0]), vmin=vmin, vmax=vmax, cmap='jet', aspect='auto')
        plt.title(map_names[0])
        plt.colorbar(img)
    plt.savefig('{}/{}.jpg'.format(cfg.evaluation_dirs[0], map_names[0]), bbox_inches='tight', pad_inches=0)
    plt.clf()
    plt.close('all')


def create_evaluation_report(gt, outputs):
    # define dicts for time series
    max_timeseries = {}
    min_timeseries = {}
    mean_timeseries = {}
    fldcor_timeseries = {}
    rmse_timeseries = {}
    rmse_over_mean_timeseries = {}

    # define arrays for dataframe
    data_sets = ['GT']
    rmses = ['0.0']
    rmses_over_mean = ['0.0']
    time_cors = ['1.0']
    total_prs = [int(metrics.total_sum(gt))]
    mean_fld_cors = ['1.0']
    fld_cor_total_sum = ['1.0']

    # define output metrics
    for output_name, output in outputs.items():
        # append values
        data_sets.append(output_name)
        rmses.append('%.5f' % metrics.rmse(gt, output))
        rmses_over_mean.append('%.5f' % metrics.rmse_over_mean(gt, output))
        time_cors.append('%.5f' % metrics.timcor(gt, output))
        total_prs.append(int(metrics.total_sum(output)))
        mean_fld_cors.append('%.5f' % metrics.timmean_fldor(gt, output))
        fld_cor_total_sum.append('%.5f' % metrics.fldor_timsum(gt, output))
        # calculate time series
        max_timeseries[output_name] = metrics.max_timeseries(output)
        min_timeseries[output_name] = metrics.min_timeseries(output)
        mean_timeseries[output_name] = metrics.mean_timeseries(output)
        fldcor_timeseries[output_name] = metrics.fldcor_timeseries(gt, output)
        rmse_timeseries[output_name] = metrics.rmse_timeseries(gt, output)
        rmse_over_mean_timeseries[output_name] = metrics.rmse_over_mean_timeseries(gt, output)

    timcor_maps = []
    rmse_maps = []
    sum_maps = [metrics.sum_map(gt)]
    timcor_names = []
    rmse_names = []
    sum_names = ['Sum_GT']


    for output_name, output in outputs.items():
        timcor_maps.append(metrics.timcor_map(gt, output))
        rmse_maps.append(metrics.rmse_map(gt, output))
        sum_maps.append(metrics.sum_map(output))
        timcor_names.append('TimCor_{}'.format(output_name))
        rmse_names.append('RMSe_{}'.format(output_name))
        sum_names.append('Sum_{}'.format(output_name))

    total_maps = [timcor_maps, rmse_maps, sum_maps]
    total_names = [timcor_names, rmse_names, sum_names]
    vmins = [0,0,0]
    vmaxs = [1,0.1,1000]
    for i in range(len(total_maps)):
        create_evaluation_maps(total_maps[i], total_names[i], vmins[i], vmaxs[i])

    # set GT time series
    max_timeseries['Ground Truth'] = metrics.max_timeseries(gt)
    min_timeseries['Ground Truth'] = metrics.min_timeseries(gt)
    mean_timeseries['Ground Truth'] = metrics.mean_timeseries(gt)

    # create dataframe for metrics
    df = pd.DataFrame()
    df['Data Set'] = data_sets
    df['RMSE'] = rmses
    df['RMSE over mean'] = rmses_over_mean
    df['Time Correlation'] = time_cors
    df['Total Precipitation'] = total_prs
    df['Mean Field Correlation'] = mean_fld_cors
    df['Field Correlation of total Field Sum'] = fld_cor_total_sum

    # create time series plots
    fig, ((ax0, ax1), (ax2, ax3), (ax4, ax5)) = plt.subplots(nrows=3, ncols=2, figsize=(2 * 4, 9))
    ax0.set_title('Max values')
    plot_data(max_timeseries, ax0)
    ax1.set_title('Min values')
    plot_data(min_timeseries, ax1)
    ax2.set_title('Mean values')
    plot_data(mean_timeseries, ax2)
    ax3.set_title('Field Cor vs GT')
    plot_data(fldcor_timeseries, ax3)
    ax4.set_title('RMSEs')
    plot_data(rmse_timeseries, ax4)
    ax5.set_title('RMSEs over mean')
    plot_data(rmse_over_mean_timeseries, ax5)
    fig.tight_layout()
    plt.savefig(cfg.evaluation_dirs[0] + '/ts.png', dpi=300)
    plt.clf()

    # Create PDF plot
    labels = []
    data = []
    for output_name, output in outputs.items():
        labels.append(output_name)
        data.append(np.sum(output, axis=(1, 2)) / 3600)
    labels.append('GT')
    data.append(np.sum(gt, axis=(1, 2)) / 3600)
    plt.hist(data, bins=cfg.PDF_BINS, label=labels, edgecolor='black')
    plt.title('Probabilistic density Histogram')
    plt.xlabel('Total precipitation fall')
    plt.ylabel('Number of hours')
    plt.legend()
    plt.xscale("log")

    plt.savefig(cfg.evaluation_dirs[0] + '/pdf.png', dpi=300)
    plt.clf()

    # create PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_xy(0, 0)
    pdf.set_font('arial', 'B', 16)
    pdf.cell(60, 40)
    pdf.cell(75, 30, "Statistical evaluation metrics", 0, 2, 'C')
    pdf.cell(90, 5, " ", 0, 2, 'C')
    pdf.cell(-53)
    pdf.set_font('arial', 'B', 12)
    pdf.cell(25, 10, 'Data Set', 1, 0, 'C')
    pdf.cell(25, 10, 'RMSE', 1, 0, 'C')
    pdf.cell(35, 10, 'RMSE ov. mean', 1, 0, 'C')
    pdf.cell(25, 10, 'Time Cor', 1, 0, 'C')
    pdf.cell(25, 10, 'Total PR', 1, 0, 'C')
    pdf.cell(30, 10, 'Mean Fld Cor', 1, 0, 'C')
    pdf.cell(30, 10, 'Fld Cor Sum', 1, 2, 'C')
    pdf.cell(-165)
    pdf.set_font('arial', '', 12)
    for i in range(0, len(df)):
        pdf.cell(25, 10, '%s' % (df['Data Set'].iloc[i]), 1, 0, 'C')
        pdf.cell(25, 10, '%s' % (str(df['RMSE'].iloc[i])), 1, 0, 'C')
        pdf.cell(35, 10, '%s' % (str(df['RMSE over mean'].iloc[i])), 1, 0, 'C')
        pdf.cell(25, 10, '%s' % (str(df['Time Correlation'].iloc[i])), 1, 0, 'C')
        pdf.cell(25, 10, '%s' % (str(df['Total Precipitation'].iloc[i])), 1, 0, 'C')
        pdf.cell(30, 10, '%s' % (str(df['Mean Field Correlation'].iloc[i])), 1, 0, 'C')
        pdf.cell(30, 10, '%s' % (str(df['Field Correlation of total Field Sum'].iloc[i])), 1, 2, 'C')
        pdf.cell(-165)
    pdf.cell(-20)
    pdf.cell(130, 10, " ", 0, 2, 'C')

    pdf.add_page()

    pdf.set_font('arial', 'B', 16)
    pdf.cell(50)
    pdf.cell(75, 10, "Time Series with smoothin factor {}".format(cfg.smoothing_factor), 0, 2, 'C')
    pdf.cell(90, 5, " ", 0, 2, 'C')
    pdf.cell(-60)
    pdf.image(cfg.evaluation_dirs[0] + '/ts.png', x=None, y=None, w=208, h=240, type='', link='')

    pdf.add_page()

    pdf.set_font('arial', 'B', 16)
    pdf.cell(50)
    pdf.cell(75, 30, "Probabilistic Density Function", 0, 2, 'C')
    pdf.cell(90, 5, " ", 0, 2, 'C')
    pdf.cell(-60)
    pdf.image(cfg.evaluation_dirs[0] + '/pdf.png', x=None, y=None, w=208, h=218, type='', link='')

    pdf.add_page()

    width = 100
    if len(total_maps[0]) > 1:
        width = 200
    pdf.set_font('arial', 'B', 16)
    pdf.cell(50)
    pdf.cell(75, 30, "RMSE Maps", 0, 2, 'C')
    pdf.cell(90, 5, " ", 0, 2, 'C')
    pdf.cell(-55)
    pdf.image('{}/RMSe_{}.jpg'.format(cfg.evaluation_dirs[0], cfg.eval_names[0]), x=None, y=None,  w=width, h=((len(total_maps[0]) // 2) + (len(total_maps[0]) % 2)) * 75, type='', link='')

    pdf.add_page()

    width = 100
    if len(total_maps[1]) > 1:
        width = 200
    pdf.set_font('arial', 'B', 16)
    pdf.cell(50)
    pdf.cell(75, 30, "TimCor Maps", 0, 2, 'C')
    pdf.cell(90, 5, " ", 0, 2, 'C')
    pdf.cell(-55)
    pdf.image('{}/TimCor_{}.jpg'.format(cfg.evaluation_dirs[0], cfg.eval_names[0]), x=None, y=None, w=width, h=((len(total_maps[1]) // 2) + (len(total_maps[1]) % 2)) * 75, type='', link='')

    pdf.add_page()

    width = 100
    if len(total_maps[2]) > 1:
        width = 200
    pdf.set_font('arial', 'B', 16)
    pdf.cell(50)
    pdf.cell(75, 30, "Sum Maps", 0, 2, 'C')
    pdf.cell(90, 5, " ", 0, 2, 'C')
    pdf.cell(-55)
    pdf.image('{}/Sum_{}.jpg'.format(cfg.evaluation_dirs[0], 'GT'), x=None, y=None, w=width, h=((len(total_maps[2]) // 2) + (len(total_maps[2]) % 2)) * 75, type='', link='')

    report_name = ''
    for name in cfg.eval_names:
        report_name += name
    pdf.output('{}/{}.pdf'.format(cfg.evaluation_dirs[0],report_name), 'F')
