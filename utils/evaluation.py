import h5py
import torch
import netCDF4
import numpy as np
import pandas as pd
import imageio
import matplotlib.pyplot as plt
import calendar

from mpl_toolkits.axes_grid1 import make_axes_locatable
from netCDF4 import Dataset
from fpdf import FPDF
from cdo import *
from numpy import ma
import config as cfg

sys.path.append('./')

import utils.metrics as metrics


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

    cvar = [image, mask, output, output_comp, gt]
    cname = ['image', 'mask', 'output', 'output_comp', 'gt']
    dname = ['time', 'lat', 'lon']
    for x in range(0, 5):
        h5 = h5py.File('%s' % (cfg.evaluation_dirs[0] + cname[x]), 'w')
        h5.create_dataset(cfg.data_types[0], data=cvar[x].to(torch.device('cpu')))
        for dim in range(0, 3):
            h5[cfg.data_types[0]].dims[dim].label = dname[dim]
        h5.close()

    return ma.masked_array(gt, mask)[:, 0, :, :], ma.masked_array(output_comp, mask)[:, 0, :, :]


def convert_all_to_netcdf():
    # convert to netCDF files
    convert_h5_to_netcdf(True, 'image')
    convert_h5_to_netcdf(False, 'gt')
    convert_h5_to_netcdf(False, 'output')
    convert_h5_to_netcdf(False, 'output_comp')


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


def create_evaluation_report(gt, outputs):
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

    # create dataframe for metrics
    df = pd.DataFrame()
    df['Data Set'] = data_sets
    df['RMSE'] = rmses
    df['RMSE over mean'] = rmses_over_mean
    df['Time Correlation'] = time_cors
    df['Total Precipitation'] = total_prs
    df['Mean Field Correlation'] = mean_fld_cors
    df['Field Correlation of total Field Sum'] = fld_cor_total_sum

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

    plt.savefig(cfg.evaluation_dirs[0] + 'pdf.png', dpi=300)
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
    pdf.cell(75, 30, "Probabilistic Density Function", 0, 2, 'C')
    pdf.cell(90, 5, " ", 0, 2, 'C')
    pdf.cell(-60)
    pdf.image(cfg.evaluation_dirs[0] + 'pdf.png', x=None, y=None, w=208, h=218, type='', link='')

    report_name = ''
    for name in cfg.eval_names:
        report_name += name
    pdf.output('evaluation/reports/{}.pdf'.format(report_name), 'F')


def init_font():
    from matplotlib import font_manager

    font_dirs = ['../fonts/']
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)

    for font_file in font_files:
        font_manager.fontManager.addfont(font_file)


def plot_ts(title, file_name, time_series_dict, time, unit):
    init_font()
    plt.rcParams.update({'font.family': 'Times New Roman'})
    index = 0
    for name, time_series in time_series_dict.items():
        if name=='Ground Truth':
            param='k:'
        else:
            param='{}-'.format(cfg.graph_colors[index])
            index+=1
        plt.plot(range(len(time_series)), time_series, param, label=name)
        plt.xlabel("Year {}".format(time[0].year))
        plt.ylabel(title + " in " + unit)
    ax = plt.gca()
    ax.set_xticks(range(12))
    ax.set_xticklabels([calendar.month_abbr[time[i].month] for i in range(len(time)) if time[i].month != time[i-1].month or i == 0])
    plt.xticks(rotation=55)
    plt.legend()
    plt.savefig('evaluation/graphs/' + file_name + '.pdf', bbox_inches="tight")
    plt.clf()


def create_evaluation_graphs(gt, outputs):
    data = Dataset('{}/{}/{}'.format(cfg.data_root_dir, 'test_large', cfg.img_names[0]))
    time = data.variables['time']
    time = netCDF4.num2date(time[:], time.units)

    # define dicts for time series
    max_timeseries = {}
    min_timeseries = {}
    mean_timeseries = {}
    rmse_timeseries = {}
    rmse_over_mean_timeseries = {}
    new_rmse_over_mean = {}
    #fldcor_timeseries = {}

    # set GT time series
    max_timeseries['Ground Truth'] = metrics.max_timeseries(gt, time)
    min_timeseries['Ground Truth'] = metrics.min_timeseries(gt, time)
    mean_timeseries['Ground Truth'] = metrics.mean_timeseries(gt, time)

    # define output metrics
    for output_name, output in outputs.items():
        # calculate time series
        max_timeseries[output_name] = metrics.max_timeseries(output, time)
        min_timeseries[output_name] = metrics.min_timeseries(output, time)
        mean_timeseries[output_name] = metrics.mean_timeseries(output, time)
        rmse_timeseries[output_name] = metrics.rmse_timeseries(gt, output, time)
        rmse_over_mean_timeseries[output_name] = metrics.rmse_over_mean_timeseries(gt, output, time)
        new_rmse_over_mean[output_name] = np.abs(mean_timeseries[output_name] - mean_timeseries['Ground Truth'])
        #fldcor_timeseries[output_name] = metrics.fldcor_timeseries(gt, output, time)



    # create time series plots
    plot_ts('Maximum', 'MaxTS{}x{}'.format(cfg.image_sizes[0], cfg.image_sizes[0]), max_timeseries, time, 'mm/h')
    plot_ts('Minimum', 'MinTS{}x{}'.format(cfg.image_sizes[0], cfg.image_sizes[0]), min_timeseries, time, 'mm/h')
    plot_ts('Mean', 'MeanTS{}x{}'.format(cfg.image_sizes[0], cfg.image_sizes[0]),  mean_timeseries, time, 'mm/h')
    plot_ts('RMSE', 'RMSETS{}x{}'.format(cfg.image_sizes[0], cfg.image_sizes[0]), rmse_timeseries, time, 'mm/h')
    plot_ts('ME', 'METS{}x{}'.format(cfg.image_sizes[0], cfg.image_sizes[0]), rmse_over_mean_timeseries, time, 'mm/h')
    plot_ts('NewME', 'NewMETS{}x{}'.format(cfg.image_sizes[0], cfg.image_sizes[0]), new_rmse_over_mean, time, 'mm/h')
    #plot_ts('ME', 'FldCorTS{}x{}'.format(cfg.image_sizes[0], cfg.image_sizes[0]), fldcor_timeseries, time, 'mm/h')


def create_evaluation_maps(gt, outputs):
    init_font()
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 22})
    timcor_maps = []
    rmse_maps = []
    timcor_names = []
    rmse_names = []
    if cfg.create_sum_maps:
        sum_maps = [metrics.sum_map(gt)]
        sum_names = ['Sum GT']
    else:
        sum_maps = []
        sum_names =[]
    for output_name, output in outputs.items():
        if cfg.create_sum_maps:
            sum_maps.append(metrics.sum_map(output))
            sum_names.append('Sum {}'.format(output_name))
        if cfg.create_timcor_maps:
            timcor_maps.append(metrics.timcor_map(gt, output))
            timcor_names.append('TimCor {}'.format(output_name))
        if cfg.create_rmse_maps:
            rmse_maps.append(metrics.rmse_map(gt, output))
            rmse_names.append('RMSE {}'.format(output_name))

    map_lists = []
    map_names = []
    if cfg.create_sum_maps:
        map_lists.append(sum_maps)
        map_names.append(sum_names)
    if cfg.create_rmse_maps:
        map_lists.append(rmse_maps)
        map_names.append(rmse_names)
    if cfg.create_timcor_maps:
        map_lists.append(timcor_maps)
        map_names.append(timcor_names)
    for i in range(len(map_lists)):
        minimum = np.min(map_lists[i])
        if 'RMSE' in map_names[i][0]:
            minimum = 0.02
            maximum = 0.1
        elif 'TimCor' in map_names[i][0]:
            maximum = 0.8
        elif 'Sum' in map_names[i][0]:
            maximum = 800
        else:
            maximum = np.max(map_lists[i])
        for j in range(len(map_lists[i])):
            # plot and save data
            img = plt.imshow(np.squeeze(map_lists[i][j]), vmin=minimum, vmax=maximum, cmap='Blues', aspect='auto')
            #plt.title(map_names[i][j])
            plt.xlabel("km")
            plt.ylabel("km")
            ax = plt.gca()
            ax.set_yticks([i+12 for i in range(cfg.image_sizes[0]) if i % 100 == 0])
            ax.set_yticklabels([i for i in range(cfg.image_sizes[0]) if i % 100 == 0][::-1])
            ax.set_xticks([i for i in range(cfg.image_sizes[0]) if i % 100 == 0])
            ax.set_xticklabels([i for i in range(cfg.image_sizes[0]) if i % 100 == 0])
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='10%', pad=0.6)
            cb = plt.colorbar(img, cax=cax, orientation='vertical')
            if 'TimCor' not in map_names[i][j]:
                cax.set_xlabel('mm/h', labelpad=20)
                cax.xaxis.set_label_position('bottom')
            plt.savefig('{}/{}{}x{}.pdf'.format('evaluation/maps', map_names[i][j], cfg.image_sizes[0], cfg.image_sizes[0]), bbox_inches='tight')
            plt.clf()
            plt.close('all')


def evaluate_selected_samples(self, dates=None):
    cdo = Cdo()
    if dates is None:
        dates = ['2017-01-12T23', '2017-04-17T15', '2017-05-02T12', '2017-05-13T12', '2017-06-04T03',
                 '2017-06-29T16', '2017-07-12T14', '2017-09-02T13']
    i = 0
    for date in dates:
        cdo.select('date=' + date, input=self.eval_save_dir + 'image.nc',
                   output=self.eval_save_dir + 'imagetmp' + str(i) + '.nc')
        cdo.select('date=' + date, input=self.eval_save_dir + 'output_comp.nc',
                   output=self.eval_save_dir + 'output_comptmp' + str(i) + '.nc')
        cdo.select('date=' + date, input=self.eval_save_dir + 'gt.nc',
                   output=self.eval_save_dir + 'gttmp' + str(i) + '.nc')
    cdo.mergetime(input=self.eval_save_dir + 'imagetmp*', output=self.eval_save_dir + 'image_selected.nc')
    cdo.mergetime(input=self.eval_save_dir + 'output_comptmp*',
                  output=self.eval_save_dir + 'output_comp_selected.nc')
    cdo.mergetime(input=self.eval_save_dir + 'gttmp*', output=self.eval_save_dir + 'gt_selected.nc')
    os.system('rm ' + self.eval_save_dir + '*tmp*')

    self.create_evaluation_images(file='image_selected.nc')
    self.create_evaluation_images(file='gt_selected.nc')
    self.create_evaluation_images(file='output_comp_selected.nc')


def convert_h5_to_netcdf(self, create_structure_template, file):
    if create_structure_template:
        os.system('ncdump ' + self.test_dir + '*.h5 > ' + self.eval_save_dir + 'tmp_dump.txt')
        os.system(
            'sed "/.*' + self.variable + ' =.*/{s///;q;}" ' + self.eval_save_dir + 'tmp_dump.txt > ' + self.eval_save_dir + 'structure.txt')
        os.system('rm ' + self.eval_save_dir + 'tmp_dump.txt')
    cdo = Cdo()
    os.system('cat ' + self.eval_save_dir + 'structure.txt >> ' + self.eval_save_dir + file + '.txt')
    os.system(
        'ncdump -v ' + self.variable + ' ' + self.eval_save_dir + file + ' | sed -e "1,/data:/d" >> ' + self.eval_save_dir + file + '.txt')
    os.system('ncgen -o ' + self.eval_save_dir + 'output-tmp ' + self.eval_save_dir + file + '.txt')
    cdo.setgrid(self.test_dir + '*.h5', input=self.eval_save_dir + 'output-tmp',
                output=self.eval_save_dir + file + '.nc')
    os.system('rm ' + self.eval_save_dir + file + '.txt ' + self.eval_save_dir + 'output-tmp')
