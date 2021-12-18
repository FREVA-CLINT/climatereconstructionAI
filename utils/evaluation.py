import h5py
import torch
import netCDF4
import numpy as np
import pandas as pd
import imageio
import matplotlib.pyplot as plt
from dateutil import parser
from netCDF4 import Dataset
from fpdf import FPDF
from cdo import *
from numpy import ma
import config as cfg

sys.path.append('./')

import utils.metrics as metrics


def create_snapshot_image(model, dataset, filename, lstm_steps):
    image, mask, gt = zip(*[dataset[int(i)] for i in cfg.eval_timesteps])

    image = torch.stack(image).to(cfg.device)
    mask = torch.stack(mask).to(cfg.device)
    gt = torch.stack(gt).to(cfg.device)
    with torch.no_grad():
        output = model(image.to(cfg.device), mask.to(cfg.device)).to(cfg.device)

    # select last element of lstm sequence as evaluation element
    image = image[:, 0, cfg.gt_channels, :, :].to(torch.device('cpu'))
    gt = gt[:, 0, cfg.gt_channels, :, :].to(torch.device('cpu'))
    mask = mask[:, 0, cfg.gt_channels, :, :].to(torch.device('cpu'))
    output = output[:, lstm_steps, :, :, :].to(torch.device('cpu'))

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


def get_data(file, var):
    data = Dataset(file)
    time = data.variables['time']
    variable = data.variables[var]
    return variable, time


def plot_data(time_series_dict, subplot):
    for name, time_series in time_series_dict.items():
        subplot.plot([i for i in range(0, time_series.shape[0])], time_series, label=name)
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
        image_part, mask_part, gt_part = zip(
            *[dataset[i + split * (dataset.__len__() // partitions)] for i in
              range(dataset.__len__() // partitions)])
        image_part = torch.stack(image_part)
        mask_part = torch.stack(mask_part)
        gt_part = torch.stack(gt_part)
        # get results from trained network
        with torch.no_grad():
            output_part = model(image_part.to(cfg.device), mask_part.to(cfg.device))

        lstm_steps = output_part.shape[1] - 1

        image_part = image_part[:, lstm_steps, :, :, :].to(torch.device('cpu'))
        mask_part = mask_part[:, lstm_steps, :, :, :].to(torch.device('cpu'))
        gt_part = gt_part[:, lstm_steps, :, :, :].to(torch.device('cpu'))
        output_part = output_part[:, lstm_steps, :, :, :].to(torch.device('cpu'))

        # only select first channel
        image_part = torch.unsqueeze(image_part[:, 0, :, :], dim=1)
        gt_part = torch.unsqueeze(gt_part[:, 0, :, :], dim=1)
        mask_part = torch.unsqueeze(mask_part[:, 0, :, :], dim=1)

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


def create_evaluation_images(self, file, create_video=False, start_date=None, end_date=None):
    if not os.path.exists(self.eval_save_dir + 'images'):
        os.makedirs('{:s}'.format(self.eval_save_dir + 'images'))

    data = Dataset(self.eval_save_dir + file)
    time = data.variables['time']
    time = netCDF4.num2date(time[:], time.units)

    if start_date and end_date:
        start = parser.parse(start_date)
        end = parser.parse(end_date)
        pr = [data.variables[self.variable][i, :, :] for i in range(time.__len__()) if
              time[i] >= start and time[i] <= end]
        time = [time[i] for i in range(time.__len__()) if time[i] >= start and time[i] <= end]
    else:
        pr = data.variables[self.variable][:, :, :]

    for i in range(time.__len__()):
        plt.imshow(np.squeeze(pr[i]), vmin=0, vmax=5)
        plt.axis('off')
        plt.title('Precipitation from ' + str(time[i]))
        plt.savefig(self.eval_save_dir + 'images/' + file + '_' + str(i) + '.jpg')
        plt.clf()

    if create_video:
        with imageio.get_writer(self.eval_save_dir + 'images/' + file + '_movie.gif', mode='I') as writer:
            for i in range(time.__len__()):
                image = imageio.imread(self.eval_save_dir + 'images/' + file + '_' + str(i) + '.jpg')
                writer.append_data(image)


def create_evaluation_report(gt, outputs):
    if cfg.ts_range is None:
        ts_range = (0, gt.shape[0])
    else:
        ts_range = (int(cfg.ts_range[0]), int(cfg.ts_range[1]))

    # define dicts for time series
    max_timeseries = {}
    min_timeseries = {}
    mean_timeseries = {}
    fldcor_timeseries = {}

    # define arrays for dataframe
    data_sets = ['GT']
    rmses = ['0.0']
    time_cors = ['1.0']
    total_prs = [int(metrics.total_sum(gt))]
    mean_fld_cors = ['1.0']
    fld_cor_total_sum = ['1.0']

    # define output metrics
    for output_name, output in outputs.items():
        # append values
        data_sets.append(output_name)
        rmses.append('%.5f' % metrics.rmse(gt, output))
        time_cors.append('%.5f' % metrics.timcor(gt, output))
        total_prs.append(int(metrics.total_sum(output)))
        mean_fld_cors.append('%.5f' % metrics.timmean_fldor(gt, output))
        fld_cor_total_sum.append('%.5f' % metrics.fldor_timsum(gt, output))
        # calculate time series
        max_timeseries[output_name] = metrics.max_timeseries(output[ts_range[0]:ts_range[1]])
        min_timeseries[output_name] = metrics.min_timeseries(output[ts_range[0]:ts_range[1]])
        mean_timeseries[output_name] = metrics.mean_timeseries(output[ts_range[0]:ts_range[1]])
        fldcor_timeseries[output_name] = metrics.fldcor_timeseries(gt, output[ts_range[0]:ts_range[1]])

    # set GT time series
    max_timeseries['Ground Truth'] = metrics.max_timeseries(gt[ts_range[0]:ts_range[1]])
    min_timeseries['Ground Truth'] = metrics.min_timeseries(gt[ts_range[0]:ts_range[1]])
    mean_timeseries['Ground Truth'] = metrics.mean_timeseries(gt[ts_range[0]:ts_range[1]])

    # create dataframe for metrics
    df = pd.DataFrame()
    df['Data Set'] = data_sets
    df['RMSE'] = rmses
    df['Time Correlation'] = time_cors
    df['Total Precipitation'] = total_prs
    df['Mean Field Correlation'] = mean_fld_cors
    df['Field Correlation of total Field Sum'] = fld_cor_total_sum

    # create time series plots
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2)
    ax0.set_title('Max values')
    plot_data(max_timeseries, ax0)
    ax1.set_title('Min values')
    plot_data(min_timeseries, ax1)
    ax2.set_title('Mean values')
    plot_data(mean_timeseries, ax2)
    ax3.set_title('Field Cor vs GT')
    plot_data(fldcor_timeseries, ax3)
    fig.tight_layout()
    plt.savefig(cfg.evaluation_dirs[0] + 'ts.png', dpi=300)
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

    plt.savefig(cfg.evaluation_dirs[0] + 'pdf.png', dpi=300)
    plt.clf()

    # create PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_xy(0, 0)
    pdf.set_font('arial', 'B', 12)
    pdf.cell(60)
    pdf.cell(75, 10, "Statistical evaluation of Ground Truth and Output Comp", 0, 2, 'C')
    pdf.cell(90, 10, " ", 0, 2, 'C')
    pdf.cell(-40)
    pdf.cell(25, 10, 'Data Set', 1, 0, 'C')
    pdf.cell(25, 10, 'RMSE', 1, 0, 'C')
    pdf.cell(25, 10, 'Time Cor', 1, 0, 'C')
    pdf.cell(25, 10, 'Total PR', 1, 0, 'C')
    pdf.cell(30, 10, 'Mean Fld Cor', 1, 0, 'C')
    pdf.cell(30, 10, 'Fld Cor Sum', 1, 2, 'C')
    pdf.cell(-130)
    pdf.set_font('arial', '', 12)
    for i in range(0, len(df)):
        pdf.cell(25, 10, '%s' % (df['Data Set'].iloc[i]), 1, 0, 'C')
        pdf.cell(25, 10, '%s' % (str(df['RMSE'].iloc[i])), 1, 0, 'C')
        pdf.cell(25, 10, '%s' % (str(df['Time Correlation'].iloc[i])), 1, 0, 'C')
        pdf.cell(25, 10, '%s' % (str(df['Total Precipitation'].iloc[i])), 1, 0, 'C')
        pdf.cell(30, 10, '%s' % (str(df['Mean Field Correlation'].iloc[i])), 1, 0, 'C')
        pdf.cell(30, 10, '%s' % (str(df['Field Correlation of total Field Sum'].iloc[i])), 1, 2, 'C')
        pdf.cell(-130)
    pdf.cell(-20)
    pdf.cell(130, 20, " ", 0, 2, 'C')

    pdf.image(cfg.evaluation_dirs[0] + 'ts.png', x=None, y=None, w=208, h=156, type='', link='')
    pdf.image(cfg.evaluation_dirs[0] + 'pdf.png', x=None, y=None, w=208, h=156, type='', link='')
    pdf.output(cfg.evaluation_dirs[0] + 'Report.pdf', 'F')


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
