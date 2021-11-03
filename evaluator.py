import h5py
import torch
from dateutil import parser
import netCDF4
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import imageio
from fpdf import FPDF
import os
os.getcwd()


def get_data(file):
    data = Dataset(file)
    time = data.variables['time']
    pr = data.variables['pr']
    return pr, time


def plot_data(file, start=0, end=None, label=''):
    data, time = get_data(file=file)
    if not end:
        end = time.__len__()
    plt.plot(time[start:end], np.squeeze(data)[start:end], label=label)
    plt.xlabel(time.units)
    plt.ylabel(data.units)
    plt.legend()


class Evaluator:
    def __init__(self, eval_save_dir, mask_dir, test_dir, data_type):
        self.eval_save_dir = eval_save_dir
        self.mask_dir = mask_dir
        self.test_dir = test_dir
        self.data_type = data_type

    def infill(self, model, dataset, device, partitions):
        if not os.path.exists(self.eval_save_dir):
            os.makedirs('{:s}'.format(self.eval_save_dir))
        image = []
        mask = []
        gt = []
        output = []
        output_comp = []
        if partitions > dataset.__len__():
            partitions = dataset.__len__()
        if dataset.__len__() % partitions != 0:
            print("WARNING: The size of the dataset should be dividable by the number of partitions. The last "
                  + str(dataset.__len__() % partitions) + " time steps will not be infilled.")
        for split in range(partitions):
            image_part, mask_part, gt_part = zip(*[dataset[i + split * (dataset.__len__() // partitions)] for i in range(dataset.__len__() // partitions)])
            image_part = torch.stack(image_part)
            mask_part = torch.stack(mask_part)
            gt_part = torch.stack(gt_part)
            with torch.no_grad():
                output_part, _ = model(image_part.to(device), mask_part.to(device))
            output_part = output_part.to(torch.device('cpu'))
            output_comp_part = mask_part * image_part + (1 - mask_part) * output_part

            image.append(image_part)
            mask.append(mask_part)
            gt.append(gt_part)
            output.append(output_part)
            output_comp.append(output_comp_part)

        image = torch.cat(image)
        mask = torch.cat(mask)
        gt = torch.cat(gt)
        output = torch.cat(output)
        output_comp = torch.cat(output_comp)

        # get mid index
        mid_index = image.shape[1] // 2

        cvar = [image[:, mid_index, :, :], mask[:, mid_index, :, :], output[:, mid_index, :, :], output_comp[:, mid_index, :, :], gt[:, mid_index, :, :]]
        cname = ['image', 'mask', 'output', 'output_comp', 'gt']
        dname = ['time', 'lat', 'lon']
        for x in range(0, 5):
            h5 = h5py.File('%s' % (self.eval_save_dir + cname[x]), 'w')
            h5.create_dataset(self.data_type, data=cvar[x])
            for dim in range(0, 3):
                h5[self.data_type].dims[dim].label = dname[dim]
            h5.close()

        # convert to netCDF files
        self.convert_h5_to_netcdf(True, 'image')
        self.convert_h5_to_netcdf(False, 'gt')
        self.convert_h5_to_netcdf(False, 'output')
        self.convert_h5_to_netcdf(False, 'output_comp')

    def create_evaluation_images(self, file, create_video=False, start_date=None, end_date=None):
        if not os.path.exists(self.eval_save_dir + 'images'):
            os.makedirs('{:s}'.format(self.eval_save_dir + 'images'))
        file = self.eval_save_dir + file

        data = Dataset(file)
        time = data.variables['time']
        time = netCDF4.num2date(time[:], time.units)

        if start_date and end_date:
            start = parser.parse(start_date)
            end = parser.parse(end_date)
            pr = [data.variables[self.data_type][i, :, :] for i in range(time.__len__()) if
                  time[i] >= start and time[i] <= end]
            time = [time[i] for i in range(time.__len__()) if time[i] >= start and time[i] <= end]
        else:
            pr = data.variables[self.data_type][:, :, :]

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

    def create_evaluation_report(self, save_dir='evaluations/', create_evalutation_files=True, clean_data=True, infilled=True):
        if not os.path.exists(self.eval_save_dir + save_dir):
            os.makedirs('{:s}'.format(self.eval_save_dir + save_dir))
        directory = self.eval_save_dir + save_dir
        if create_evalutation_files:
            self.create_evaluation_files(clean_data, infilled)

        mse, _ = get_data(file=directory + 'mse.nc')
        mse = mse[0][0][0]

        timcor, _ = get_data(file=directory + 'timcor.nc')
        timcor = timcor[0][0][0]

        total_pr_gt, _ = get_data(file=directory + 'fldsum_gt.nc')
        total_pr_gt = total_pr_gt[0][0][0]

        total_pr_output_comp, _ = get_data(file=directory + 'fldsum_output_comp.nc')
        total_pr_output_comp = total_pr_output_comp[0][0][0]

        plt.title('Max values')
        plot_data(file=directory + 'gt_max.nc', start=0, label='Ground Truth')
        plot_data(file=sdirectory + 'output_comp_max.nc', start=0, label='Output')
        plt.savefig(directory + 'max.png')
        plt.clf()

        plt.title('Min values')
        plot_data(file=directory + 'gt_min.nc', start=0, label='Ground Truth')
        plot_data(file=directory + 'output_comp_min.nc', start=0, label='Output')
        plt.savefig(directory + 'min.png')

        plt.clf()
        plt.title('Mean values')
        plot_data(file=directory + 'gt_mean.nc', label='Ground Truth')
        plot_data(file=directory + 'output_comp_mean.nc', label='Output')
        plt.savefig(directory + 'mean.png')

        df = pd.DataFrame()
        df['Statistical Value'] = ["MSE", "Time Correlation", "Total Precipitation", "Other"]
        df['Ground Truth'] = ['%.5f' % mse, '%.5f' % timcor, total_pr_gt, 0]
        df['Output'] = ['%.5f' % mse, '%.5f' % timcor, total_pr_output_comp, 0]

        pdf = FPDF()
        pdf.add_page()
        pdf.set_xy(0, 0)
        pdf.set_font('arial', 'B', 12)
        pdf.cell(60)
        pdf.cell(75, 10, "Statistical evaluation of Ground Truth and Output Comp", 0, 2, 'C')
        pdf.cell(90, 10, " ", 0, 2, 'C')
        pdf.cell(-40)
        pdf.cell(50, 10, 'Statistical Value', 1, 0, 'C')
        pdf.cell(40, 10, 'Ground Truth', 1, 0, 'C')
        pdf.cell(40, 10, 'Output', 1, 2, 'C')
        pdf.cell(-90)
        pdf.set_font('arial', '', 12)
        for i in range(0, len(df)):
            pdf.cell(50, 10, '%s' % (df['Statistical Value'].iloc[i]), 1, 0, 'C')
            pdf.cell(40, 10, '%s' % (str(df['Ground Truth'].iloc[i])), 1, 0, 'C')
            pdf.cell(40, 10, '%s' % (str(df['Output'].iloc[i])), 1, 2, 'C')
            pdf.cell(-90)
        pdf.cell(90, 10, " ", 0, 2, 'C')
        pdf.cell(-30)
        pdf.image(directory + 'max.png', x=None, y=None, w=0, h=0, type='', link='')
        pdf.image(directory + 'min.png', x=None, y=None, w=0, h=0, type='', link='')
        pdf.image(directory + 'mean.png', x=None, y=None, w=0, h=0, type='', link='')
        pdf.output(directory + 'Report.pdf', 'F')

    def evaluate_selected_samples(self, dates=None):
        if dates is None:
            dates = ['2017-01-12T23', '2017-04-17T15', '2017-05-02T12', '2017-05-13T12', '2017-06-04T03',
                     '2017-06-29T16', '2017-07-12T14', '2017-09-02T13']
        i = 0
        for date in dates:
            os.system('cdo select,date=' + date + ' ' + self.eval_save_dir + 'image.nc ' + self.eval_save_dir + 'imagetmp' + str(i) + '.nc')
            os.system('cdo select,date=' + date + ' ' + self.eval_save_dir + 'output_comp.nc ' + self.eval_save_dir + 'output_comptmp' + str(i) + '.nc')
            os.system('cdo select,date=' + date + ' ' + self.eval_save_dir + 'gt.nc ' + self.eval_save_dir + 'gttmp' + str(i) + '.nc')
        os.system('mergetime ' + self.eval_save_dir + 'imagetmp* ' + self.eval_save_dir + 'image_selected.nc')
        os.system('mergetime ' + self.eval_save_dir + 'gttmp* ' + self.eval_save_dir + 'gt_selected.nc')
        os.system('mergetime ' + self.eval_save_dir + 'output_comptmp* ' + self.eval_save_dir + 'output_comp_selected.nc')
        os.system('rm ' + self.eval_save_dir + '*tmp*')

        self.create_evaluation_images(file='image_selected.nc')
        self.create_evaluation_images(file='gt_selected.nc')
        self.create_evaluation_images(file='output_comp_selected.nc')

    def create_evaluation_files(self, clean_data, infilled, save_dir):
        output_comp = 'output_comp.nc'
        gt = 'gt.nc'
        if clean_data:
            os.system('cdo gec,0.0 ' + self.eval_save_dir + 'output_comp.nc ' + self.eval_save_dir + 'tmp.nc')
            os.system('cdo mul ' + self.eval_save_dir + 'output_comp.nc ' + self.eval_save_dir + 'tmp.nc ' + self.eval_save_dir + 'output_comp_cleaned.nc')
            os.system('rm ' + self.eval_save_dir + 'tmp.nc')
            output_comp = 'output_comp_cleaned.nc'
        if infilled:
            os.system('cdo ifnotthen ' + self.mask_dir + ' ' + self.eval_save_dir + output_comp + ' ' + self.eval_save_dir + 'infilled_output_comp.nc')
            output_comp = 'infilled_output_comp.nc'
            os.system('cdo ifnotthen ' + self.mask_dir + ' ' + self.eval_save_dir + gt + ' ' + self.eval_save_dir + 'infilled_gt.nc')
            gt = 'infilled_gt.nc'

        # create correlation
        os.system('cdo timcor -hourmean -fldmean ' + self.eval_save_dir + output_comp + ' -hourmean -fldmean ' + self.eval_save_dir + gt + ' ' + save_dir + 'timcor.nc')
        # create sum in field
        os.system('cdo timcor -hourmean -fldsum ' + self.eval_save_dir + output_comp + ' -hourmean -fldsum ' + self.eval_save_dir + gt + ' ' + save_dir + 'fldsum_timcor.nc')
        # create mse
        os.system('cdo sqrt -timmean -sqr -hourlmean -fldmean ' + self.eval_save_dir + output_comp + ' -hourmean -fldmean ' + self.eval_save_dir + gt + ' ' + save_dir + 'mse.nc')
        # create total fldsum
        os.system('cdo fldsum -timsum ' + self.eval_save_dir + output_comp + ' ' + save_dir + 'fldsum_output_comp.nc')
        os.system('cdo fldsum -timsum ' + self.eval_save_dir + gt + ' ' + save_dir + 'fldsum_gt.nc')
        # create timeseries of time correlation
        os.system('cdo fldcor -setmisstoc,0 ' + self.eval_save_dir + output_comp + ' -setmisstoc,0 ' + self.eval_save_dir + gt + ' ' + save_dir + 'time_series.nc')
        # create min max mean time series
        os.system('cdo fldmax ' + self.eval_save_dir + output_comp + ' ' + save_dir + 'output_comp_max.nc')
        os.system('cdo fldmax ' + self.eval_save_dir + gt + ' ' + save_dir + 'gt_max.nc')
        os.system('cdo fldmin ' + self.eval_save_dir + output_comp + ' ' + save_dir + 'output_comp_min.nc')
        os.system('cdo fldmin ' + self.eval_save_dir + gt + ' ' + save_dir + 'gt_min.nc')
        os.system('cdo fldmean ' + self.eval_save_dir + output_comp + ' ' + save_dir + 'output_comp_mean.nc')
        os.system('cdo fldmean ' + self.eval_save_dir + gt + ' ' + save_dir + 'gt_mean.nc')

    def convert_h5_to_netcdf(self, create_structure_template, file):
        if create_structure_template:
            os.system('ncdump ' + self.test_dir + '*.h5 > ' + self.eval_save_dir + 'tmp_dump.txt')
            os.system('sed "/.*' + self.data_type + ' =.*/{s///;q;}" ' + self.eval_save_dir + 'tmp_dump.txt > ' + self.eval_save_dir + 'structure.txt')
            os.system('rm ' + self.eval_save_dir + 'tmp_dump.txt')
        os.system('cat ' + self.eval_save_dir + 'structure.txt >> ' + self.eval_save_dir + file + '.txt')
        os.system('ncdump -v ' + self.data_type + ' ' + self.eval_save_dir + file + ' | sed -e "1,/data:/d" >> ' + self.eval_save_dir + file + '.txt')
        os.system('ncgen -o ' + self.eval_save_dir + 'output-tmp ' + self.eval_save_dir + file + '.txt')
        os.system('cdo -setgrid,' + self.test_dir + '*.h5 ' + self.eval_save_dir + 'output-tmp ' + self.eval_save_dir + file + '.nc')
        os.system('rm ' + self.eval_save_dir + file + '.txt ' + self.eval_save_dir + 'output-tmp')