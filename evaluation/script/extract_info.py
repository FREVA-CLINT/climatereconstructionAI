import argparse
import re
import netCDF4
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
from fpdf import FPDF


def get_data(file):
    data = Dataset(file)
    time = data.variables['time']
    #time = netCDF4.num2date(time[:], time.units)
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


arg_parser = argparse.ArgumentParser()
# training options
arg_parser.add_argument('--data-root', type=str, default=None)
args = arg_parser.parse_args()

mse, _ = get_data(file=args.data_root + 'mse.nc')
mse = mse[0][0][0]

timcor, _ = get_data(file=args.data_root + 'timcor.nc')
timcor = timcor[0][0][0]

total_pr_gt, _ = get_data(file=args.data_root + 'fldsum_gt.nc')
total_pr_gt = total_pr_gt[0][0][0]

total_pr_output_comp, _ = get_data(file=args.data_root + 'fldsum_output_comp.nc')
total_pr_output_comp = total_pr_output_comp[0][0][0]

plt.title('Max values')
plot_data(file=args.data_root + 'gt_max.nc', start=0, label='Ground Truth')
plot_data(file=args.data_root + 'output_comp_max.nc', start=0, label='Output')
plt.savefig(args.data_root + 'max.png')
plt.clf()

plt.title('Min values')
plot_data(file=args.data_root + 'gt_min.nc', start=0, label='Ground Truth')
plot_data(file=args.data_root + 'output_comp_min.nc', start=0, label='Output')
plt.savefig(args.data_root + 'min.png')

plt.clf()
plt.title('Mean values')
plot_data(file=args.data_root + 'gt_mean.nc', label='Ground Truth')
plot_data(file=args.data_root + 'output_comp_mean.nc', label='Output')
plt.savefig(args.data_root + 'mean.png')


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
pdf.image(args.data_root + 'max.png', x = None, y = None, w = 0, h = 0, type = '', link = '')
pdf.image(args.data_root + 'min.png', x = None, y = None, w = 0, h = 0, type = '', link = '')
pdf.image(args.data_root + 'mean.png', x = None, y = None, w = 0, h = 0, type = '', link = '')
pdf.output(args.data_root + 'Report.pdf', 'F')