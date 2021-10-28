import argparse

import netCDF4
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
from dateutil import parser
import imageio

arg_parser = argparse.ArgumentParser()
# training options
arg_parser.add_argument('--file', type=str, default=None)
arg_parser.add_argument('--data-root', type=str, default=None)
arg_parser.add_argument('--mask', type=str, default=None)
arg_parser.add_argument('--start', type=str, default=None)
arg_parser.add_argument('--end', type=str, default=None)
arg_parser.add_argument('--video', type=str, default=False)
arg_parser.add_argument('--clean', type=str, default=False)
arg_parser.add_argument('--var', type=str, default=False)

args = arg_parser.parse_args()

file = args.data_root + args.file

#start_date_string = '2017-07-12-14:00'
#end_date_string = '2017-07-12-14:00'


data = Dataset(file)
lats = data.variables['lat'][:]
lons = data.variables['lon'][:]
time = data.variables['time']
time = netCDF4.num2date(time[:], time.units)

if args.start and args.end:
    start = parser.parse(args.start)
    end = parser.parse(args.end)
    pr = [data.variables[args.var][i, :, :] for i in range(time.__len__()) if time[i] >= start and time[i] <= end]
    time = [time[i] for i in range(time.__len__()) if time[i] >= start and time[i] <= end]
else:
    pr = data.variables[args.var][:, :, :]

units = data.variables[args.var].units

if args.mask:
    mask_data = Dataset(args.mask)
    mask = mask_data.variables[args.var][0, :, :]


for i in range(time.__len__()):
    if args.mask:
        for k in range(pr[i].__len__()):
            for l in range(pr[i][k].__len__()):
                if mask[k][l] == 0.0:
                    pr[i][k][l] = np.nan

    if args.clean:
        for k in range(pr[i].__len__()):
            for l in range(pr[i][k].__len__()):
                if pr[i][k][l] < 0.0:
                    pr[i][k][l] = 0.0

    preplot = plt.imshow(np.squeeze(pr[i]), vmin=0, vmax=5)
    plt.axis('off')
    plt.title('Precipitation from ' + str(time[i]))
    if 'gt' in args.file:
        name = 'gt'
    elif 'image' in args.file:
        name = 'image'
    elif 'output_comp' in args.file:
        name = 'output_comp'
    elif 'output' in args.file:
        name = 'output'
    else:
        name = 'sample'
    plt.savefig(args.data_root + 'images/' + name + '_' + str(i) + '.jpg')
    plt.clf()

if args.video:
    with imageio.get_writer(args.data_root + 'images/movie.gif', mode='I') as writer:
        for i in range(time.__len__()):
            image = imageio.imread('images/image_' + str(i) + '.jpg')
            writer.append_data(image)
