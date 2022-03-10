import sys
import numpy as np
from numpy import ma
sys.path.append('./')
import config as cfg


def smooth(x, time):
    unique, bin_indices = np.unique([t.month for t in time], return_counts=True)
    for i in range(1, len(bin_indices)):
        bin_indices[i] += bin_indices[i-1]
    bins = np.split(x, bin_indices, axis=0)
    ts = []
    for bin in bins:
        if np.mean(bin):
            ts.append(np.mean(bin))
    return np.asarray(ts)


def rmse_over_mean(gt, output):
    return np.sqrt(np.mean((np.mean(gt, axis=(1, 2)) - np.mean(output, axis=(1, 2))) ** 2))


def rmse(gt, output):
    return np.sqrt(np.mean((gt - output) ** 2))


def timcor(gt, output):
    return np.corrcoef(np.mean(gt, axis=(1, 2)), np.mean(output, axis=(1, 2)))[0][1]


def max_timeseries(input, time):
    return smooth(np.max(input, axis=(1, 2)), time)


def min_timeseries(input, time):
    return smooth(np.min(input, axis=(1, 2)), time)


def mean_timeseries(input, time):
    return smooth(np.mean(input, axis=(1, 2)), time)


def total_sum(input):
    return np.sum(input)


def fldcor_timeseries(gt, output, time=None):
    time_series = np.zeros(gt.shape[0])
    mask = np.zeros(gt.shape[0])
    for i in range(gt.shape[0]):
        gt_flat = gt[i].flatten().compressed()
        output_flat = output[i].flatten().compressed()
        if cfg.eval_threshold or np.max(gt_flat) == np.min(gt_flat) or np.max(output_flat) == np.min(output_flat):
            mask[i] = 1
        else:
            time_series[i] = np.corrcoef(gt_flat, output_flat)[0][1]
    if time.any():
        return smooth(ma.masked_array(np.array(time_series), mask), time)
    else:
        return ma.masked_array(np.array(time_series), mask)


def fldor_timsum(gt, output):
    return np.corrcoef(np.sum(gt, axis=0).flatten().compressed(), np.sum(output, axis=0).flatten().compressed())[0][1]


def timmean_fldor(gt, output):
    return np.mean(fldcor_timeseries(gt, output))


def timcor_map(gt, output):
    map = np.zeros((1, gt.shape[1], gt.shape[2]))
    for i in range(gt.shape[1]):
        for j in range(gt.shape[2]):
            map[0, i, j] = np.corrcoef(gt[:,i,j], output[:,i,j])[0][1]
    return ma.masked_array(map, ma.getmask(gt[0,:,:]))


def sum_map(image):
    return np.expand_dims(np.sum(image, axis=0), 0)


def rmse_map(gt, output):
    return np.expand_dims(np.mean(np.sqrt((gt - output) ** 2), axis=0), 0)


def rmse_timeseries(gt, output, time):
    return smooth(np.mean(np.sqrt((gt - output) ** 2), axis=(1, 2)), time)


def rmse_over_mean_timeseries(gt, output, time):
    return smooth(np.abs((np.mean(gt,  axis=(1, 2)) - np.mean(output,  axis=(1, 2)))), time)