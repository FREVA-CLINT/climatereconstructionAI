import numpy as np
from numpy import ma


def rmse(gt, output):
    return np.sqrt(np.mean((np.mean(gt, axis=(1, 2)) - np.mean(output, axis=(1, 2))) ** 2))


def timcor(gt, output):
    return np.corrcoef(np.mean(gt, axis=(1, 2)), np.mean(output, axis=(1, 2)))[0][1]


def max_timeseries(input):
    return np.max(input, axis=(1, 2))


def min_timeseries(input):
    return np.min(input, axis=(1, 2))


def mean_timeseries(input):
    return np.mean(input, axis=(1, 2))


def total_sum(input):
    return np.sum(input)


def fldcor_timeseries(gt, output):
    time_series = np.zeros(gt.shape[0])
    mask = np.zeros(gt.shape[0])
    for i in range(gt.shape[0]):
        gt_flat = gt[i].flatten().compressed()
        output_flat = output[i].flatten().compressed()
        if gt_flat == np.array([]) or output_flat == np.array([]) or np.max(gt_flat) == np.min(gt_flat) or np.max(output_flat) == np.min(output_flat):
            mask[i] = 1
        else:
            time_series[i] = np.corrcoef(gt_flat, output_flat)[0][1]
    return ma.masked_array(np.array(time_series), mask)


def fldor_timsum(gt, output):
    return np.corrcoef(np.sum(gt, axis=0).flatten().compressed(), np.sum(output, axis=0).flatten().compressed())[0][1]


def timmean_fldor(gt, output):
    return np.mean(fldcor_timeseries(gt, output))
