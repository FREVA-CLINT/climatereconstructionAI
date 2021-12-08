import numpy as np


def rmse(gt, output):
    return np.sqrt(np.mean((np.mean(gt, axis=(1, 2)) - np.mean(output, axis=(1, 2))) ** 2))


def timcor(gt, output):
    return np.corrcoef(np.mean(gt, axis=(1, 2)), np.mean(output, axis=(1, 2)))[0][1]


def max_timeseries(input):
    return np.max(input, axis=(1, 2))


def min_timeseries(input):
    return np.min(input, axis=(1, 2))


def mean_timeseries(input):
    sums = []
    for i in range(input.shape[0]):
        sum = np.sum(input[i])
        sums.append(sum / (input[i].shape[0] * input[i].shape[1]))

    return sums


def total_sum(input):
    return np.sum(input)


def fldcor_timeseries(gt, output):
    time_series = []
    for i in range(gt.shape[0]):
        flat_gt = gt[i].flatten()
        output_flat = output[i].flatten()
        time_series.append(np.corrcoef(flat_gt, output_flat)[0][1])
    return np.array(time_series)


def fldor_timsum(gt, output):
    return np.corrcoef(np.sum(gt, axis=0).flatten(), np.sum(output, axis=0).flatten())[0][1]


def timmean_fldor(gt, output):
    return np.mean(fldcor_timeseries(gt, output))
