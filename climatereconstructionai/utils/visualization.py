import matplotlib.pyplot as plt
import numpy as np
import torch


def calculate_distributions(mask, steady_mask, output, gt, domain="valid", num_samples=1000):
    if steady_mask is not None:
        mask += steady_mask
        mask[mask < 0] = 0
        mask[mask > 1] = 1
        assert ((mask == 0) | (mask == 1)).all(), "Not all values in mask are zeros or ones!"

    value_list_pred = []
    value_list_target = []
    for ch in range(gt.shape[2]):
        mask_ch = mask[:, :, ch, :, :]
        gt_ch = gt[:, :, ch, :, :]
        output_ch = output[:, :, ch, :, :]

        if domain == 'valid':
            pred = output_ch[mask_ch == 1]
            target = gt_ch[mask_ch == 1]
        elif domain == 'hole':
            pred = output_ch[mask == 0]
            target = gt_ch[mask == 0]
        elif domain == 'comp_infill':
            pred = mask_ch * gt_ch + (1 - mask_ch) * output_ch
            target = gt_ch
        pred = pred.flatten()
        target = target.flatten()

        sample_indices = torch.randint(len(pred), (num_samples,))
        value_list_pred.append(pred[sample_indices])
        value_list_target.append(target[sample_indices])

    return value_list_pred, value_list_target


def calculate_error_distributions(mask, steady_mask, output, gt, operation="AE", domain="valid", num_samples=1000):
    preds, targets = calculate_distributions(mask, steady_mask, output, gt, domain=domain, num_samples=num_samples)

    if steady_mask is not None:
        mask += steady_mask
        mask[mask < 0] = 0
        mask[mask > 1] = 1
        assert ((mask == 0) | (mask == 1)).all(), "Not all values in mask are zeros or ones!"

    value_list = []
    for ch in range(len(preds)):
        pred = preds[ch]
        target = targets[ch]

        if operation == "AE":
            values = torch.sqrt((pred - target) ** 2)
        elif operation == "E":
            values = pred - target
        elif operation == "RAE":
            values = (pred - target).abs() / (target + 1e-9)
        elif operation == "RE":
            values = (pred - target) / (target + 1e-9)

        values = values.flatten()
        sample_indices = torch.randint(len(values), (num_samples,))
        value_list.append(values[sample_indices])
    return value_list


def create_error_dist_plot(mask, steady_mask, output, gt, operation='E', domain="valid", num_samples=1000):
    preds, targets = calculate_distributions(mask, steady_mask, output, gt, domain=domain, num_samples=num_samples)

    fig, axs = plt.subplots(1, len(preds), squeeze=False)

    for ch in range(len(preds)):

        pred = preds[ch].cpu()
        target = targets[ch].cpu()

        if operation == "AE":
            errors_ch = np.sqrt((pred - target) ** 2)
        elif operation == "E":
            errors_ch = pred - target
        elif operation == "RAE":
            errors_ch = (pred - target).abs() / (target + 1e-9)
        elif operation == "RE":
            errors_ch = (pred - target) / (target + 1e-9)

        m = (errors_ch).mean()
        s = (errors_ch).std()
        xlims = [target.min() - 0.5 * (target).diff().abs().mean(), target.max() + 0.5 * (target).diff().abs().mean()]

        axs[0, ch].hlines([m, m - s, m + s], xlims[0], xlims[1], colors=['grey', 'red', 'red'], linestyles='dashed')
        axs[0, ch].scatter(target, errors_ch, color='black')
        axs[0, ch].grid()
        axs[0, ch].set_xlabel('target values')
        axs[0, ch].set_ylabel('errors')
        axs[0, ch].set_xlim(xlims)
    return fig


def create_correlation_plot(mask, steady_mask, output, gt, domain="valid", num_samples=1000):
    preds, targets = calculate_distributions(mask, steady_mask, output, gt, domain=domain, num_samples=num_samples)

    fig, axs = plt.subplots(1, len(preds), squeeze=False)

    for ch in range(len(preds)):
        target_data = targets[ch]
        pred_data = preds[ch]
        R = torch.corrcoef(torch.vstack((target_data, pred_data)))[0, 1]

        axs[0, ch].scatter(target_data.cpu(), pred_data.cpu(), color='red', alpha=0.5)
        axs[0, ch].plot(target_data.cpu(), target_data.cpu(), color='black')
        axs[0, ch].grid()
        axs[0, ch].set_xlabel('target values')
        axs[0, ch].set_ylabel('predicted values')
        axs[0, ch].set_title(f'R = {R:.4}')

    return fig


def create_error_map(mask, steady_mask, output, gt, num_samples=3, operation="AE", domain="valid"):
    if steady_mask is not None:
        mask += steady_mask
        mask[mask < 0] = 0
        mask[mask > 1] = 1
        assert ((mask == 0) | (mask == 1)).all(), "Not all values in mask are zeros or ones!"

    num_channels = gt.shape[2]
    samples = torch.randint(gt.shape[0], (num_samples,))

    fig, axs = plt.subplots(num_channels, num_samples, squeeze=False, figsize=(num_samples * 7, num_channels * 7))

    for ch in range(gt.shape[2]):
        gt_ch = gt[:, :, ch, :, :]
        output_ch = output[:, :, ch, :, :]

        for sample_num in range(num_samples):

            target = np.squeeze(gt_ch[samples[sample_num]].squeeze())
            pred = np.squeeze(output_ch[samples[sample_num]].squeeze())

            if operation == "AE":
                values = (pred - target).abs()
                cm = 'cividis'
            elif operation == "E":
                values = pred - target
                cm = 'coolwarm'
            elif operation == "RAE":
                values = (pred - target).abs() / (target.abs() + 1e-9)
                cm = 'cividis'
            elif operation == "RE":
                values = (pred - target) / (target + 1e-9)
                cm = 'coolwarm'

            vmin, vmax = torch.quantile(values, torch.tensor([0.05, 0.95], device=values.device))
            cp = axs[ch, sample_num].matshow(values.cpu(), cmap=cm, vmin=vmin, vmax=vmax)
            axs[ch, sample_num].set_xticks([])
            axs[ch, sample_num].set_yticks([])
            axs[ch, sample_num].set_title(f'sample {sample_num}')
            plt.colorbar(cp, ax=axs[ch, sample_num])

    return fig


def create_map(mask, steady_mask, output, gt, input, num_samples=3):
    if steady_mask is not None:
        mask += steady_mask
        mask[mask < 0] = 0
        mask[mask > 1] = 1
        assert ((mask == 0) | (mask == 1)).all(), "Not all values in mask are zeros or ones!"

    samples = torch.randint(output.shape[0], (num_samples,))

    fig, axs = plt.subplots(3, num_samples, squeeze=False, figsize=(num_samples * 7, 14))

    gt_ch = gt[:, :, 0, :, :]
    output_ch = output[:, :, 0, :, :]
    input_ch = input[:, :, 0, :, :]

    for sample_num in range(num_samples):
        target = gt_ch[samples[sample_num]].squeeze()
        pred = output_ch[samples[sample_num]].squeeze()
        input = input_ch[samples[sample_num]].squeeze()

        vmin, vmax = torch.quantile(target, torch.tensor([0.05, 0.95], device=target.device))

        cp1 = axs[0, sample_num].matshow(target.cpu(), cmap='viridis', vmin=vmin, vmax=vmax)
        cp2 = axs[1, sample_num].matshow(pred.cpu(), cmap='viridis', vmin=vmin, vmax=vmax)
        cp3 = axs[2, sample_num].matshow(input.cpu(), cmap='viridis', vmin=vmin, vmax=vmax)

        plt.colorbar(cp1, ax=axs[0, sample_num])
        plt.colorbar(cp2, ax=axs[1, sample_num])
        plt.colorbar(cp3, ax=axs[2, sample_num])

        axs[0, sample_num].set_xticks([])
        axs[0, sample_num].set_yticks([])
        axs[1, sample_num].set_xticks([])
        axs[1, sample_num].set_yticks([])
        axs[2, sample_num].set_xticks([])
        axs[2, sample_num].set_yticks([])
        axs[0, sample_num].set_title(f'gt - sample {samples[sample_num]}')
        axs[1, sample_num].set_title(f'output - sample {samples[sample_num]}')
        axs[2, sample_num].set_title(f'input - sample {samples[sample_num]}')
    return fig


def get_all_error_distributions(mask, steady_mask, output, gt, domain="valid", num_samples=1000):
    error_dists = [calculate_error_distributions(mask, steady_mask, output, gt, operation=op, domain=domain,
                                                 num_samples=num_samples) for op in ['E', 'AE', 'RE', 'RAE']]
    return error_dists


def get_all_error_maps(mask, steady_mask, output, gt, num_samples=3):
    error_maps = [create_error_map(mask, steady_mask, output, gt, num_samples=num_samples, operation=op, domain="valid")
                  for op in ['E', 'AE', 'RE', 'RAE']]
    return error_maps



def create_scatter_plots(data_output, data_target, coords, var_spatial_dim_dict_target):

    b,n,c,g = list(data_output.values())[0].shape

    fig, axs = plt.subplots(len(data_output.keys()), 2+(g-1), squeeze=False, figsize=(5*len(data_output.keys()), 10*(2+(g-1))))

    sample_idx = 0#np.random.randint(0,b,(1))

    var_idx = 0

    for var in data_output.keys():
        coords_sd = coords[var_spatial_dim_dict_target[var]].cpu()
        

        p = axs[var_idx, 0].scatter(coords_sd[sample_idx,0], coords_sd[sample_idx,1], c=data_target[var][sample_idx,:,0])
        axs[var_idx, 0].set_title(f'{var} - target')
        plt.colorbar(p,ax=axs[var_idx, 0])

        p =axs[var_idx, 1].scatter(coords_sd[sample_idx,0], coords_sd[sample_idx,1], c=data_output[var][sample_idx,:,0,0])
        axs[var_idx, 1].set_title(f'{var} - output')
        plt.colorbar(p,ax=axs[var_idx, 1])

        if g>1:
            p = axs[var_idx, 2].scatter(coords_sd[sample_idx,0], coords_sd[sample_idx,1], c=data_output[var][sample_idx,:,0,1])
            axs[var_idx, 2].set_title(f'{var} - output std')
            plt.colorbar(p,ax=axs[var_idx, 2])

        var_idx+=1

