import torch
import numpy as np
import matplotlib.pyplot as plt
import os.path
from tensorboardX import SummaryWriter

from .. import config as cfg

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
    image = np.ma.masked_array(image, mask)

    mask = np.ma.masked_array(mask, mask)

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


def infill(model, dataset, partitions, eval_path):
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

        if split == 0 and cfg.create_graph:
            writer = SummaryWriter(log_dir=cfg.log_dir)
            writer.add_graph(model, (image_part,mask_part,rea_images_part,rea_masks_part))
            writer.close()

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
    write_outputs(cvar, dataset.img_data[0], eval_path)

    return np.ma.masked_array(gt, mask)[:, 0, :, :], np.ma.masked_array(output_comp[:, 0, :, :], mask[:, 0, :, :])

def write_outputs(cvar, img_data, eval_path):

    for cname in cvar:
        output_name = '{}_{}'.format(eval_path,cname)
        data = cvar[cname].to(torch.device('cpu')).detach().numpy()
        data = data[:,0,:,:]

        ds = img_data.copy(data={cfg.data_types[0]: data})
        ds.to_netcdf(output_name+".nc")
