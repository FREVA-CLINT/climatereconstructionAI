import numpy as np
import h5py
import torch
import local_settings
from torchvision.utils import make_grid
from torchvision.utils import save_image

from util.image import unnormalize


def evaluate2(model, dataset, device, filename, partitions):
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

        #grid = make_grid(
        #        torch.cat((unnormalize(image_part), mask_part, unnormalize(output_part),
        #                   unnormalize(output_comp_part), unnormalize(gt_part)), dim=0))
        #save_image(grid, filename + str(split) + '.jpg')

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

    cvar = [image[:, 1, :, :], mask[:, 1, :, :], output[:, 1, :, :], output_comp[:, 1, :, :], gt[:, 1, :, :]]
    cname = ['image', 'mask', 'output', 'output_comp', 'gt']
    dname = ['time', 'lat', 'lon']
    for x in range(0, 5):
        h5 = h5py.File('h5/%s' % (cname[x]), 'w')
        h5.create_dataset(local_settings.data_type, data=cvar[x])
        for dim in range(0, 3):
            h5['pr'].dims[dim].label = dname[dim]
        h5.close()
