import numpy as np
import h5py
import torch
from torchvision.utils import make_grid
from torchvision.utils import save_image

from util.image import unnormalize


def evaluate2(model, dataset, device, filename):
    image, mask, gt = zip(*[dataset[i] for i in range(2028)])
    image = torch.stack(image)
    mask = torch.stack(mask)
    gt = torch.stack(gt)
    with torch.no_grad():
        output, _ = model(image.to(device), mask.to(device))
    output = output.to(torch.device('cpu'))
    output_comp = mask * image + (1 - mask) * output

    #grid = make_grid(
    #    torch.cat((unnormalize(image), mask, unnormalize(output),
    #               unnormalize(output_comp), unnormalize(gt)), dim=0))
    #save_image(grid, filename)
    
    cvar = [ image[:,1,:,:] , mask[:,1,:,:], output[:,1,:,:], output_comp[:,1,:,:], gt[:,1,:,:] ]
    cname = [ 'image', 'mask', 'output', 'output_comp', 'gt' ]
    dname = [ 'time', 'lat', 'lon' ]
    for x in range(0, 5):
        h5 = h5py.File('h5/%s' % (cname[x]), 'w')
        h5.create_dataset('tas', data=cvar[x])
        for dim in range(0, 3):
            h5['tas'].dims[dim].label = dname[dim]
        h5.close()
