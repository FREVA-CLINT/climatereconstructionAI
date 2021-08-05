import torch
from torchvision.utils import make_grid
from torchvision.utils import save_image

from util.image import unnormalize


def evaluate(model, dataset, device, filename):
    image, mask, gt = zip(*[dataset[i] for i in range(8)])
    image = torch.stack(image)
    mask = torch.stack(mask)
    gt = torch.stack(gt)
    with torch.no_grad():
        output, _ = model(image.to(device), mask.to(device))
    output = output.to(torch.device('cpu'))
    output_comp = mask * image + (1 - mask) * output

    output_comp = output_comp[:][0][:][:]
    output_comp = output_comp.unsqueeze(1)
    output = output[:][0][:][:]
    output = output.unsqueeze(1)
    gt = gt[:][0][:][:]
    gt = gt.unsqueeze(1)
    image = image[:][0][:][:]
    image = image.unsqueeze(1)

    grid = make_grid(
        torch.cat((image, mask, output,
                   output_comp, gt), dim=0))
    save_image(grid, filename)
