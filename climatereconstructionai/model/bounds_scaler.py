import torch
import numpy as np


def bnd_pass(output):
    return output


def bnd_sigmoid(output, ymin, ymax):
    output = torch.sigmoid(output)
    return output * (ymax - ymin) + ymin


def bnd_exp(output, y0, sign=1):
    output = sign * torch.exp(sign * output)
    return output + y0


class constrain_bounds():
    def __init__(self, bounds):

        self.params = []
        self.binders = []
        self.n_bounds = len(bounds)
        for i in range(self.n_bounds):
            n_inf = np.isinf(bounds[i]).sum()
            if n_inf == 2:
                self.params.append([])
                self.binders.append(bnd_pass)
            elif n_inf == 0:
                self.params.append(bounds[i])
                self.binders.append(bnd_sigmoid)
            else:
                self.binders.append(bnd_exp)
                if bounds[i, 0] == np.inf:
                    self.params.append([bounds[i, 1], -1])
                else:
                    self.params.append([bounds[i, 0]])

    def scale(self, output):
        for i in range(self.n_bounds):
            output[:, :, i, :, :] = self.binders[i](output[:, :, i, :, :], *self.params[i])

        return output
