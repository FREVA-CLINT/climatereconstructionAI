import torch
import torch.nn as nn


def get_state_dict_on_cpu(obj):
    cpu_device = torch.device('cpu')
    state_dict = obj.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].to(cpu_device)
    return state_dict


def save_ckpt(ckpt_name, savelist):
    ckpt_dict = {'iters': []}
    for i, model, optimizer in savelist:
        s_iter = str(i)
        ckpt_dict["iters"].append(s_iter)
        ckpt_dict[s_iter] = {"model": get_state_dict_on_cpu(model), "optimizer": optimizer.state_dict()}
    torch.save(ckpt_dict, ckpt_name)


def load_ckpt(ckpt_name, device):
    return torch.load(ckpt_name, map_location=device)


def load_model(ckpt_dict, model, optimizer=None, s_iter=None):
    assert isinstance(model, nn.Module)
    if s_iter is None:
        s_iter = ckpt_dict["iters"][-1]

    ckpt_dict[s_iter]["model"] = \
        {key.replace("module.", ""): value for key, value in ckpt_dict[s_iter]["model"].items()}
    model.load_state_dict(ckpt_dict[s_iter]["model"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt_dict[s_iter]["optimizer"])
    return int(s_iter)
