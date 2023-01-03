import torch
import torch.nn as nn


def get_state_dict_on_cpu(obj):
    cpu_device = torch.device('cpu')
    state_dict = obj.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].to(cpu_device)
    return state_dict


def save_ckpt(ckpt_name, stat_target, savelist):
    ckpt_dict = {'labels': [], 'stat_target': stat_target}
    for label, iter, model, optimizer in savelist:
        ckpt_dict["labels"].append(label)
        ckpt_dict[label] = {"n_iter": iter, "model": get_state_dict_on_cpu(model),
                            "optimizer": optimizer.state_dict()}


    torch.save(ckpt_dict, ckpt_name)


def load_ckpt(ckpt_name, device):
    ckpt_dict = torch.load(ckpt_name, map_location=device)
    keys = ckpt_dict.keys()
    if "labels" not in keys:
        label = str(ckpt_dict["n_iter"])
        ckpt_dict["labels"] = [label]
        ckpt_dict[label] = {key: ckpt_dict[key] for key in keys}

    return ckpt_dict


def load_model(ckpt_dict, model, optimizer=None, label=None):
    assert isinstance(model, nn.Module)
    if label is None:
        label = ckpt_dict["labels"][-1]

    ckpt_dict[label]["model"] = \
        {key.replace("module.", ""): value for key, value in ckpt_dict[label]["model"].items()}
    model.load_state_dict(ckpt_dict[label]["model"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt_dict[label]["optimizer"])
    return ckpt_dict[label]["n_iter"]
