import torch
import torch.nn as nn


def get_state_dict_on_cpu(obj):
    cpu_device = torch.device('cpu')
    state_dict = obj.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].to(cpu_device)
    return state_dict


def save_ckpt(ckpt_name, train_stats, savelist, model_settings={}):
    ckpt_dict = {'labels': [], 'train_stats': train_stats, 'model_settings':model_settings}
    for label, iter, model, optimizer in savelist:
        if model is not None:
            ckpt_dict["labels"].append(label)
            ckpt_dict[label] = {"n_iter": iter, "model": get_state_dict_on_cpu(model),
                                "optimizer": optimizer.state_dict()}
     
    if ckpt_dict["labels"]:
        torch.save(ckpt_dict, ckpt_name)


def load_ckpt(ckpt_name, device):
    ckpt_dict = torch.load(ckpt_name, map_location=device)
    keys = ckpt_dict.keys()
    if "labels" not in keys:
        label = str(ckpt_dict["n_iter"])
        ckpt_dict["labels"] = [label]
        ckpt_dict[label] = {key: ckpt_dict[key] for key in keys}

    return ckpt_dict


def load_model(ckpt_dict, model, optimizer=None, label=None, strict=False, match_list=None, not_match=None):
    assert isinstance(model, nn.Module)
    if label is None:
        label = ckpt_dict["labels"][-1]

    ckpt_dict[label]["model"] = \
        {key.replace("module.", ""): value for key, value in ckpt_dict[label]["model"].items()}
    
    if match_list is not None:
        ckpt_dict_match = {}
        for key, value in ckpt_dict[label]["model"].items():
            if ((not_match is not None) and (not_match not in key)) or match_list in key:
                ckpt_dict_match[key]=value

    elif not_match is not None:
        ckpt_dict_match = {}
        for key, value in ckpt_dict[label]["model"].items():
            if (not_match not in key):
                ckpt_dict_match[key]=value
    else:
        ckpt_dict_match = ckpt_dict[label]["model"]

    model.load_state_dict(ckpt_dict_match, strict=strict)

    if optimizer is not None:
        optimizer.load_state_dict(ckpt_dict[label]["optimizer"])
    return ckpt_dict[label]["n_iter"]
