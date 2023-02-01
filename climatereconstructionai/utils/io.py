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


def read_input_file_as_dict(inp_file_path):
    input_dict = {}
    with open(inp_file_path,'r') as f:
        for line in f:
            line=line.replace('\t',' ')
            line_split = line.split(' ')
            if len(line_split)==1:
                input_dict[line_split[0].replace('\n','')] = True
            else:
                input_dict[line_split[0].replace('\n','')] = line_split[1].replace('\n','')
    return input_dict


def read_input_file_as_text(inp_file_path):
    input_file = ''

    with open(inp_file_path,'r') as f:
        for line in f:
            input_file+=line
    return input_file


def get_parameters_as_dict(input_dict, arg_parser):
    omit = ['help','load_from_file']
    parameter_dict = {}

    for action in vars(arg_parser)['_actions']:
        if action.dest not in omit:
            option_str = action.option_strings[-1]

            if option_str in input_dict.keys():
                if action.type==int:
                    value = int(input_dict[action.option_strings[-1]])
                elif action.type==float:
                    value = float(input_dict[action.option_strings[-1]])
      
                else:
                    if isinstance(input_dict[action.option_strings[-1]],str):
                        if str.isnumeric(input_dict[action.option_strings[-1]]):
                            value = int(input_dict[action.option_strings[-1]])
                        else:
                            value = input_dict[action.option_strings[-1]]
                    else:            
                        value = input_dict[action.option_strings[-1]]
            else:
                value = action.default
            parameter_dict[action.dest]=value
    return parameter_dict

def get_hparams(parameters_dict):
    hparams_dict = {}
    for key, value in parameters_dict.items():
        if not isinstance(value, str):
            hparams_dict[key] = value
    return hparams_dict


def write_parameters_as_inp(parameters_dict, file_path):

    with open(file_path, 'w') as f:
        for key, value in parameters_dict.items():
            if value is not None:
                if isinstance(value, bool):
                    if value:
                        input_line_str = '--' + key.replace('_','-') + '\n'
                        f.write(input_line_str)
                else:
                    input_line_str = '--' + key.replace('_','-') + ' ' + str(value) + '\n'
                    f.write(input_line_str)
                
        f.close()


