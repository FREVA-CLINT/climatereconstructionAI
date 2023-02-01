from climatereconstructionai import train, evaluate
from climatereconstructionai.utils import io
from climatereconstructionai import config as cfg
import itertools
import os


def key_value_list_to_dict(key_value_list):
    keys = [arg for arg in key_value_list if not str.isnumeric(arg)]
    values = [float(arg) for arg in key_value_list if str.isnumeric(arg)]
    return dict(zip(keys, values))

def dict_to_key_value_list(key_value_dict):
    key_value_list = []
    for key,value in key_value_dict.items():
        key_value_list.append(key)
        key_value_list.append(value)
    return key_value_list

def iterate_tuning(parameters, tuning):

    if 'lambda_loss' in parameters:
        lambda_dict = key_value_list_to_dict(parameters['lambda_loss'].split(','))
    else:
        lambda_dict = {}
    
    run_id = 0

    keys, values = zip(*tuning.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    print(f'starting tuning for {len(permutations_dicts)} permutations')

    for permutation in permutations_dicts:
        for key, value in permutation.items():
            if 'lambda' in key:
                lambda_dict[key.replace('lambda_','')]=value
            else:
                parameters[key]=value
    
        if lambda_dict:
            lambda_key_value_list = dict_to_key_value_list(lambda_dict)
            lambda_loss_str = ''
            for val in lambda_key_value_list:
                lambda_loss_str+=str(val) + ','
            parameters['lambda_loss'] = lambda_loss_str[:-1]

        tmp_file_name = f'temp{run_id}.inp'

        io.write_parameters_as_inp(parameters,tmp_file_name)
        train(tmp_file_name)
        print(f'iteration {run_id} of {len(permutations_dicts)} done')
        run_id+=1



if __name__ == "__main__":
    src_input_file = '/Users/maxwitte/work/crai_sr/inputs/sr_dyamond.inp'
    ap = cfg.set_train_args(src_input_file)
 
    input_dict = io.read_input_file_as_dict(src_input_file)
    parameters = io.get_parameters_as_dict(input_dict, arg_parser=ap)

    tuning = {
        'encoding_layers': [4],
        'pooling_layers': [4],
        'conv_factor': [64],
        'lambda_style': [0.]
    }

    iterate_tuning(parameters, tuning)
    
