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

def generate_inputs(parameters, tuning, auto_dir='', run_id_start=0):

    if 'lambda_loss' in parameters and parameters['lambda_loss'] is not None:
        lambda_dict = key_value_list_to_dict(parameters['lambda_loss'].split(','))
    else:
        lambda_dict = {}
    
    run_id = run_id_start

    keys, values = zip(*tuning.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    print(f'Creating {len(permutations_dicts)}.inp files in {auto_dir}')

    run_files = []
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

        run_file_name = f'run_{run_id}.inp'
        auto_sub_dir = os.path.join(auto_dir, f'run_{run_id}')
        
        if not os.path.isdir(auto_sub_dir):
            os.makedirs(auto_sub_dir)
        
        parameters['snapshot_dir'] = auto_sub_dir

        #if mode is tuning
        parameters['writer_mode'] = 'snapshot_subdir'

        run_file_path = os.path.join(auto_sub_dir, run_file_name)
        io.write_parameters_as_inp(parameters, run_file_path)

        run_files.append(run_file_path)
        run_id+=1
    
    run_files_path = os.path.join(auto_dir, 'inp_files_tuning.txt')
    with open(run_files_path,'w') as f:
        [f.write(line + '\n')  for line in run_files]
        f.close()



if __name__ == "__main__":

    auto_dir = '/Users/maxwitte/work/crai_sr/auto'
    src_input_train_file = '/Users/maxwitte/work/crai_sr/inputs/sr_dyamond.inp'
    run_id_start = 0

    ap = cfg.set_train_args(src_input_train_file)
    input_dict = io.read_input_file_as_dict(src_input_train_file)
    parameters = io.get_parameters_as_dict(input_dict, arg_parser=ap)

    tuning = {
        'loss_criterion': [3],
        'lambda_style': [0,10,100],
        'lambda_prc': [0,0.05],
        'lambda_tv': [0,0.1],
        'conv_factor': [16]
    }

    generate_inputs(parameters, tuning, auto_dir=auto_dir, run_id_start=run_id_start)
    
