import os
import json
import argparse
from climatereconstructionai.model import core_model_crai


parser = argparse.ArgumentParser()
parser.add_argument("-f","--script_dict")

    
if __name__ == "__main__":
    
    args = parser.parse_args()
    script_dict = args.script_dict

    if isinstance(script_dict, str):
        with open(script_dict,'r') as file:
            script_dict = json.load(file)

    model = core_model_crai.CoreCRAI(script_dict['model_settings'])
    model.set_training_configuration(train_settings=script_dict['training_settings'])

    for task in script_dict['tasks']:
        if task=='sample_creation':
            model.create_samples(script_dict['sample_creation_settings'])
        
        elif task=='train_shell':
            model.train_(use_samples=True, subdir='shell')

        elif task=='train':
            model.train_(use_samples=True)