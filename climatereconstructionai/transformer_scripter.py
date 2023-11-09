import os
import json
import argparse
from climatereconstructionai.model import core_model_crai, pyramid_step_model


parser = argparse.ArgumentParser()
parser.add_argument("-f","--script_dict")

    
if __name__ == "__main__":
    
    args = parser.parse_args()
    script_dict = args.script_dict

    if isinstance(script_dict, str):
        with open(script_dict,'r') as file:
            script_dict = json.load(file)

    
    model_init = False

    for task_dict in script_dict['tasks']:
        task = task_dict['task']

        model_settings = script_dict['model_settings'] if "model_settings" not in task_dict.keys() else task_dict['model_settings']
        train_settings = script_dict['training_settings'] if "training_settings" not in task_dict.keys() else task_dict['training_settings']

        if task=='train_shell':
            model = pyramid_step_model.pyramid_step_model(model_settings)
        else:
            if not model_init:
                model = core_model_crai.CoreCRAI(model_settings)    
                model_init = True

        model.set_training_configuration(train_settings=train_settings)
        
        if task=='train_shell':
            model.train_(subdir='shell')

        elif task=='train_samples':
            model.train_()

        elif task=='train':
            model.train_()

        elif task=='train_with_pretrained_shell':
            model.train_(pretrain_subdir='shell')