import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-f","--script_dict")
parser.add_argument("-m","--model_dir", required=False, default=None)
    
if __name__ == "__main__":
    
    args = parser.parse_args()
    script_dict = args.script_dict
    model_dir = args.model_dir

    if isinstance(script_dict, str):
        with open(script_dict,'r') as file:
            script_dict = json.load(file)

    
    model_init = False

    for task_dict in script_dict['tasks']:
        task = task_dict['task']

        model_settings = script_dict['model_settings'] if "model_settings" not in task_dict.keys() else task_dict['model_settings']
        train_settings = script_dict['training_settings'] if "training_settings" not in task_dict.keys() else task_dict['training_settings']

        if task=='train_shell':
            from climatereconstructionai.model import pyramid_step_model
            model = pyramid_step_model.pyramid_step_model(model_settings, model_dir=model_dir)
        elif task =='init_global':
            from climatereconstructionai.model import pyramid_model
            model = pyramid_model.pyramid_model(model_settings, model_dir=model_dir)
        else:
            if not model_init:
                from climatereconstructionai.model import pyramid_step_model
                model_settings = pyramid_step_model.load_settings(model_settings)
                if 'model' not in model_settings.keys():
                    model_type = 'crai'
                else:
                    model_type = model_settings['model']    
                if model_type=='crai':
                    from climatereconstructionai.model import core_model_crai
                    model = core_model_crai.CoreCRAI(model_settings, model_dir=model_dir)
                elif model_type=='shuffle':
                    from climatereconstructionai.model import core_model_resushuffle
                    model = core_model_resushuffle.core_ResUNet(model_settings, model_dir=model_dir)
                elif model_type=='shuffle_vae':
                    from climatereconstructionai.model import core_model_resushuffle_vae
                    model = core_model_resushuffle_vae.core_ResVAE(model_settings, model_dir=model_dir)
                elif model_type=='icon_transformer':
                    from climatereconstructionai.model import ICONTransformer
                    model = ICONTransformer.ICON_Transformer(model_settings)
                elif model_type=='icon_transformer_decmop':
                    from climatereconstructionai.model import ICONTransformer_decomp
                    model = ICONTransformer_decomp.ICON_Transformer(model_settings)
                elif model_type=='icon_transformer_proj':
                    from climatereconstructionai.model import ICONTransformer_proj
                    model = ICONTransformer_proj.ICON_Transformer(model_settings)
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