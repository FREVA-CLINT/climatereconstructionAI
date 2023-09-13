import os
import json
import argparse
from climatereconstructionai.model import EncDecLGTransformer


parser = argparse.ArgumentParser()
parser.add_argument("-f","--script_dict")

    
if __name__ == "__main__":
    
    args = parser.parse_args()
    script_dict = args.script_dict

    if isinstance(script_dict, str):
        with open(script_dict,'r') as file:
            script_dict = json.load(file)

    model = EncDecLGTransformer.SpatialTransNet(script_dict['model_settings'])
    model.train_(script_dict['training_settings']) 

