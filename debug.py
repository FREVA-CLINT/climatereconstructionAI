from climatereconstructionai.model import EncDecLGTransformer

from climatereconstructionai import config as cfg
import json

if __name__ == "__main__":

  model_settings = '/Users/maxwitte/work/crai_sr/src/climatereconstructionAI_remote/climatereconstructionAI/notebooks/transformer_settings/LGEncDec_Transformer.json'
  training_settings = '/Users/maxwitte/work/crai_sr/src/climatereconstructionAI_remote/climatereconstructionAI/notebooks/transformer_settings/training_settings.json'

  model = EncDecLGTransformer.SpatialTransNet(model_settings)
  model.train_(training_settings)