from climatereconstructionai.model import EncDecLGTransformer,trans_u_net

from climatereconstructionai import config as cfg
import json

if __name__ == "__main__":

  model_settings = '/Users/maxwitte/work/crai_sr/src/climatereconstructionAI_remote/climatereconstructionAI/notebooks/transformer_settings/LGEncDec_Transformer.json'
  training_settings = '/Users/maxwitte/work/crai_sr/src/climatereconstructionAI_remote/climatereconstructionAI/notebooks/transformer_settings/training_settings.json'

  model = EncDecLGTransformer.SpatialTransNet(model_settings)
  model.train_(training_settings) 


  #model_settings = '/Users/maxwitte/work/crai_sr/src/climatereconstructionAI_remote/climatereconstructionAI/notebooks/transformer_settings/LGEncDec_Transformer_infer.json'
  #settings = '/Users/maxwitte/work/crai_sr/src/climatereconstructionAI_remote/climatereconstructionAI/notebooks/transformer_settings/inference_settings.json'

  #model = EncDecLGTransformer.SpatialTransNet(model_settings)
  #model.infer(settings) 


  #model_settings = '/Users/maxwitte/work/crai_sr/src/climatereconstructionAI_remote/climatereconstructionAI/notebooks/transformer_settings/TransUNet.json'
  #settings = '/Users/maxwitte/work/crai_sr/src/climatereconstructionAI_remote/climatereconstructionAI/notebooks/transformer_settings/training_settings_trans_unet.json'

  #model = trans_u_net.TransUNet(model_settings)
  #model.train(settings) 