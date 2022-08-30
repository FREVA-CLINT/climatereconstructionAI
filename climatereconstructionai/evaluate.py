import os

from .model.net import PConvLSTM
from .utils.evaluation import infill, create_outputs
from .utils.io import load_ckpt, load_model
from .utils.netcdfloader import NetCDFLoader
from . import config as cfg


def evaluate(arg_file=None, prog_func=None):
    cfg.set_evaluate_args(arg_file, prog_func)

    if not os.path.exists(cfg.log_dir):
        os.makedirs(cfg.log_dir)

    n_models = len(cfg.model_names)
    assert n_models == len(cfg.eval_names)

    for i_model in range(n_models):

        if cfg.lstm_steps:
            recurrent = True
            time_steps = cfg.lstm_steps
        elif cfg.gru_steps:
            recurrent = True
            time_steps = cfg.gru_steps
        elif cfg.prev_next_steps:
            recurrent = False
            time_steps = cfg.prev_next_steps
        else:
            recurrent = False
            time_steps = 0

        dataset_val = NetCDFLoader(cfg.data_root_dir, cfg.img_names, cfg.mask_dir, cfg.mask_names, "infill",
                                   cfg.data_types, time_steps)

        if len(cfg.image_sizes) > 1:
            model = PConvLSTM(radar_img_size=cfg.image_sizes[0],
                              radar_enc_dec_layers=cfg.encoding_layers[0],
                              radar_pool_layers=cfg.pooling_layers[0],
                              radar_in_channels=2 * cfg.prev_next_steps + 1,
                              radar_out_channels=cfg.out_channels,
                              rea_img_size=cfg.image_sizes[1],
                              rea_enc_layers=cfg.encoding_layers[1],
                              rea_pool_layers=cfg.pooling_layers[1],
                              rea_in_channels=(len(cfg.image_sizes) - 1) * (2 * cfg.prev_next_steps + 1),
                              recurrent=recurrent).to(cfg.device)
        else:
            model = PConvLSTM(radar_img_size=cfg.image_sizes[0],
                              radar_enc_dec_layers=cfg.encoding_layers[0],
                              radar_pool_layers=cfg.pooling_layers[0],
                              radar_in_channels=2 * cfg.prev_next_steps + 1,
                              radar_out_channels=cfg.out_channels,
                              recurrent=recurrent).to(cfg.device)

        ckpt_dict = load_ckpt("{}/{}".format(cfg.model_dir, cfg.model_names[i_model]), cfg.device)
        output_name = "{}/{}".format(cfg.evaluation_dirs[0], cfg.eval_names[i_model])
        outputs = []
        for k in range(len(ckpt_dict["labels"])):
            load_model(ckpt_dict, model, label=ckpt_dict["labels"][k])
            model.eval()
            outputs.append(infill(model, dataset_val))
            if cfg.split_outputs:
                create_outputs(outputs, dataset_val, output_name, k)
                outputs = []
        if not cfg.split_outputs:
            create_outputs(outputs, dataset_val, output_name)


if __name__ == "__main__":
    evaluate()
