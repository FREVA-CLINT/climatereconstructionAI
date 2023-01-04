import os

from .model.net import CRAINet
from .utils.evaluation import infill, create_outputs
from .utils.io import load_ckpt, load_model
from .utils.netcdfloader import NetCDFLoader
from . import config as cfg


def evaluate(arg_file=None, prog_func=None):
    cfg.set_evaluate_args(arg_file, prog_func)

    if not os.path.exists(cfg.log_dir):
        os.makedirs(cfg.log_dir)

    n_models = len(cfg.model_names)

    for i_model in range(n_models):

        if cfg.lstm_steps:
            time_steps = cfg.lstm_steps
        elif cfg.gru_steps:
            time_steps = cfg.gru_steps
        elif cfg.channel_steps:
            time_steps = cfg.channel_steps
        else:
            time_steps = 0

        ckpt_dict = load_ckpt("{}/{}".format(cfg.model_dir, cfg.model_names[i_model]), cfg.device)

        if "stat_target" in ckpt_dict.keys():
            stat_target = ckpt_dict["stat_target"]
        else:
            stat_target = None

        dataset_val = NetCDFLoader(cfg.data_root_dir, cfg.data_names, cfg.mask_dir, cfg.mask_names, "infill",
                                   cfg.data_types, time_steps, stat_target)

        if len(cfg.image_sizes) > 1:
            model = CRAINet(img_size=cfg.image_sizes[0],
                            enc_dec_layers=cfg.encoding_layers[0],
                            pool_layers=cfg.pooling_layers[0],
                            in_channels=2 * cfg.channel_steps + 1,
                            out_channels=cfg.out_channels,
                            fusion_img_size=cfg.image_sizes[1],
                            fusion_enc_layers=cfg.encoding_layers[1],
                            fusion_pool_layers=cfg.pooling_layers[1],
                            fusion_in_channels=(len(cfg.image_sizes) - 1 - len(cfg.target_data_indices)
                                                ) * (2 * cfg.channel_steps + 1),
                            bounds=dataset_val.bounds).to(cfg.device)
        else:
            model = CRAINet(img_size=cfg.image_sizes[0],
                            enc_dec_layers=cfg.encoding_layers[0],
                            pool_layers=cfg.pooling_layers[0],
                            in_channels=2 * cfg.channel_steps + 1,
                            out_channels=cfg.out_channels,
                            bounds=dataset_val.bounds).to(cfg.device)

        output_names = ["{}/{}".format(cfg.evaluation_dirs[0], name) for name in cfg.eval_names]
        outputs = []
        for k in range(len(ckpt_dict["labels"])):
            label = ckpt_dict["labels"][k]
            load_model(ckpt_dict, model, label=label)
            model.eval()
            outputs.append(infill(model, dataset_val))
            if cfg.split_outputs:
                create_outputs(outputs, dataset_val, output_names, stat_target, k)
                outputs = []
        if not cfg.split_outputs:
            create_outputs(outputs, dataset_val, output_names, stat_target)


if __name__ == "__main__":
    evaluate()
