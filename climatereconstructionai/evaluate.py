import os

from .model.net import CRAINet
from .utils.evaluation import infill
from .utils.io import load_ckpt, load_model
from .utils.netcdfloader import NetCDFLoader
import xarray as xr
from . import config as cfg


def store_encoding(ds):
    global encoding
    encoding = ds['time'].encoding
    return ds


def evaluate(arg_file=None, prog_func=None):
    cfg.set_evaluate_args(arg_file, prog_func)

    if not os.path.exists(cfg.log_dir):
        os.makedirs(cfg.log_dir)

    n_models = len(cfg.model_names)

    eval_path = ["{}/{}".format(cfg.evaluation_dirs[0], name) for name in cfg.eval_names]
    output_names = {}
    count = 0
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

        image_sizes = dataset_val.img_sizes
        if cfg.conv_factor is None:
            cfg.conv_factor = max(image_sizes[0])

        if len(image_sizes) - cfg.n_target_data > 1:
            model = CRAINet(img_size=image_sizes[0],
                            enc_dec_layers=cfg.encoding_layers[0],
                            pool_layers=cfg.pooling_layers[0],
                            in_channels=2 * cfg.channel_steps + 1,
                            out_channels=cfg.out_channels,
                            fusion_img_size=image_sizes[1],
                            fusion_enc_layers=cfg.encoding_layers[1],
                            fusion_pool_layers=cfg.pooling_layers[1],
                            fusion_in_channels=(len(image_sizes) - 1 - cfg.n_target_data
                                                ) * (2 * cfg.channel_steps + 1),
                            bounds=dataset_val.bounds).to(cfg.device)
        else:
            model = CRAINet(img_size=image_sizes[0],
                            enc_dec_layers=cfg.encoding_layers[0],
                            pool_layers=cfg.pooling_layers[0],
                            in_channels=2 * cfg.channel_steps + 1,
                            out_channels=cfg.out_channels,
                            bounds=dataset_val.bounds).to(cfg.device)

        for k in range(len(ckpt_dict["labels"])):
            count += 1
            label = ckpt_dict["labels"][k]
            load_model(ckpt_dict, model, label=label)
            model.eval()
            infill(model, dataset_val, eval_path, output_names, stat_target, count)

    for name in output_names:
        if len(output_names[name]) == 1:
            os.rename(output_names[name][0], name + ".nc")
        else:
            if not cfg.split_outputs:
                ds = xr.open_mfdataset(output_names[name], preprocess=store_encoding, autoclose=True, combine='nested',
                                       data_vars='minimal', concat_dim="time", chunks={})
                ds['time'].encoding = encoding
                ds['time'].encoding['original_shape'] = len(ds["time"])
                ds.to_netcdf(name + ".nc")
                for output_name in output_names[name]:
                    os.remove(output_name)


if __name__ == "__main__":
    evaluate()
