import os

from .model.net import CRAINet
from .utils.evaluation import infill, get_batch_size
from .utils.io import load_ckpt, load_model
from torch.utils.data import DataLoader
from .utils.netcdfloader import NetCDFLoader, FiniteSampler
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

        ckpt_dict = load_ckpt("{}/{}".format(cfg.model_dir, cfg.model_names[i_model]), cfg.device)

        if cfg.use_train_stats:
            data_stats = ckpt_dict["train_stats"]
        else:
            data_stats = None

        dataset_val = NetCDFLoader(cfg.data_root_dir, cfg.data_names, cfg.mask_dir, cfg.mask_names, "infill",
                                   cfg.data_types, cfg.time_steps, data_stats)

        n_samples = len(dataset_val)

        if data_stats is None:
            if cfg.normalize_data:
                print("* Warning! Using mean and std from current data.")
                if cfg.n_target_data != 0:
                    print("* Warning! Mean and std from target data will be used to renormalize output."
                          " Mean and std from training data can be used with use_train_stats option.")
            data_stats = {"mean": dataset_val.img_mean, "std": dataset_val.img_std}

        image_sizes = dataset_val.img_sizes
        if cfg.conv_factor is None:
            cfg.conv_factor = max(image_sizes[0])

        if len(image_sizes) - cfg.n_target_data > 1:
            model = CRAINet(img_size=image_sizes[0],
                            enc_dec_layers=cfg.encoding_layers[0],
                            pool_layers=cfg.pooling_layers[0],
                            in_channels=cfg.n_channel_steps,
                            out_channels=cfg.out_channels,
                            fusion_img_size=image_sizes[1],
                            fusion_enc_layers=cfg.encoding_layers[1],
                            fusion_pool_layers=cfg.pooling_layers[1],
                            fusion_in_channels=(len(image_sizes) - 1 - cfg.n_target_data) * cfg.n_channel_steps,
                            bounds=dataset_val.bounds).to(cfg.device)
        else:
            model = CRAINet(img_size=image_sizes[0],
                            enc_dec_layers=cfg.encoding_layers[0],
                            pool_layers=cfg.pooling_layers[0],
                            in_channels=cfg.n_channel_steps,
                            out_channels=cfg.out_channels,
                            bounds=dataset_val.bounds).to(cfg.device)

        for k in range(len(ckpt_dict["labels"])):
            count += 1
            label = ckpt_dict["labels"][k]
            load_model(ckpt_dict, model, label=label)
            model.eval()
            batch_size = get_batch_size(model.parameters(), n_samples, image_sizes)
            iterator_val = iter(DataLoader(dataset_val, batch_size=batch_size,
                                           sampler=FiniteSampler(len(dataset_val)), num_workers=0))
            infill(model, iterator_val, eval_path, output_names, data_stats, dataset_val.xr_dss, count)

    for name in output_names:
        if len(output_names[name]) == 1 and len(output_names[name][1]) == 1:
            os.rename(output_names[name][1][0], name + ".nc")
        else:
            if not cfg.split_outputs:
                dss = []
                for i_model in output_names[name]:
                    dss.append(xr.open_mfdataset(output_names[name][i_model], preprocess=store_encoding, autoclose=True,
                                                combine='nested', data_vars='minimal', concat_dim="time", chunks={}))
                    dss[-1] = dss[-1].assign_coords({"member": i_model})

                if len(dss) == 1:
                    ds = dss[-1].drop("member")
                else:
                    ds = xr.concat(dss, dim="member")

                ds['time'].encoding = encoding
                ds['time'].encoding['original_shape'] = len(ds["time"])
                ds = ds.transpose("time", ...).reset_coords(drop=True)
                ds.to_netcdf(name + ".nc")

                for i_model in output_names[name]:
                    for output_name in output_names[name][i_model]:
                        os.remove(output_name)


if __name__ == "__main__":
    evaluate()
