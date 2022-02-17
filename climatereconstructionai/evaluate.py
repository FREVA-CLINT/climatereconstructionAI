import h5py
import sys
import os
from . import config as cfg
from .model.net import PConvLSTM
from .utils.evaluation import *
from .utils.netcdfloader import NetCDFLoader
from .utils.io import load_ckpt

def evaluate(arg_file=None):

    import time
    start_time = time.time()

    cfg.set_evaluate_args(arg_file)

    if not os.path.exists(cfg.log_dir):
        os.makedirs(cfg.log_dir)

    n_models = len(cfg.model_names)
    assert n_models == len(cfg.eval_names)

    for i_model in range(n_models):
        gt = None
        outputs = None

        if cfg.infill:

            dataset_val = NetCDFLoader(cfg.data_root_dir, cfg.img_names, cfg.mask_dir, cfg.mask_names, cfg.infill,
                                       cfg.data_types, cfg.lstm_steps, cfg.prev_next_steps)
            lstm = True
            if cfg.lstm_steps == 0:
                lstm = False

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
                                  lstm=lstm).to(cfg.device)
            else:
                model = PConvLSTM(radar_img_size=cfg.image_sizes[0],
                                  radar_enc_dec_layers=cfg.encoding_layers[0],
                                  radar_pool_layers=cfg.pooling_layers[0],
                                  radar_in_channels=2 * cfg.prev_next_steps + 1,
                                  radar_out_channels=cfg.out_channels,
                                  lstm=lstm).to(cfg.device)

            load_ckpt("{}/{}".format(cfg.model_dir,cfg.model_names[i_model]), [('model', model)], cfg.device)

            model.eval()

            gt, output = infill(model, dataset_val, cfg.partitions, "{}/{}".format(cfg.evaluation_dirs[0],cfg.eval_names[i_model]))
            outputs = {cfg.eval_names[i_model]: output}

        if cfg.create_report:
            if cfg.eval_range:
                r = (int(cfg.eval_range[0]), int(cfg.eval_range[1]))
                gt = h5py.File('{}{}'.format(cfg.evaluation_dirs[0], 'gt'), 'r').get(cfg.data_types[0])[r[0]:r[1], :, :]
                mask = h5py.File('{}{}'.format(cfg.evaluation_dirs[0], 'mask'), 'r').get(cfg.data_types[0])[r[0]:r[1], :, :]
            else:
                gt = h5py.File('{}{}'.format(cfg.evaluation_dirs[0], 'gt'), 'r').get(cfg.data_types[0])[:, :, :]
                mask = h5py.File('{}{}'.format(cfg.evaluation_dirs[0], 'mask'), 'r').get(cfg.data_types[0])[:, :, :]
            if gt.ndim == 4:
                gt = gt[:, 0, :, :]
            if mask.ndim == 4:
                mask = mask[:, 0, :, :]
            if cfg.eval_threshold:
                mask[gt < cfg.eval_threshold] = 1
            gt = ma.masked_array(gt, mask)[:, :, :]
            outputs = {}
            for i in range(len(cfg.evaluation_dirs)):
                if cfg.eval_range:
                    r = (int(cfg.eval_range[0]), int(cfg.eval_range[1]))
                    output = h5py.File('{}{}'.format(cfg.evaluation_dirs[i], 'output'), 'r').get(cfg.data_types[0])[r[0]:r[1], :, :]
                else:
                    output = h5py.File('{}{}'.format(cfg.evaluation_dirs[i], 'output'), 'r').get(cfg.data_types[0])[:, :, :]
                if output.ndim == 4:
                    output = output[:, 0, :, :]
                output[output < 0.0] = 0.0
                output = ma.masked_array(output, mask)[:, :, :]
                outputs[cfg.eval_names[i]] = output
            create_evaluation_report(gt, outputs)

        if cfg.create_images:
            r = (int(cfg.create_images[0]), int(cfg.create_images[1]))
            gt = h5py.File('{}{}'.format(cfg.evaluation_dirs[0], 'gt'), 'r').get(cfg.data_types[0])[r[0]:r[1], :, :]
            mask = h5py.File('{}{}'.format(cfg.evaluation_dirs[0], 'mask'), 'r').get(cfg.data_types[0])[r[0]:r[1], :, :]
            image = h5py.File('{}{}'.format(cfg.evaluation_dirs[0], 'image'), 'r').get(cfg.data_types[0])[r[0]:r[1], :, :]
            if gt.ndim == 4:
                gt = gt[:, 0, :, :]
            if mask.ndim == 4:
                mask = mask[:, 0, :, :]
            if image.ndim == 4:
                image = image[:, 0, :, :]
            image = ma.masked_array(image, 1-mask)[:, :, :]

            data_sets = {'GT': gt, 'Input': image}

            for i in range(len(cfg.evaluation_dirs)):
                output = h5py.File('{}{}'.format(cfg.evaluation_dirs[i], 'output_comp'), 'r').get(cfg.data_types[0])[r[0]:r[1], :, :]
                if output.ndim == 4:
                    output = output[:, 0, :, :]
                output[output < 0.0] = 0.0
                data_sets[cfg.eval_names[i]] = output

            create_video = False
            if cfg.create_video:
                create_video = True
            for key,value in data_sets.items():
                create_evaluation_images(key, value, create_video, save_dir='images/{}'.format(key))

        print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    evaluate()
