import sys

sys.path.append('./')

from model.PConvLSTM import PConvLSTM
from utils.evaluation import *
from utils.netcdfloader import NetCDFLoader
from utils.io import load_ckpt
import config as cfg

cfg.set_evaluation_args()
gt = None
outputs = None

if cfg.infill:
    dataset_val = NetCDFLoader(cfg.data_root_dir, cfg.img_names, cfg.mask_dir, cfg.mask_names, cfg.infill,
                               cfg.data_types, cfg.lstm_steps)
    lstm = True
    if cfg.lstm_steps == 0:
        lstm = False
    model = PConvLSTM(image_size=cfg.image_size,
                      num_enc_dec_layers=cfg.encoding_layers,
                      num_pool_layers=cfg.pooling_layers,
                      num_in_channels=len(cfg.data_types),
                      lstm=lstm).to(cfg.device)

    load_ckpt(cfg.snapshot_dir, [('model', model)], cfg.device)

    model.eval()

    gt, output = infill(model, dataset_val, cfg.partitions)
    outputs = {cfg.eval_names[0]: output}

if cfg.create_report:
    r = (122,122)
    if gt is None or outputs is None:
        gt = h5py.File('{}{}'.format(cfg.evaluation_dirs[0], 'gt'), 'r').get(cfg.data_types[0])[:, :, :]
        mask = h5py.File('{}{}'.format(cfg.evaluation_dirs[0], 'mask'), 'r').get(cfg.data_types[0])[:, :, :]
        if gt.ndim == 4:
            gt = gt[:, 0, :, :]
        if mask.ndim == 4:
            mask = mask[:, 0, :, :]
        if cfg.mask_zero:
            mask[gt < cfg.mask_zero] = 1
        gt = ma.masked_array(gt, mask)[r[0]:r[1], :, :]
        outputs = {}
        for i in range(len(cfg.evaluation_dirs)):
            output = h5py.File('{}{}'.format(cfg.evaluation_dirs[i], 'output'), 'r').get(cfg.data_types[0])[:, :, :]
            output = ma.masked_array(output, mask)[r[0]:r[1], :, :]
            output[output < 0.0] = 0.0
            if output.ndim == 4:
                output = output[:, 0, :, :]
            outputs[cfg.eval_names[i]] = output
    create_evaluation_report(gt, outputs)

if cfg.create_images:
    start_date = cfg.create_images.split(',')[0]
    end_date = cfg.create_images.split(',')[1]
    create_video = False
    if cfg.create_video:
        create_video = True
    create_evaluation_images('image.nc', create_video, start_date, end_date)
    create_evaluation_images('gt.nc', create_video, start_date, end_date)
    create_evaluation_images('output.nc', create_video, start_date, end_date)
    create_evaluation_images('output_comp.nc', create_video, start_date, end_date)