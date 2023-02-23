from tensorboardX import SummaryWriter
from climatereconstructionai import config as cfg
from .io import get_hparams
import os
from datetime import datetime

class writer():

    def __init__(self) -> None:
        
        if cfg.writer_mode == 'model_config':

            sub_dir = (
                f'Nenc-{cfg.encoding_layers[0]}' +
                f'_Npool-{cfg.pooling_layers[0]}_Fconv-{cfg.conv_factor}' +
                f'_LSTM-{cfg.lstm_steps}_Ch-{cfg.channel_steps}_Att-{int(cfg.attention)}'
                )

        elif cfg.writer_mode == 'numeric_asc':
            paths = [int(path) for path in os.listdir(cfg.log_dir) if str.isnumeric(path)]
            sub_dir = '0' if len(paths)==0 else str(max(paths)+1)

        elif cfg.writer_mode == 'snapshot_subdir':         
            sub_dir = os.path.basename(cfg.snapshot_dir)

        cfg.log_dir = os.path.join(cfg.log_dir, sub_dir)
        
        if not os.path.exists(cfg.log_dir):
            os.makedirs(cfg.log_dir)

        if cfg.writer_mode=='snapshot_subdir':
            self.suffix = os.path.basename(cfg.snapshot_dir)
        else:
            paths = [int(path) for path in os.listdir(cfg.log_dir) if str.isnumeric(path)]
            self.suffix = '0' if len(paths)==0 else str(max(paths)+1)
        
        layout = {}
        layout['losses'] = {}
        for loss in cfg.lambda_dict.keys():
            layout['losses'][loss] = ['Multiline', ['{}/train'.format(loss),'{}/val'.format(loss)]]
        layout['losses']['total'] = ['Multiline', ['total/train'.format(loss),'total/val'.format(loss)]]
        
        self.writer = SummaryWriter(cfg.log_dir, filename_suffix=self.suffix)

        self.writer.add_custom_scalars(layout)
        
        metric_names = []
        if cfg.val_metrics is not None:
            metric_names += [f'metric/val/{metric}' for metric in cfg.val_metrics]
        if cfg.train_metrics is not None:
            metric_names += [f'metric/train/{metric}' for metric in cfg.train_metrics]
        if cfg.test_metrics is not None:
            metric_names += [f'metric/test/{metric}' for metric in cfg.test_metrics]

        self.metrics = {}
        if len(metric_names)>0:
            self.metrics = dict(zip(metric_names,[0]*len(metric_names)))

    def set_hparams(self,parameters):
        [self.writer.add_text(key,str(parameters[key])) for key in parameters.keys()]
        hparams = get_hparams(parameters)
        hparams.update(cfg.lambda_dict)
        self.hparams = hparams


    def update_scalar(self, setname, loss_name, value, iter_index):
        self.writer.add_scalar('{:s}/{:s}'.format(loss_name, setname), value, iter_index)

    def update_scalars(self, scalars: dict, iter_index: int, setname: str):
        for scalar, scalar_value in scalars.items():
            self.writer.add_scalar('{:s}/{:s}'.format(scalar, setname), scalar_value, iter_index)

    def update_hparams(self, metrics, iter_index):
        self.metrics.update(metrics)
        self.writer.add_hparams(self.hparams, self.metrics, name=self.suffix, global_step=iter_index)

    def add_figure(self, fig, iter_index, name_tag=None):
        if name_tag is None:
            name_tag=self.suffix
        self.writer.add_figure(name_tag, fig, global_step=iter_index)

    def add_distribution(self, values, iter_index, name_tag=None):
        if name_tag is None:
            name_tag=self.suffix
        self.writer.add_histogram(name_tag, values, global_step=iter_index)

    def add_distributions(self, value_list, iter_index, name):
        #entries in value_list correspond to channels
        for ch, values in enumerate(value_list):
            self.add_distribution(values, iter_index, name_tag=f'{name}_channel{ch}')

    def close(self):
        self.writer.close()
