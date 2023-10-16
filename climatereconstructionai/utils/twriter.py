import os

from tensorboardX import SummaryWriter

from . import visualization
from .. import config as cfg


class writer():

    def __init__(self) -> None:

        if not os.path.exists(cfg.log_dir):
            os.makedirs(cfg.log_dir)

        paths = [int(path) for path in os.listdir(cfg.log_dir) if str.isnumeric(path)]
        self.suffix = '0' if len(paths) == 0 else str(max(paths) + 1)

        layout = {}
        layout['losses'] = {}
        for loss in cfg.lambda_dict.keys():
            layout['losses'][loss] = ['Multiline', ['{}/train'.format(loss), '{}/val'.format(loss)]]
        layout['losses']['total'] = ['Multiline', ['total/train', 'total/val']]

        self.writer = SummaryWriter(cfg.log_dir, filename_suffix=self.suffix)

        self.writer.add_custom_scalars(layout)

        metric_names = []
        if cfg.val_metrics is not None:
            metric_names += [f'metric/val/{metric}' for metric in cfg.val_metrics]

        self.metrics = {}
        if len(metric_names) > 0:
            self.metrics = dict(zip(metric_names, [0] * len(metric_names)))

    def get_hparams(self, parameters_dict):
        hparams_dict = {}
        for key, value in parameters_dict.items():
            if isinstance(value, list):
                if key=='data_names':
                    for k, data_name in enumerate(value):
                        if str.isnumeric(data_name.replace('.nc','')):
                            if k < (len(value)-cfg.n_target_data):
                                hparams_dict['source'] = int(data_name.replace('.nc',''))
                            else:
                                hparams_dict['target'] = int(data_name.replace('.nc',''))
                else:
                    value = value[0]
            if type(value) in (int, float) or ((type(value) == bool) and ('plot' not in key)):
                hparams_dict[key] = value
        return hparams_dict

    def set_hparams(self, parameters):
        [self.writer.add_text(key, str(parameters[key])) for key in parameters.keys()]
        hparams = self.get_hparams(parameters)
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
            name_tag = self.suffix
        self.writer.add_figure(name_tag, fig, global_step=iter_index)

    def add_error_maps(self, mask, steady_mask, output, gt, iter_index, setname):
        error_maps = visualization.get_all_error_maps(mask, steady_mask, output, gt, num_samples=3)

        for error_map, name in zip(error_maps, ['error', 'relative error', 'abs error', 'relative abs error']):
            self.add_figure(error_map, iter_index, f'map/{setname}/{name}')

    def add_correlation_plots(self, mask, steady_mask, output, gt, iter_index, setname):
        fig = visualization.create_correlation_plot(mask, steady_mask, output, gt)
        self.add_figure(fig, iter_index, name_tag=f'plot/{setname}/correlation')

    def add_error_dist_plot(self, mask, steady_mask, output, gt, iter_index, setname):
        fig = visualization.create_error_dist_plot(mask, steady_mask, output, gt)
        self.add_figure(fig, iter_index, name_tag=f'plot/{setname}/error_dist')

    def add_maps(self, mask, steady_mask, output, gt, input, iter_index, setname):
        fig = visualization.create_map(mask, steady_mask, output, gt,input, num_samples=3)
        self.add_figure(fig, iter_index, name_tag=f'map/{setname}/values')

    def add_distribution(self, values, iter_index, name_tag=None):
        if name_tag is None:
            name_tag = self.suffix
        self.writer.add_histogram(name_tag, values, global_step=iter_index)

    def add_distributions(self, mask, steady_mask, output, gt, iter_index, setname):

        errors_dists = visualization.get_all_error_distributions(mask, steady_mask, output, gt, num_samples=1000)
        for error_dist, suffix in zip(errors_dists, ['error', 'abs error', 'relative error', 'relative abs error']):
            name = f'dist/{setname}/{suffix}'
            # entries in value_list correspond to channels
            for ch, values in enumerate(error_dist):
                self.add_distribution(values, iter_index, name_tag=f'{name}_channel{ch}')

    def add_visualizations(self, mask, steady_mask, output, gt, input, iter_index, setname):
        if "correlation" in cfg.tensor_plots:
            self.add_correlation_plots(mask, steady_mask, output, gt, iter_index, setname)
            self.add_error_dist_plot(mask, steady_mask, output, gt, iter_index, setname)

        if "distribution" in cfg.tensor_plots:
            self.add_distributions(mask, steady_mask, output, gt, iter_index, setname)

        if "error" in cfg.tensor_plots:
            self.add_error_maps(mask, steady_mask, output, gt, iter_index, setname)
            self.add_maps(mask, steady_mask, output, gt, input, iter_index, setname)

    def close(self):
        self.writer.close()
