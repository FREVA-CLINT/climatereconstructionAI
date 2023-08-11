import os

from tensorboardX import SummaryWriter

from . import visualization


class writer():

    def __init__(self, log_dir) -> None:

        paths = [int(path) for path in os.listdir(log_dir) if str.isnumeric(path)]
        self.suffix = '0' if len(paths) == 0 else str(max(paths) + 1)

        layout = {}
    
        self.writer = SummaryWriter(log_dir, filename_suffix=self.suffix)
        self.writer.add_custom_scalars(layout)


    def set_hparams(self, parameters):
        [self.writer.add_text(key, str(parameters[key])) for key in parameters.keys()]
        self.hparams = parameters

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

    def add_visualizations(self):
        pass

    def close(self):
        self.writer.close()
