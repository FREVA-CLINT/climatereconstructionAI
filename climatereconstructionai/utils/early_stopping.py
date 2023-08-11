import copy

import numpy as np


class early_stopping():
    def __init__(self, delta=0, patience=1e10) -> None:

        self.lowest_loss = 1e10
        self.global_iter_best = 0
        self.best_model = None

        self.value_index = 0
        self.pat_index = 0

        self.criterion_diff = delta

        self.relative = True
        self.min_delta = delta

        self.patience = patience

        self.patience_ar = np.zeros((self.patience,))

        self.terminate = False
        self.best_loss = False

    def eval_termination(self):
        delta = np.diff(self.patience_ar)

        if self.relative:
            delta = delta / self.patience_ar[1:]

        self.criterion_diff = self.min_delta + (delta).mean()

        if self.criterion_diff > 0:
            self.terminate = True
        else:
            self.terminate = False

    def update_patience_array(self, value):

        self.patience_ar[self.pat_index] = value

        if (self.pat_index + 1) % self.patience == 0:

            self.eval_termination()
            self.pat_index = self.patience - 2
            self.patience_ar = np.roll(self.patience_ar, -1)
        else:
            self.pat_index += 1

    def update_best_iter(self, value, global_iter, model_save=None):
        self.global_iter = global_iter
        if value < self.lowest_loss:
            self.lowest_loss = value
            self.global_iter_best = global_iter
            self.best_loss = True
            if model_save is not None:
                self.best_model = copy.deepcopy(model_save)
        else:
            self.best_loss = False

    def update(self, value, global_iter, model_save=None):

        self.update_best_iter(value, global_iter, model_save=model_save)
        self.update_patience_array(value)
