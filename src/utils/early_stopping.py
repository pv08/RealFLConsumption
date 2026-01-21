import copy
import logging

import numpy as np

from logging import INFO
from src.utils.logger import log
class EarlyStopping:
    def __init__(self, patience: int=20, delta: int=0, trace: bool=True, trace_func=log):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace = trace
        self.trace_func = trace_func
        self.best_model = None

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.cache_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.trace:
                self.trace_func(INFO, f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.cache_checkpoint(val_loss, model)
            self.counter = 0


    def cache_checkpoint(self, val_loss, model):
        if self.trace:
            self.trace_func(INFO, f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Caching model...")

        self.val_loss_min = val_loss
        self.best_model = copy.deepcopy(model)