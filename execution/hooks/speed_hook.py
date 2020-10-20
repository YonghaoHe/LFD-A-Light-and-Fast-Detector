# -*- coding: utf-8 -*-

import time
from .hook import Hook


class SpeedHook(Hook):

    def __init__(self):
        super(SpeedHook, self).__init__()
        self._train_start_time = 0
        self._val_start_time = 0

    def before_train_iter(self, executor):
        self._train_start_time = time.time()

    def before_val_iter(self, executor):
        self._val_start_time = time.time()

    def after_train_iter(self, executor):
        time_elapsed = time.time() - self._train_start_time
        executor.config_dict['train_average_meter'].update('speed', executor.config_dict['batch_size'], time_elapsed)

    def after_val_iter(self, executor):
        time_elapsed = time.time() - self._val_start_time
        executor.config_dict['val_average_meter'].update('speed', executor.config_dict['batch_size'], time_elapsed)

