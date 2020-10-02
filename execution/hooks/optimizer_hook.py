# -*- coding: utf-8 -*-

from .hook import Hook


class OptimizerHook(Hook):

    def after_train_iter(self, executor):
        executor.config_dict['optimizer'].zero_grad()
        executor.config_dict['loss'].backward()
        executor.config_dict['optimizer'].step()

