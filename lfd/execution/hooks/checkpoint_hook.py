# -*- coding: utf-8 -*-

from .hook import Hook


class CheckpointHook(Hook):

    def after_train_epoch(self, executor):
        if (executor.config_dict['epoch']) % executor.config_dict['save_interval'] == 0:
            executor.save()
