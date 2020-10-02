# -*- coding: utf-8 -*-
# author: Yonghao He
# description: 

from .hook import Hook


class CheckpointHook(Hook):

    def after_train_epoch(self, executor):
        if (executor.config_dict['epoch'] + 1) % executor.config_dict['save_interval'] == 0:
            executor.save()
