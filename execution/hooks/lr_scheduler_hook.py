# -*- coding: utf-8 -*-

from __future__ import division

from .hook import Hook


class LrSchedulerHook(Hook):

    def __init__(self,
                 by_epoch=False,  # determine the basic unit to change lr in warmup: False-based on 'iter';True-based on 'epoch'
                 warmup_mode=None,
                 warmup_loops=0,  # to avoid confusion, we use 'loops' instead of 'iters'
                 warmup_ratio=0.1):
        super(LrSchedulerHook, self).__init__()
        # validate the "warmup" argument
        if warmup_mode is not None:
            assert warmup_mode in ['constant', 'linear', 'exp'], '"{}" is not a supported type for warming up, valid types are "constant", "linear" and "exp"'.format(warmup_mode)
            assert warmup_loops >= 0, '"warmup_iters" must be a non-negative integer!'
            assert 0 < warmup_ratio <= 1.0, '"warmup_ratio" must be in range (0,1]!'

        self._by_epoch = by_epoch
        self._warmup_mode = warmup_mode
        self._warmup_loops = warmup_loops
        self._warmup_ratio = warmup_ratio
        self._epochs_warmup_skips = 0  # record the number of epochs the warmup process skips
        self._resume_init_flag = False

        self._base_lr = []  # initial lr for all param groups

    def _set_lr(self, executor, lr_groups):
        for param_group, lr in zip(executor.config_dict['optimizer'].param_groups, lr_groups):
            param_group['lr'] = lr

    def get_warmup_lr(self, current_loops):

        if self._warmup_mode == 'constant':

            warmup_lr = [_lr * self._warmup_ratio for _lr in self._base_lr]

        elif self._warmup_mode == 'linear':

            k = (1 - current_loops / self._warmup_loops) * (1 - self._warmup_ratio)
            warmup_lr = [_lr * (1 - k) for _lr in self._base_lr]

        elif self._warmup_mode == 'exp':

            k = self._warmup_ratio ** (1 - current_loops / self._warmup_loops)
            warmup_lr = [_lr * k for _lr in self._base_lr]

        else:
            raise ValueError('Unknown warmup mode: {}'.format(self._warmup_mode))

        return warmup_lr

    def _resume_init_lr(self, executor):
        if not self._resume_init_flag:
            self._set_lr(executor, self._base_lr)  # resume init lr
            self._resume_init_flag = True

    def before_run(self, executor):
        # NOTE: when resuming from a checkpoint, if 'initial_lr' is not saved,
        # it will be set according to the optimizer params
        for group in executor.config_dict['optimizer'].param_groups:
            group.setdefault('initial_lr', group['lr'])
        self._base_lr = [group['initial_lr'] for group in executor.config_dict['optimizer'].param_groups]

    def before_train_epoch(self, executor):
        if self._by_epoch:
            current_loop = executor.config_dict['epoch'] + 1
            if self._warmup_mode is not None and current_loop <= self._warmup_loops:
                warmup_lr = self.get_warmup_lr(current_loop)
                self._set_lr(executor, warmup_lr)
            else:
                self._resume_init_lr(executor)
                if self._epochs_warmup_skips > 0:
                    for i in range(self._epochs_warmup_skips):
                        executor.config_dict['lr_scheduler'].step()
                    self._epochs_warmup_skips = 0

    def before_train_iter(self, executor):
        if not self._by_epoch:
            current_loop = executor.config_dict['train_iter'] + 1
            if self._warmup_mode is not None and current_loop <= self._warmup_loops:
                warmup_lr = self.get_warmup_lr(current_loop)
                self._set_lr(executor, warmup_lr)
            else:
                self._resume_init_lr(executor)
                if self._epochs_warmup_skips > 0:
                    for i in range(self._epochs_warmup_skips):
                        executor.config_dict['lr_scheduler'].step()
                    self._epochs_warmup_skips = 0

    def after_train_epoch(self, executor):
        # the lr scheduler does not execute step while warming up
        current_loop = executor.config_dict['train_iter'] if not self._by_epoch else executor.config_dict['epoch']
        if self._warmup_mode is not None and current_loop <= self._warmup_loops:
            self._epochs_warmup_skips += 1
        else:
            executor.config_dict['lr_scheduler'].step()
