# -*- coding: utf-8 -*-

from .hook import Hook
from torch.nn.utils import clip_grad


class OptimizerHook(Hook):

    def __init__(self, grad_clip_cfg, training_epochs):
        super(OptimizerHook, self).__init__()
        assert isinstance(grad_clip_cfg, dict) or grad_clip_cfg is None
        self._grad_clip_cfg = grad_clip_cfg
        if self._grad_clip_cfg is not None:
            if 'duration' in self._grad_clip_cfg:
                self._grad_clip_duration = self._grad_clip_cfg['duration']
                assert self._grad_clip_duration > 0 and isinstance(self._grad_clip_duration, int)
                del self._grad_clip_cfg['duration']
            else:
                self._grad_clip_duration = training_epochs

    def _clip_grad(self, parameters):
        filtered_parameters = list(filter(lambda p: p.requires_grad and p.grad is not None, parameters))
        if len(filtered_parameters) > 0:
            return clip_grad.clip_grad_norm_(filtered_parameters, **self._grad_clip_cfg)

    def after_train_iter(self, executor):
        executor.config_dict['optimizer'].zero_grad()
        executor.config_dict['loss'].backward()
        if self._grad_clip_cfg is not None:
            if executor.config_dict['epoch'] < self._grad_clip_duration:
                grad_norm = self._clip_grad(executor.config_dict['model'].parameters())
                executor.config_dict['grad_norm'] = grad_norm
            else:
                executor.config_dict['grad_norm'] = 0

        executor.config_dict['optimizer'].step()

