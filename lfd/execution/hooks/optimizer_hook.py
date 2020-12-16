# -*- coding: utf-8 -*-

from .hook import Hook
from torch.nn.utils import clip_grad


class OptimizerHook(Hook):

    def __init__(self, grad_clip_cfg):
        super(OptimizerHook, self).__init__()
        assert isinstance(grad_clip_cfg, dict) or grad_clip_cfg is None
        self._grad_clip_cfg = grad_clip_cfg

    def _clip_grad(self, parameters):
        filtered_parameters = list(filter(lambda p: p.requires_grad and p.grad is not None, parameters))
        if len(filtered_parameters) > 0:
            return clip_grad.clip_grad_norm_(filtered_parameters, **self._grad_clip_cfg)

    def after_train_iter(self, executor):
        executor.config_dict['optimizer'].zero_grad()
        executor.config_dict['loss'].backward()
        if self._grad_clip_cfg is not None:
            grad_norm = self._clip_grad(executor.config_dict['model'].parameters())
            if grad_norm is not None:
                executor.config_dict['grad_norm'] = grad_norm

        executor.config_dict['optimizer'].step()

