# -*- coding: utf-8 -*-

from .hook import Hook


class EvaluationHook(Hook):

    def after_val_iter(self, executor):
        executor.config_dict['evaluator'].update(executor.config_dict['eval_results'])

    def after_val_epoch(self, executor):
        executor.config_dict['evaluator'].evaluate()
