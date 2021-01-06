# -*- coding: utf-8 -*-

import os
import logging
from collections import OrderedDict
import torch
from torch.nn.parallel import DataParallel

from .utils import *
from .hooks import *


class Executor(object):

    def __init__(self, config_dict):
        self.config_dict = config_dict

        # create work dir
        if not os.path.exists(self.config_dict['work_dir']):
            os.makedirs(self.config_dict['work_dir'])

        # create logger
        self.config_dict['logger'] = get_root_logger(self.config_dict['log_path'], log_level=logging.INFO)

        self.config_dict.update(epoch=0)  # the counter of train epoch (val epoch does not need a counter)
        self.config_dict.update(train_iter=0)  # the counter of train iter for whole training process
        self.config_dict.update(inner_train_iter=0)  # the counter of train iter for one train epoch
        self.config_dict.update(inner_val_iter=0)  # the counter of val iter for one val epoch
        self.config_dict.update(train_average_meter=AverageMeter())
        self.config_dict.update(val_average_meter=AverageMeter())

        # resume training or load checkpoint only
        if self.config_dict['resume_path'] is not None:
            self.resume_weight()
        elif self.config_dict['weight_path'] is not None:
            self.load()

        # use DataParallel to wrap the model
        self.config_dict['model'] = DataParallel(self.config_dict['model'], device_ids=self.config_dict['gpu_list']).cuda(torch.device('cuda', self.config_dict['gpu_list'][0]))

        # optimizer dict must be updated after DataParallel wrap
        if self.config_dict['resume_path'] is not None:
            self.resume_optimizer()

        # register hooks
        self._hooks = list()
        self._register_all_hooks()

    def _register_hook(self, hook, priority='NORMAL'):
        """
        register hooks with priorities
        :return:
        """
        priority = get_priority(priority)
        hook.priority = priority

        inserted = False
        for i in range(len(self._hooks)-1, -1, -1):
            if priority >= self._hooks[i].priority:
                self._hooks.insert(i + 1, hook)
                inserted = True
                break
        if not inserted:
            self._hooks.insert(0, hook)

    def _register_checkpoint_hook(self):
        checkpoint_hook = CheckpointHook()
        self._register_hook(checkpoint_hook, 'LOWEST')

    def _register_logger_hook(self):
        logger_hook = LoggerHook()
        self._register_hook(logger_hook, 'VERY_LOW')

    def _register_lr_scheduler_hook(self):
        lr_scheduler_hook = LrSchedulerHook(**self.config_dict['warmup_setting']) if 'warmup_setting' in self.config_dict else LrSchedulerHook()
        self._register_hook(lr_scheduler_hook, 'NORMAL')

    def _register_optimizer_hook(self):
        optimizer_grad_clip_cfg = self.config_dict.get('optimizer_grad_clip_cfg', None)
        training_epochs = self.config_dict['training_epochs']
        optimizer_hook = OptimizerHook(optimizer_grad_clip_cfg, training_epochs)
        self._register_hook(optimizer_hook, 'NORMAL')

    def _register_speed_hook(self):
        speed_hook = SpeedHook()
        self._register_hook(speed_hook, 'LOW')

    def _register_evaluation_hook(self):
        evaluation_hook = EvaluationHook()
        self._register_hook(evaluation_hook, 'NORMAL')

    def _register_all_hooks(self):
        self._register_checkpoint_hook()
        self._register_logger_hook()
        self._register_lr_scheduler_hook()
        self._register_optimizer_hook()
        self._register_speed_hook()
        self._register_evaluation_hook()

    def _generate_meta(self):
        """
        select some properties from config_dict as meta info
        all config params with basic types are regarded as meta info
        :return:
        """
        types = [str, int, float, list, dict, bool, type(None), OrderedDict]
        meta = dict()

        # add info to meta
        meta.update({k: v for k, v in self.config_dict.items() if type(v) in types})

        return meta

    def _call_hooks(self, fn_name):
        """
        possible fn_names: before_run, after_run, before_epoch, after_epoch,
        before_iter, after_iter, before_train_epoch, before_val_epoch, after_train_epoch, after_val_epoch,
        before_train_iter, before_val_iter, after_train_iter, after_val_iter
        :param fn_name:
        :return:
        """
        for hook in self._hooks:
            getattr(hook, fn_name)(self)

    def save(self):
        """
        save training state according to the config
        :return:
        """
        save_path = os.path.join(self.config_dict['work_dir'], 'epoch_'+str(self.config_dict['epoch'])+'.pth')
        save_checkpoint(self.config_dict['model'], save_path=save_path, optimizer=self.config_dict['optimizer'], meta=self._generate_meta())

    def load(self):
        """
        only load model weights
        :return:
        """
        self.config_dict['logger'].info('Load weights from checkpoint:{}'.format(self.config_dict['weight_path']))
        load_checkpoint(self.config_dict['model'], load_path=self.config_dict['weight_path'], strict=True, logger=self.config_dict['logger'])

    def resume_weight(self):
        """
        :return:
        """
        self.config_dict['logger'].info('Resume training from checkpoint:{}'.format(self.config_dict['resume_path']))

        checkpoint = load_checkpoint(self.config_dict['model'], load_path=self.config_dict['resume_path'], strict=True, logger=self.config_dict['logger'])
        self.config_dict['checkpoint'] = checkpoint

        # filter some configs
        checkpoint['meta'].pop('timestamp') if 'timestamp' in checkpoint['meta'] else None
        checkpoint['meta'].pop('work_dir') if 'work_dir' in checkpoint['meta'] else None
        checkpoint['meta'].pop('log_path') if 'log_path' in checkpoint['meta'] else None
        checkpoint['meta'].pop('training_epochs') if 'training_epochs' in checkpoint['meta'] else None
        checkpoint['meta'].pop('gpu_list') if 'gpu_list' in checkpoint['meta'] else None
        checkpoint['meta'].pop('display_interval') if 'display_interval' in checkpoint['meta'] else None
        checkpoint['meta'].pop('save_interval') if 'save_interval' in checkpoint['meta'] else None
        checkpoint['meta'].pop('val_interval') if 'val_interval' in checkpoint['meta'] else None
        checkpoint['meta'].pop('weight_path') if 'weight_path' in checkpoint['meta'] else None
        checkpoint['meta'].pop('resume_path') if 'resume_path' in checkpoint['meta'] else None
        checkpoint['meta'].pop('batch_size') if 'batch_size' in checkpoint['meta'] else None
        checkpoint['meta'].pop('num_train_workers') if 'num_train_workers' in checkpoint['meta'] else None
        checkpoint['meta'].pop('num_val_workers') if 'num_val_workers' in checkpoint['meta'] else None
        checkpoint['meta'].pop('train_dataset_path') if 'train_dataset_path' in checkpoint['meta'] else None
        checkpoint['meta'].pop('optimizer_grad_clip_cfg') if 'optimizer_grad_clip_cfg' in checkpoint['meta'] else None

        self.config_dict.update(checkpoint['meta'])

    def resume_optimizer(self):
        if 'optimizer_state_dict' in self.config_dict['checkpoint']:
            self.config_dict['optimizer'].load_state_dict(self.config_dict['checkpoint']['optimizer_state_dict'])

    def get_current_lr(self):
        """
        since the optimizer stores all learning rates for each group (in most cases, only one group exists), the first lr is returned
        :return:
        """
        return self.config_dict['optimizer'].param_groups[0]['lr']

    def train(self):
        self.config_dict['mode'] = 'train'
        self.config_dict['model'].train()

        self._call_hooks('before_train_epoch')

        for i, data_batch in enumerate(self.config_dict['train_data_loader']):
            self.config_dict.update(inner_train_iter=i)
            self._call_hooks('before_train_iter')

            image_batch, annotation_batch, meta_batch = data_batch
            self.config_dict.update(batch_size=len(annotation_batch))  # dynamically update batch_size, because the last iter may contain less samples than the initial
            image_batch = torch.from_numpy(image_batch)
            predict_outputs = self.config_dict['model'](image_batch)
            if hasattr(self.config_dict['model'], 'module'):
                loss_dict = self.config_dict['model'].module.get_loss(predict_outputs, annotation_batch, meta_batch)
            else:
                loss_dict = self.config_dict['model'].get_loss(predict_outputs, annotation_batch, meta_batch)
            self.config_dict.update(loss=loss_dict['loss'])

            # update train average meter for losses
            loss_values_dict = loss_dict['loss_values']
            for name, value in loss_values_dict.items():
                self.config_dict['train_average_meter'].update(name, value, self.config_dict['batch_size'])

            self.config_dict['train_iter'] += 1
            self._call_hooks('after_train_iter')

        self.config_dict['epoch'] += 1
        self._call_hooks('after_train_epoch')

    def val(self):
        self.config_dict['mode'] = 'val'
        self.config_dict['model'].eval()

        self._call_hooks('before_val_epoch')

        for i, data_batch in enumerate(self.config_dict['val_data_loader']):
            self.config_dict.update(inner_val_iter=i)
            self._call_hooks('before_val_iter')

            image_batch, annotation_batch, meta_batch = data_batch
            self.config_dict.update(batch_size=len(annotation_batch))
            with torch.no_grad():
                image_batch = torch.from_numpy(image_batch)
                predict_outputs = self.config_dict['model'](image_batch)
                if hasattr(self.config_dict['model'], 'module'):
                    loss_dict = self.config_dict['model'].module.get_loss(predict_outputs, annotation_batch, meta_batch)
                    predict_results = self.config_dict['model'].module.get_results(predict_outputs, meta_batch)
                else:
                    loss_dict = self.config_dict['model'].get_loss(predict_outputs, annotation_batch, meta_batch)
                    predict_results = self.config_dict['model'].get_results(predict_outputs, meta_batch)

            # update val average meter for losses
            loss_values_dict = loss_dict['loss_values']
            for name, value in loss_values_dict.items():
                self.config_dict['val_average_meter'].update(name, value, self.config_dict['batch_size'])

            self.config_dict.update(eval_results=(predict_results, meta_batch))  # used by evaluation hook

            self._call_hooks('after_val_iter')

        self._call_hooks('after_val_epoch')

    def run(self):

        self._call_hooks('before_run')

        while self.config_dict['epoch'] < self.config_dict['training_epochs']:
            self.train()

            if self.config_dict['evaluator'] is not None and self.config_dict['val_interval'] > 0 and (self.config_dict['epoch']) % self.config_dict['val_interval'] == 0:
                self.val()

        self._call_hooks('after_run')
