# -*- coding: utf-8 -*-

from .hook import Hook
from ..utils import collect_envs


class LoggerHook(Hook):

    def _get_log_info(self, executor):

        # format information
        if executor.config_dict['mode'] == 'train':
            log_str = 'Epoch[{}][{}/{}], lr:{:.5f}'.format(executor.config_dict['epoch'] + 1, executor.config_dict['inner_train_iter'] + 1, len(executor.config_dict['train_data_loader']), executor.get_current_lr())
            log_str += ', speed:{:.2f} images/s'.format(executor.config_dict['train_average_meter'].get_average(name='speed', avg_mode='sum'))

            # add loss
            loss_names = [name for name in executor.config_dict['train_average_meter'].get_all_names() if 'loss' in name]  # loss names are selected by checking the substring 'loss'
            for loss_name in loss_names:
                log_str += ', {}:{:.5f}'.format(loss_name, executor.config_dict['train_average_meter'].get_average(name=loss_name, avg_mode='weighted_sum'))

        else:
            # val process has only one epoch
            log_str = 'Val Epoch[{}/{}]'.format(executor.config_dict['inner_val_iter'] + 1, len(executor.config_dict['val_data_loader']))
            log_str += ', speed:{:.2f} images/s'.format(executor.config_dict['val_average_meter'].get_average(name='speed', avg_mode='sum'))

            # add loss
            loss_names = [name for name in executor.config_dict['val_average_meter'].get_all_names() if 'loss' in name]
            for loss_name in loss_names:
                log_str += ', {}:{:.5f}'.format(loss_name, executor.config_dict['val_average_meter'].get_average(name=loss_name, avg_mode='weighted_sum'))

        return log_str

    def before_run(self, executor):
        """
        print info before training
        :param executor:
        :return:
        """
        env_info = collect_envs()
        executor.config_dict['logger'].info('Training environment summary --------')
        for k, v in env_info.items():
            executor.config_dict['logger'].info('{:<20}:{}'.format(k, v))

        executor.config_dict['logger'].info('-----------------------------------------------')
        executor.config_dict['logger'].info('Training settings --------')
        executor.config_dict['logger'].info('{:<20}:{}'.format('work_dir', executor.config_dict['work_dir']))
        executor.config_dict['logger'].info('{:<20}:{}'.format('training_epochs', executor.config_dict['training_epochs']))
        executor.config_dict['logger'].info('{:<20}:{}'.format('batch_size', executor.config_dict['batch_size']))
        executor.config_dict['logger'].info('{:<20}:{}'.format('seed', executor.config_dict['seed']))
        executor.config_dict['logger'].info('{:<20}:{}'.format('cudnn_benchmark', executor.config_dict['cudnn_benchmark']))
        executor.config_dict['logger'].info('{:<20}:{}'.format('gpu_list', ','.join([str(i) for i in executor.config_dict['gpu_list']])))
        executor.config_dict['logger'].info('{:<20}:{}'.format('display_interval', executor.config_dict['display_interval']))
        executor.config_dict['logger'].info('{:<20}:{}'.format('save_interval', executor.config_dict['save_interval']))
        executor.config_dict['logger'].info('{:<20}:{}'.format('val_interval', executor.config_dict['val_interval']))
        executor.config_dict['logger'].info('{:<20}:{}'.format('num_train_workers', executor.config_dict['num_train_workers']))
        executor.config_dict['logger'].info('{:<20}:{}'.format('num_val_workers', executor.config_dict['num_val_workers']))
        executor.config_dict['logger'].info('{:<20}:{}'.format('num_classes', executor.config_dict['num_classes']))
        executor.config_dict['logger'].info('{:<20}:{}'.format('weight_path', executor.config_dict['weight_path']))
        executor.config_dict['logger'].info('{:<20}:{}'.format('resume_path', executor.config_dict['resume_path']))
        executor.config_dict['logger'].info('{:<20}:{}'.format('learning_rate', executor.config_dict['learning_rate']))
        executor.config_dict['logger'].info('{:<20}:{}'.format('weight_decay', executor.config_dict['weight_decay']))
        executor.config_dict['logger'].info('{:<20}:{}'.format('warmup_setting', executor.config_dict['warmup_setting']))
        executor.config_dict['logger'].info('-----------------------------------------------')

    def after_run(self, executor):
        executor.config_dict['logger'].info('Training finishes.')

    def before_train_epoch(self, executor):
        executor.config_dict['logger'].info('Train Epoch[{}] starts......'.format(executor.config_dict['epoch'] + 1))

    def before_val_epoch(self, executor):
        executor.config_dict['logger'].info('Val Epoch starts......')

    def after_train_iter(self, executor):
        if (executor.config_dict['inner_train_iter'] + 1) % executor.config_dict['display_interval'] == 0 or (executor.config_dict['inner_train_iter'] + 1) == len(executor.config_dict['train_data_loader']):
            executor.config_dict['logger'].info(self._get_log_info(executor))
            executor.config_dict['train_average_meter'].clear()

    def after_val_iter(self, executor):
        if (executor.config_dict['inner_val_iter'] + 1) % executor.config_dict['display_interval'] == 0 or (executor.config_dict['inner_val_iter'] + 1) == len(executor.config_dict['val_data_loader']):
            executor.config_dict['logger'].info(self._get_log_info(executor))
            executor.config_dict['val_average_meter'].clear()

    # def after_train_epoch(self, executor):
    #     executor.config_dict['train_average_meter'].clear()

    def after_val_epoch(self, executor):
        # first print evaluation metrics
        executor.config_dict['logger'].info(executor.config_dict['evaluator'].get_eval_display_str())
        # executor.config_dict['val_average_meter'].clear()
