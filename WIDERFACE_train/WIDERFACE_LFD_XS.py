# -*- coding: utf-8 -*-
# author: Yonghao He
# description:
import sys
sys.path.append('../..')
import shutil
import os
import time
import torch
from lfd.execution.utils import set_random_seed, set_cudnn_backend
from lfd.model.backbone import LFDResNet
from lfd.model.neck import SimpleNeck
from lfd.model.head import LFDHead
from lfd.model.losses import *
from lfd.model import *
from lfd.data_pipeline.data_loader import DataLoader
from lfd.data_pipeline.dataset import Dataset
from lfd.data_pipeline.sampler import *
from lfd.data_pipeline.augmentation import *
from lfd.execution.executor import Executor
from lfd.execution.utils import customize_exception_hook

assert torch.cuda.is_available(), 'GPU training supported only!'

memo = 'WIDERFACE XS 模型' \
       'head不共享, 进行path merge, 使用BN' \
       '采用CE作为分类loss, loss weight设置为1.' \
       '采用IoULoss作为回归loss，distance_to_bbox_mode采用exp, loss weight设置为0.1' \


# all config parameters will be stored in config_dict
config_dict = dict()


def prepare_common_settings():
    # work directory (saving log and model weights)
    config_dict['timestamp'] = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    config_dict['work_dir'] = './' + os.path.basename(__file__).split('.')[0] + '_work_dir_' + config_dict['timestamp']

    # copy current config file to work dir for backup
    if not os.path.exists(config_dict['work_dir']):
        os.makedirs(config_dict['work_dir'])
    shutil.copyfile(__file__, os.path.join(config_dict['work_dir'], os.path.basename(__file__)))

    # log file path
    config_dict['log_path'] = os.path.join(config_dict['work_dir'], 'log_' + config_dict['timestamp'] + '.log')

    # set exception hook, record all output info including exceptions and errors
    sys.excepthook = customize_exception_hook(os.path.join(config_dict['work_dir'], 'exception_log_' + config_dict['timestamp'] + '.log'))

    # training epochs
    config_dict['training_epochs'] = 300

    # reproductive
    config_dict['seed'] = 10
    config_dict['cudnn_benchmark'] = True
    if config_dict['seed'] is not None:
        set_random_seed(config_dict['seed'])
    set_cudnn_backend(config_dict['cudnn_benchmark'])

    # GPU list
    config_dict['gpu_list'] = [0]
    assert isinstance(config_dict['gpu_list'], list)

    # display interval in iterations
    config_dict['display_interval'] = 2

    # checkpoint save interval in epochs
    config_dict['save_interval'] = 50

    # validation interval in epochs
    config_dict['val_interval'] = 0


'''
prepare data loader -----------------------------------------------------------------------------------------
'''


def prepare_data_pipeline():
    # batch size
    config_dict['batch_size'] = 8

    # number of train data_loader workers
    config_dict['num_train_workers'] = 6

    # number of val data_loader workers
    config_dict['num_val_workers'] = 0

    # construct train data_loader
    config_dict['train_dataset_path'] = '../code_test/WIDERFACE_pack/widerface_train.pkl'
    train_dataset = Dataset(load_path=config_dict['train_dataset_path'])
    train_dataset_sampler = RandomWithNegDatasetSampler(
        train_dataset,
        batch_size=config_dict['batch_size'],
        neg_ratio=0.2,
        shuffle=True,
        ignore_last=False
    )
    train_region_sampler = RandomBBoxCropRegionSampler(crop_size=480, resize_range=(0.5, 2))
    config_dict['train_data_loader'] = DataLoader(dataset=train_dataset,
                                                  dataset_sampler=train_dataset_sampler,
                                                  region_sampler=train_region_sampler,
                                                  augmentation_pipeline=simple_widerface_train_pipeline,
                                                  num_workers=config_dict['num_train_workers'])

    # construct val data_loader
    # config_dict['val_dataset_path'] = 'xxxxxxxxxx'
    # val_dataset = Dataset(load_path=config_dict['val_dataset_path'])
    # val_dataset_sampler = RandomDatasetSampler(dataset=val_dataset,
    #                                            batch_size=config_dict['batch_size'],
    #                                            shuffle=False,
    #                                            ignore_last=False)
    # val_region_sampler = IdleRegionSampler()
    # config_dict['val_data_loader'] = DataLoader(dataset=val_dataset,
    #                                             dataset_sampler=val_dataset_sampler,
    #                                             region_sampler=val_region_sampler,
    #                                             augmentation_pipeline=simple_widerface_val_pipeline,
    #                                             num_workers=config_dict['num_val_workers'])


'''
build model ----------------------------------------------------------------------------------------------
'''


def prepare_model():
    # input image channels: BGR--3, gray--1
    config_dict['num_input_channels'] = 3

    # loss functions
    # classification_loss = FocalLoss(use_sigmoid=True,
    #                                gamma=2.0,
    #                                alpha=0.25,
    #                                reduction='mean',
    #                                loss_weight=1.0)
    classification_loss = CrossEntropyLoss(
        reduction='mean',
        loss_weight=1.0
    )
    # classification_loss = BCEWithLogitsLoss(
    #     reduction='mean',
    #     loss_weight=1.0
    # )

    # regression_loss = SmoothL1Loss(
    #     beta=1.0,
    #     reduction='mean',
    #     loss_weight=1.0
    # )
    regression_loss = IoULoss(
        eps=1e-6,
        reduction='mean',
        loss_weight=0.1
    )

    # number of classes
    config_dict['num_classes'] = 1
    config_dict['backbone_init_param_file_path'] = None  # if no pretrained weights, set to None
    lfd_backbone = LFDResNet(
        block_mode='faster',  # affect block type
        stem_mode='faster',  # affect stem type
        body_mode=None,  # affect body architecture
        input_channels=config_dict['num_input_channels'],
        stem_channels=32,
        body_architecture=[2, 1, 1, 2],
        body_channels=[64, 64, 64, 64],
        out_indices=((0, 1), (1, 0), (2, 0), (3, 0), (3, 1)),
        frozen_stages=-1,
        activation_cfg=dict(type='ReLU', inplace=True),
        norm_cfg=dict(type='BatchNorm2d'),
        init_with_weight_file=config_dict['backbone_init_param_file_path'],
        norm_eval=False
    )

    lfd_neck = SimpleNeck(
        num_neck_channels=128,
        num_input_channels_list=lfd_backbone.num_output_channels_list,
        num_input_strides_list=lfd_backbone.num_output_strides_list,
        norm_cfg=dict(type='BatchNorm2d'),
        activation_cfg=dict(type='ReLU', inplace=True)
    )

    lfd_head = LFDHead(
        num_classes=config_dict['num_classes'],
        num_heads=len(lfd_neck.num_output_strides_list),
        num_input_channels=128,
        num_head_channels=128,
        num_conv_layers=2,
        activation_cfg=dict(type='ReLU', inplace=True),
        norm_cfg=dict(type='BatchNorm2d'),
        share_head_flag=False,
        merge_path_flag=True,
        classification_loss_type=type(classification_loss).__name__,
        regression_loss_type=type(regression_loss).__name__
    )

    config_dict['model'] = LFD(
        backbone=lfd_backbone,
        neck=lfd_neck,
        head=lfd_head,
        num_classes=config_dict['num_classes'],
        regression_ranges=((2, 20), (20, 40), (40, 80), (80, 160), (160, 320)),
        gray_range_factors=(0.9, 1.1),
        point_strides=lfd_neck.num_output_strides_list,
        classification_loss_func=classification_loss,
        regression_loss_func=regression_loss,
        distance_to_bbox_mode='exp',
        classification_threshold=0.05,
        nms_threshold=0.5,
        pre_nms_bbox_limit=1000,
        post_nms_bbox_limit=100,
    )

    # init param weights file
    # when set, the executor will init the whole net using this file
    config_dict['weight_path'] = None

    # resume training path
    # when set, the 'weight_path' will be ignored. The executor will init the whole net and training parameters using this file
    config_dict['resume_path'] = None

    # evaluator
    # the evaluator should match the dataset
    config_dict['evaluator'] = None


'''
learning rate and optimizer --------------------------------------------------------------------------------
optimizer and scheduler can be customized
'''


def prepare_optimizer():
    config_dict['learning_rate'] = 0.1
    config_dict['momentum'] = 0.9
    config_dict['weight_decay'] = 0.0001
    config_dict['optimizer'] = torch.optim.SGD(params=config_dict['model'].parameters(),
                                               lr=config_dict['learning_rate'],
                                               momentum=config_dict['momentum'],
                                               weight_decay=config_dict['weight_decay'])

    # config_dict['optimizer'] = torch.optim.Adam(params=config_dict['model'].parameters(),
    #                                             lr=config_dict['learning_rate'],
    #                                             weight_decay=config_dict['weight_decay'])

    config_dict['optimizer_grad_clip_cfg'] = None  # dict(max_norm=10, norm_type=2)

    # multi step lr scheduler is used here
    config_dict['milestones'] = [150, 250]
    config_dict['gamma'] = 0.1
    assert max(config_dict['milestones']) < config_dict['training_epochs'], 'the max value in milestones should be less than total epochs!'

    config_dict['lr_scheduler'] = torch.optim.lr_scheduler.MultiStepLR(config_dict['optimizer'],
                                                                       milestones=config_dict['milestones'],
                                                                       gamma=config_dict['gamma'])  # scheduler 也需要被保存在checkpoint中

    # add warmup parameters
    config_dict['warmup_setting'] = dict(by_epoch=False,
                                         warmup_mode='linear',  # if no warmup needed, set warmup_mode = None
                                         warmup_loops=100,
                                         warmup_ratio=0.1)

    assert isinstance(config_dict['warmup_setting'], dict) and 'by_epoch' in config_dict['warmup_setting'] and 'warmup_mode' in config_dict['warmup_setting'] \
           and 'warmup_loops' in config_dict['warmup_setting'] and 'warmup_ratio' in config_dict['warmup_setting']


if __name__ == '__main__':
    prepare_common_settings()

    prepare_data_pipeline()

    prepare_model()

    prepare_optimizer()

    training_executor = Executor(config_dict)

    training_executor.run()
