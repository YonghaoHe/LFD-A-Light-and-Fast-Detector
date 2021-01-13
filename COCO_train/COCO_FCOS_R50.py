# -*- coding: utf-8 -*-

import sys

sys.path.append('../..')
import shutil
import os
import time
import torch
from lfd.execution.utils import set_random_seed, set_cudnn_backend
from lfd.model.backbone import ResNet
from lfd.model.neck import FPN
from lfd.model.head import FCOSHead
from lfd.model import FCOS
from lfd.model.losses import *
from lfd.data_pipeline.data_loader import DataLoader
from lfd.data_pipeline.dataset import Dataset
from lfd.data_pipeline.sampler import *
from lfd.data_pipeline.augmentation import *
from lfd.evaluation import COCOEvaluator
from lfd.execution.executor import Executor
from lfd.execution.utils import customize_exception_hook

assert torch.cuda.is_available(), 'GPU training supported only!'

memo = 'COCO R50'

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
    config_dict['training_epochs'] = 12

    # reproductive
    config_dict['seed'] = 666
    config_dict['cudnn_benchmark'] = True
    if config_dict['seed'] is not None:
        set_random_seed(config_dict['seed'])
    set_cudnn_backend(config_dict['cudnn_benchmark'])

    # GPU list
    config_dict['gpu_list'] = [0]
    assert isinstance(config_dict['gpu_list'], list)

    # display interval in iterations
    config_dict['display_interval'] = 1

    # checkpoint save interval in epochs
    config_dict['save_interval'] = 12

    # validation interval in epochs
    config_dict['val_interval'] = 1


'''
build model ----------------------------------------------------------------------------------------------
'''


def prepare_model():
    # input image channels: BGR--3, gray--1
    config_dict['num_input_channels'] = 3

    classification_loss = FocalLoss(
        use_sigmoid=True,
        gamma=2.0,
        alpha=0.25,
        reduction='mean',
        loss_weight=1.0
    )
    regression_loss = IoULoss(
        eps=1e-6,
        reduction='mean',
        loss_weight=1.0
    )
    centerness_loss = BCEWithLogitsLoss(
        reduction='mean',
        loss_weight=1.0
    )

    # number of classes
    config_dict['num_classes'] = 80
    config_dict['backbone_init_param_file_path'] = os.path.join(os.path.dirname(__file__), '../lfd/model/backbone/pretrained_backbone_weights', 'resnet50_caffe.pth')  # if no pretrained weights, set to None
    fcos_backbone = ResNet(
        depth=50,
        in_channels=config_dict['num_input_channels'],
        base_channels=64,
        out_indices=((2, 3), (3, 5), (4, 2)),
        style='caffe',
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,  # 注意这个参数
        dcn=None,
        stage_with_dcn=(False, False, False, False),
        zero_init_residual=True,
        init_with_weight_file=config_dict['backbone_init_param_file_path']
    )

    fcos_neck = FPN(
        num_input_channels_list=fcos_backbone.num_output_channels_list,
        num_input_strides_list=fcos_backbone.num_output_strides_list,
        num_output_channels=256,
        num_outputs=5,
        extra_on_input=False,
        extra_type='conv',
        norm_on_lateral=False,  # consistent with mmdet
        relu_on_lateral=False,  # consistent with mmdet
        relu_before_extra=True,
        norm_cfg=None,
    )

    fcos_head = FCOSHead(
        num_classes=80,
        num_input_channels=256,
        num_head_channels=256,
        num_layers=4,
        norm_cfg=None
    )
    config_dict['detection_ranges'] = ((0, 64), (64, 128), (128, 256), (256, 512), (512, 10000))
    config_dict['bias_lr_cfg'] = dict(bias_lr=2., bias_weight_decay=0.)
    config_dict['model'] = FCOS(
        backbone=fcos_backbone,
        neck=fcos_neck,
        head=fcos_head,
        num_classes=config_dict['num_classes'],
        regress_ranges=config_dict['detection_ranges'],
        point_strides=fcos_neck.num_output_strides_list,
        classification_loss_func=classification_loss,
        regression_loss_func=regression_loss,
        centerness_loss_func=centerness_loss,
        classification_threshold=0.05,
        nms_threshold=0.5,
        pre_nms_bbox_limit=1000,
        post_nms_bbox_limit=100,
        param_groups_cfg=config_dict['bias_lr_cfg']
    )

    # init param weights file
    # when set, the executor will init the whole net using this file
    config_dict['weight_path'] = None

    # resume training path
    # when set, the 'weight_path' will be ignored. The executor will init the whole net and training parameters using this file
    config_dict['resume_path'] = None


'''
prepare data loader -----------------------------------------------------------------------------------------
'''


def prepare_data_pipeline():
    # batch size
    config_dict['batch_size'] = 4

    # number of train data_loader workers
    config_dict['num_train_workers'] = 4

    # number of val data_loader workers
    config_dict['num_val_workers'] = 4

    # construct train data_loader
    config_dict['train_dataset_path'] = './COCO_pack/coco_train2017.pkl'
    train_dataset = Dataset(load_path=config_dict['train_dataset_path'])

    train_dataset_sampler = COCORandomDatasetSampler(dataset=train_dataset,
                                                     batch_size=config_dict['batch_size'],
                                                     shuffle=True, )

    train_region_sampler = TypicalCOCOTrainingRegionSampler(resize_shorter_range=(800,), resize_longer_limit=1333, pad_divisor=32)

    config_dict['train_data_loader'] = DataLoader(dataset=train_dataset,
                                                  dataset_sampler=train_dataset_sampler,
                                                  region_sampler=train_region_sampler,
                                                  augmentation_pipeline=typical_coco_train_pipeline,
                                                  num_workers=config_dict['num_train_workers'])

    # construct val data_loader
    config_dict['val_dataset_path'] = './COCO_pack/coco_val2017.pkl'
    val_dataset = Dataset(load_path=config_dict['val_dataset_path'])
    val_dataset_sampler = RandomDatasetSampler(dataset=val_dataset,
                                               batch_size=config_dict['batch_size'],
                                               shuffle=False,
                                               ignore_last=False)
    val_region_sampler = TypicalCOCOTrainingRegionSampler(resize_shorter_range=(800,), resize_longer_limit=1333, pad_divisor=32)
    config_dict['val_data_loader'] = DataLoader(dataset=val_dataset,
                                                dataset_sampler=val_dataset_sampler,
                                                region_sampler=val_region_sampler,
                                                augmentation_pipeline=typical_coco_val_pipeline,
                                                num_workers=config_dict['num_val_workers'])

    # evaluator
    # the evaluator should match the dataset
    config_dict['val_annotation_path'] = '/home/yonghaohe/datasets/COCO/annotations/instances_val2017.json'
    config_dict['evaluator'] = COCOEvaluator(annotation_path=config_dict['val_annotation_path'],
                                             label_indexes_to_category_ids=val_dataset.meta_info['label_indexes_to_category_ids'])


'''
learning rate and optimizer --------------------------------------------------------------------------------
optimizer and scheduler can be customized
'''


def prepare_optimizer():
    config_dict['learning_rate'] = 0.01
    config_dict['momentum'] = 0.9
    config_dict['weight_decay'] = 0.0001
    config_dict['optimizer'] = torch.optim.SGD(params=config_dict['model'].parameters(),
                                               lr=config_dict['learning_rate'],
                                               momentum=config_dict['momentum'],
                                               weight_decay=config_dict['weight_decay'])

    config_dict['optimizer_grad_clip_cfg'] = dict(max_norm=35, norm_type=2)

    # multi step lr scheduler is used here
    config_dict['milestones'] = [8, 11]
    config_dict['gamma'] = 0.1
    assert max(config_dict['milestones']) < config_dict['training_epochs'], 'the max value in milestones should be less than total epochs!'

    config_dict['lr_scheduler'] = torch.optim.lr_scheduler.MultiStepLR(config_dict['optimizer'],
                                                                       milestones=config_dict['milestones'],
                                                                       gamma=config_dict['gamma'])  # scheduler 也需要被保存在checkpoint中

    # add warmup parameters
    config_dict['warmup_setting'] = dict(by_epoch=False,
                                         warmup_mode='constant',  # if no warmup needed, set warmup_mode = None
                                         warmup_loops=500,
                                         warmup_ratio=1.0 / 3)

    assert isinstance(config_dict['warmup_setting'], dict) and 'by_epoch' in config_dict['warmup_setting'] and 'warmup_mode' in config_dict['warmup_setting'] \
           and 'warmup_loops' in config_dict['warmup_setting'] and 'warmup_ratio' in config_dict['warmup_setting']


if __name__ == '__main__':
    prepare_common_settings()

    prepare_model()

    prepare_data_pipeline()

    prepare_optimizer()

    training_executor = Executor(config_dict)

    training_executor.run()
