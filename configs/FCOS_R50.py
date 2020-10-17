# -*- coding: utf-8 -*-

import os
import time
import torch
from execution.utils import set_random_seed, set_cudnn_backend
from model.backbone import ResNet
from model.neck import FPN
from model.head import FCOSHead
from model import FCOS
from model.losses import FocalLoss, IoULoss, CrossEntropyLoss
from data_pipeline.data_loader import DataLoader
from data_pipeline.dataset import Dataset
from data_pipeline.sampler import *
from data_pipeline.augmentation import *
from evaluation import COCOEvaluator

assert torch.cuda.is_available(), 'GPU training supported only!'

# all config parameters will be stored in config_dict
# all keys in param_dict are reserved, do not change key names casually
config_dict = dict()

# work directory (saving log and model weights)
config_dict['timestamp'] = time.strftime('%Y%m%d_%H%M%S', time.localtime())
config_dict['work_dir'] = './' + os.path.basename(__file__).split('.')[0] + '_work_dir_' + config_dict['timestamp']

# log file path
config_dict['log_path'] = os.path.join(config_dict['work_dir'], os.path.basename(__file__).split('.')[0] + '_' + config_dict['timestamp'] + '.log')

# training epochs
config_dict['training_epochs'] = 12

# batch size
config_dict['batch_size'] = 2

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
config_dict['display_interval'] = 2

# checkpoint save interval in epochs
config_dict['save_interval'] = 10

# validation interval in epochs
config_dict['val_interval'] = 1

'''
prepare data loader -----------------------------------------------------------------------------------------
'''
# input image channels: BGR--3, gray--1
config_dict['num_input_channels'] = 3

# number of train data_loader workers
config_dict['num_train_workers'] = 4

# number of val data_loader workers
config_dict['num_val_workers'] = 2

# construct train data_loader
train_dataset_path = os.path.join(os.path.dirname(__file__), '..', 'data_pipeline/pack/mini_coco_trainval2017.pkl')
train_dataset = Dataset(load_path=train_dataset_path)
train_dataset_sampler = RandomDatasetSampler(index_annotation_dict=train_dataset.index_annotation_dict,
                                             batch_size=config_dict['batch_size'],
                                             shuffle=True,
                                             ignore_last=False)
train_region_sampler = TypicalCOCOTrainingRegionSampler(output_size=(1333, 1333), resize_shorter_range=(800, ), resize_longer_limit=1333)
config_dict['train_data_loader'] = DataLoader(dataset=train_dataset,
                                              dataset_sampler=train_dataset_sampler,
                                              region_sampler=train_region_sampler,
                                              augmentation_pipeline=typical_coco_train_pipeline,
                                              num_input_channels=config_dict['num_input_channels'],
                                              num_workers=config_dict['num_train_workers'])

# construct val data_loader
val_dataset_path = os.path.join(os.path.dirname(__file__), '..', 'data_pipeline/pack/mini_coco_trainval2017.pkl')
val_dataset = Dataset(load_path=val_dataset_path)
val_dataset_sampler = RandomDatasetSampler(index_annotation_dict=val_dataset.index_annotation_dict,
                                           batch_size=config_dict['batch_size'],
                                           shuffle=False,
                                           ignore_last=False)
val_region_sampler = TypicalCOCOTrainingRegionSampler(output_size=(1333, 1333), resize_shorter_range=(800, ), resize_longer_limit=1333)
config_dict['val_data_loader'] = DataLoader(dataset=val_dataset,
                                            dataset_sampler=val_dataset_sampler,
                                            region_sampler=val_region_sampler,
                                            augmentation_pipeline=typical_coco_val_pipeline,
                                            num_input_channels=config_dict['num_input_channels'],
                                            num_workers=config_dict['num_val_workers'])

'''
build model ----------------------------------------------------------------------------------------------
'''
# number of classes
config_dict['num_classes'] = 80
backbone_init_param_file_path = os.path.join(os.path.dirname(__file__), '..', 'model/backbone/pretrained_backbone_weights', 'resnet50_caffe.pth')  # if no pretrained weights, set to None
fcos_backbone = ResNet(depth=50,
                       in_channels=config_dict['num_input_channels'],
                       base_channels=64,
                       out_indices=((2, 3), (3, 5), (4, 2)),
                       style='caffe',
                       frozen_stages=1,
                       conv_cfg=None,
                       norm_cfg=dict(type='BN', requires_grad=False),
                       norm_eval=True,  # 注意这个参数
                       dcn=None,
                       stage_with_dcn=(False, False, False, False),
                       zero_init_residual=True,
                       init_with_weight_file=backbone_init_param_file_path)

fcos_neck = FPN(num_input_channels_list=fcos_backbone.num_output_channels_list,
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

fcos_head = FCOSHead(num_classes=80,
                     num_input_channels=256,
                     num_head_channels=256,
                     num_layers=4,
                     norm_cfg=None
                     )

classifiation_loss = FocalLoss(use_sigmoid=True,
                               gamma=2.0,
                               alpha=0.25,
                               reduction='mean',
                               loss_weight=1.0)
regression_loss = IoULoss(eps=1e-6, reduction='mean', loss_weight=1.0)
centerness_loss = CrossEntropyLoss(use_sigmoid=True,
                                   use_mask=False,
                                   reduction='mean',
                                   loss_weight=1.0)

config_dict['model'] = FCOS(backbone=fcos_backbone,
                            neck=fcos_neck,
                            head=fcos_head,
                            num_classes=config_dict['num_classes'],
                            regress_ranges=((0, 64), (64, 128), (128, 256), (256, 512), (512, 800)),
                            point_strides=fcos_neck.num_output_strides_list,
                            classification_loss_func=classifiation_loss,
                            regression_loss_func=regression_loss,
                            centerness_loss_func=centerness_loss,
                            classification_threshold=0.05,
                            nms_threshold=0.5
                            )

# init param weights file
# when set, the executor will init the whole net using this file
config_dict['weight_path'] = None

# resume training path
# when set, the 'weight_path' will be ignored. The executor will init the whole net and training parameters using this file
config_dict['resume_path'] = None

# evaluator
# the evaluator should match the dataset
config_dict['evaluator'] = COCOEvaluator(annotation_path=os.path.join(os.path.dirname(__file__), '../datasets/coco/instances_val2017.json'),
                                         label_indexes_to_category_ids=val_dataset.meta_info[1])

'''
learning rate and optimizer --------------------------------------------------------------------------------
optimizer and scheduler can be customized
'''
config_dict['learning_rate'] = 0.01
config_dict['momentum'] = 0.9
config_dict['weight_decay'] = 0.0001
config_dict['optimizer'] = torch.optim.SGD(params=config_dict['model'].parameters(),
                                           lr=config_dict['learning_rate'],
                                           momentum=config_dict['momentum'],
                                           weight_decay=config_dict['weight_decay'])

# multi step lr scheduler is used here
milestones = [8, 11]
assert max(milestones) < config_dict['training_epochs'], 'the max value in milestones should be less than total epochs!'
config_dict['lr_scheduler'] = torch.optim.lr_scheduler.MultiStepLR(config_dict['optimizer'],
                                                                   milestones=milestones,
                                                                   gamma=0.1)  # scheduler 也需要被保存在checkpoint中

# add warmup parameters
config_dict['warmup_setting'] = dict(by_epoch=False,
                                     warmup_mode='linear',  # if no warmup needed, set warmup_mode = None
                                     warmup_loops=1000,
                                     warmup_ratio=0.1)
assert isinstance(config_dict['warmup_setting'], dict) and 'by_epoch' in config_dict['warmup_setting'] and 'warmup_mode' in config_dict['warmup_setting'] \
       and 'warmup_loops' in config_dict['warmup_setting'] and 'warmup_ratio' in config_dict['warmup_setting']
