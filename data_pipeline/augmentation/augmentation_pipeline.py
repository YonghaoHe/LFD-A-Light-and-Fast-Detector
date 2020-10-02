# -*- coding: utf-8 -*-

"""
作者: 何泳澔
日期: 2020-05-20
模块文件: augmentation_pipeline.py
模块描述: 
"""
from albumentations import *

__all__ = ['typical_coco_train_pipeline', 'typical_coco_val_pipeline']

random_horizon_flip = HorizontalFlip(p=0.5)

# CAUTION: normalize may vary along with different pretrained backbones
normalize = Normalize(
    mean=(102.9801, 115.9465, 122.7717),
    std=(1.0, 1.0, 1.0),
    max_pixel_value=1.0,
    p=1.0
)

bbox_param = BboxParams(format='coco', label_fields=['bbox_labels'])  # x,y,w,h
train_pipeline_with_bboxes = Compose([random_horizon_flip,
                                      normalize],
                                     bbox_params=bbox_param,
                                     p=1.)
train_pipeline_without_bboxes = Compose([random_horizon_flip,
                                         normalize],
                                        p=1.)

val_pipeline_with_bboxes = Compose([normalize],
                                   bbox_params=bbox_param,
                                   p=1.)
val_pipeline_without_bboxes = Compose([normalize],
                                      p=1.)


def typical_coco_train_pipeline(sample):
    if 'bboxes' in sample:
        return train_pipeline_with_bboxes(**sample)
    else:
        return train_pipeline_without_bboxes(**sample)


def typical_coco_val_pipeline(sample):
    if 'bboxes' in sample:
        return val_pipeline_with_bboxes(**sample)
    else:
        return val_pipeline_without_bboxes(**sample)
