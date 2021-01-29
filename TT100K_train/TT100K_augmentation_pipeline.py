# -*- coding: utf-8 -*-
from albumentations import *
from lfd.data_pipeline.augmentation import simple_normalize, bbox_param

__all__ = ['train_pipeline',
           'val_pipeline']

# CAUTION: do not perform HorizontalFlip aug, because traffic signs are not centro-symmetry

train_pipeline_with_bboxes = Compose([
    simple_normalize],
    bbox_params=bbox_param,
    p=1.)
train_pipeline_without_bboxes = Compose([
    simple_normalize],
    p=1.)

val_pipeline_with_bboxes = Compose([
    simple_normalize],
    bbox_params=bbox_param,
    p=1.)
val_pipeline_without_bboxes = Compose([
    simple_normalize],
    p=1.)


def train_pipeline(sample):
    if 'bboxes' in sample:
        return train_pipeline_with_bboxes(**sample)
    else:
        return train_pipeline_without_bboxes(**sample)


def val_pipeline(sample):
    if 'bboxes' in sample:
        return val_pipeline_with_bboxes(**sample)
    else:
        return val_pipeline_without_bboxes(**sample)
