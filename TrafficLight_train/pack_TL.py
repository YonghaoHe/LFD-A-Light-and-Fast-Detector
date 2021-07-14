# -*- coding: utf-8 -*-
from lfd.data_pipeline.dataset import Dataset
from lfd.data_pipeline.dataset.coco_parser import COCOParser

train_annotation_file_path = './debug_data/annotations/instances_train2017.json'
train_image_root = './debug_data/images/train2017'
train_pkl_save_path = './debug_data/train.pkl'

train_parser = COCOParser(
    coco_annotation_path=train_annotation_file_path,
    image_root=train_image_root,
    filter_no_gt=False,
    filter_min_size=32
)

train_dataset = Dataset(
    parser=train_parser,
    save_path=train_pkl_save_path
)

val_annotation_file_path = './debug_data/annotations/instances_val2017.json'
val_image_root = './debug_data/images/val2017'
val_pkl_save_path = './debug_data/val.pkl'

val_parser = COCOParser(
    coco_annotation_path=val_annotation_file_path,
    image_root=val_image_root,
    filter_no_gt=False,
    filter_min_size=32
)

val_dataset = Dataset(
    parser=val_parser,
    save_path=val_pkl_save_path
)

