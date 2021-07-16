# -*- coding: utf-8 -*-

# perform COCO style MAP evaluation
from lfd.execution.utils import load_checkpoint
from lfd.data_pipeline.dataset import Dataset
from lfd.evaluation import COCOEvaluator
from pycocotools.coco import COCO
import cv2
import os

from TL_augmentation_pipeline import *
from TL_LFD_L_work_dir_20210714_173824.TL_LFD_L import config_dict, prepare_model

prepare_model()
param_file_path = './TL_LFD_L_work_dir_20210714_173824/epoch_100.pth'
load_checkpoint(config_dict['model'], load_path=param_file_path, strict=True)
classification_threshold = 0.1
nms_threshold = 0.3


val_annotation_path = './debug_data/annotations/instances_train2017.json'
val_image_root = './debug_data/images/train2017'
val_dataset_pkl = './debug_data/train.pkl'

val_dataset = Dataset(load_path=val_dataset_pkl)
label_indexes_to_category_ids = val_dataset.meta_info['label_indexes_to_category_ids']

coco_evaluator = COCOEvaluator(annotation_path=val_annotation_path,
                               label_indexes_to_category_ids=label_indexes_to_category_ids)


coco = COCO(annotation_file=val_annotation_path)

image_ids = coco.getImgIds()

for i, image_id in enumerate(image_ids):
    image_info = coco.loadImgs(image_id)

    image_path = os.path.join(val_image_root, image_info[0]['file_name'])

    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    results = config_dict['model'].predict_for_single_image(image,
                                                            aug_pipeline=val_pipeline,
                                                            classification_threshold=classification_threshold,
                                                            nms_threshold=nms_threshold,
                                                            class_agnostic=True)
    meta_info = [{'image_id': image_id}]
    pred_bboxes = [results]

    coco_evaluator.update((pred_bboxes, meta_info))

    print('Predicting: %d/%d' % (i, len(image_ids)))

coco_evaluator.evaluate()
print(coco_evaluator.get_eval_display_str())
