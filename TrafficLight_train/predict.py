# -*- coding: utf-8 -*-

from lfd.execution.utils import load_checkpoint
import cv2
import os

from TL_augmentation_pipeline import *
from TL_LFD_L_work_dir_20210714_173824.TL_LFD_L import config_dict, prepare_model

prepare_model()

param_file_path = './TL_LFD_L_work_dir_20210714_173824/epoch_50.pth'

load_checkpoint(config_dict['model'], load_path=param_file_path, strict=True)

image_path = '/home/yonghaohe/projects/LFD-A-Light-And-Fast-Detector/local_TrafficLight_train/debug_data/images/val2017/60b5015b03919900e418c18a.jpg'
image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

results = config_dict['model'].predict_for_single_image(image, aug_pipeline=val_pipeline, classification_threshold=0.5, nms_threshold=0.3, class_agnostic=True)


for bbox in results:
    print(bbox)
    cv2.rectangle(image, (int(bbox[2]), int(bbox[3])), (int(bbox[2] + bbox[4]), int(bbox[3] + bbox[5])), (0, 255, 0), 2)
print('%d lights are detected!' % len(results))
cv2.imshow('im', image)
cv2.waitKey()

cv2.imwrite(os.path.join(os.path.dirname(image_path), os.path.basename(image_path).replace('.jpg', '_result.jpg')), image)
