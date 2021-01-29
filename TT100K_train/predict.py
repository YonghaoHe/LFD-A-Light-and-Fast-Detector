# -*- coding: utf-8 -*-

from lfd.execution.utils import load_checkpoint
from lfd.data_pipeline.augmentation import *
from lfd.data_pipeline.dataset import Dataset
import cv2

from TT100K_LFD_S_work_dir_20210127_170801.TT100K_LFD_S import config_dict, prepare_model


prepare_model()

param_file_path = './TT100K_LFD_S_work_dir_20210127_170801/epoch_500.pth'

load_checkpoint(config_dict['model'], load_path=param_file_path, strict=True)

dataset_path = './TT100K_pack/train.pkl'
dataset = Dataset(load_path=dataset_path)
label_indexes_to_category_names = dataset.meta_info['label_indexes_to_category_names']

image_path = '/home/yonghaohe/datasets/TT100K/data/test/66449.jpg'
image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)


results = config_dict['model'].predict_for_single_image(image, aug_pipeline=simple_widerface_val_pipeline, classification_threshold=0.5, nms_threshold=0.1, class_agnostic=True)
for bbox in results:
    print(bbox)
    category_name = label_indexes_to_category_names[bbox[0]]
    cv2.rectangle(image, (int(bbox[2]), int(bbox[3])), (int(bbox[2] + bbox[4]), int(bbox[3] + bbox[5])), (0, 255, 0), 2)
    cv2.putText(image, category_name, (int(bbox[2] - 0), int(bbox[3] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

print('%d traffic signs detected!' % len(results))
cv2.imshow('im', image)
cv2.waitKey()
