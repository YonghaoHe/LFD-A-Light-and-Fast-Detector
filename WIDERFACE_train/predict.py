# -*- coding: utf-8 -*-

from lfd.execution.utils import load_checkpoint
from lfd.data_pipeline.augmentation import *
import cv2

# set the target model script
from WIDERFACE_LFD_M_work_dir_20210116_153509.WIDERFACE_LFD_M import config_dict, prepare_model


prepare_model()

# set the model weight file
param_file_path = './WIDERFACE_LFD_M_work_dir_20210116_153509/epoch_1000.pth'

load_checkpoint(config_dict['model'], load_path=param_file_path, strict=True)

# set the image path to be tested
image_path = './test_images/image1.jpg'
image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

results = config_dict['model'].predict_for_single_image(image, aug_pipeline=simple_widerface_val_pipeline, classification_threshold=0.5, nms_threshold=0.3)


for bbox in results:
    print(bbox)
    cv2.rectangle(image, (int(bbox[2]), int(bbox[3])), (int(bbox[2] + bbox[4]), int(bbox[3] + bbox[5])), (0, 255, 0), 1)
print('%d faces are detected!' % len(results))
cv2.imshow('im', image)
cv2.waitKey()
