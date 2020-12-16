# -*- coding: utf-8 -*-
# author: Yonghao He
# description:

import os
import math
from ..data_pipeline.augmentation import *
__all__ = ['SIO_evaluation']


def SIO_evaluation(model,
                   val_image_root,
                   results_save_root='.',
                   classification_threshold=0.5,
                   nms_threshold=0.3,
                   ):
    assert os.path.exists(val_image_root)

    if not os.path.exists(results_save_root):
        os.makedirs(results_save_root)

    counter = 0
    for parent, dir_names, file_names in os.walk(val_image_root):
        for file_name in file_names:
            if not file_name.lower().endswith(('.jpg', '.jpeg')):
                continue

            results = model.predict_for_single_image(
                image=os.path.join(parent, file_name),
                aug_pipeline=simple_widerface_val_pipeline,
                classification_threshold=classification_threshold,
                nms_threshold=nms_threshold
            )

            event_name = parent.split('/')[-1]
            if not os.path.exists(os.path.join(results_save_root, event_name)):
                os.makedirs(os.path.join(results_save_root, event_name))
            fout = open(os.path.join(results_save_root, event_name, file_name.split('.')[0] + '.txt'), 'w')
            fout.write(file_name.split('.')[0] + '\n')
            fout.write(str(len(results) + 1) + '\n')
            fout.write('0 0 0 0 0.001\n')
            for bbox in results:
                fout.write('%d %d %d %d %.03f' % (math.floor(bbox[2]), math.floor(bbox[3]), math.ceil(bbox[4]), math.ceil(bbox[5]), bbox[1] if bbox[1] <= 1 else 1) + '\n')
            fout.close()

            counter += 1
            print('[%5d] %s is processed.' % (counter, file_name))
