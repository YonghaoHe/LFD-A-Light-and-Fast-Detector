# -*- coding: utf-8 -*-
# author: Yonghao He
# description:

import os


def SIO_evaluation(model,
                   val_image_root,
                   results_save_root='.',
                   classification_threshold=0.5,
                   nms_threshold=0.3,
                   ):
    assert os.path.exists(val_image_root)

    if not os.path.exists(results_save_root):
        os.makedirs(results_save_root)

    for parent, dir_names, file_names in os.walk(val_image_root):
        for file_name in file_names:
            if not file_name.lower().endswith(('.jpg', '.jpeg')):
                continue

            results = model.predict_for_single_image(
                os.path.join(parent, file_name),
            )
