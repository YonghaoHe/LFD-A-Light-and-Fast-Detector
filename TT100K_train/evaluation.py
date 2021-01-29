# -*- coding: utf-8 -*-
import cv2
import json
import os
from lfd.execution.utils import load_checkpoint
from lfd.data_pipeline.augmentation import *
from lfd.data_pipeline.dataset import Dataset
import official_eval


def evaluate():
    #  set model to be evaluated ----------------------------------------------------------
    from TT100K_LFD_S_work_dir_20210127_170801.TT100K_LFD_S import config_dict, prepare_model
    weight_file_path = './TT100K_LFD_S_work_dir_20210127_170801/epoch_500.pth'
    classification_threshold = 0.1
    nms_threshold = 0.1

    prepare_model()
    load_checkpoint(config_dict['model'], load_path=weight_file_path, strict=True)

    # predict results and save to json
    results_json = dict()
    results_json['imgs'] = dict()
    test_image_root = '/home/yonghaohe/datasets/TT100K/data/test'
    test_image_paths_list = [os.path.join(test_image_root, file_name) for file_name in os.listdir(test_image_root) if file_name.endswith('.jpg')]

    dataset_path = './TT100K_pack/train.pkl'
    dataset = Dataset(load_path=dataset_path)
    label_indexes_to_category_names = dataset.meta_info['label_indexes_to_category_names']

    results_json_save_path = os.path.join('./TT100K_evaluation/', os.path.dirname(weight_file_path).split('/')[-1] + '_results.json')
    if not os.path.exists(results_json_save_path):

        for i, image_path in enumerate(test_image_paths_list):
            image_id = os.path.basename(image_path).split('.')[0]
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            results = config_dict['model'].predict_for_single_image(image,
                                                                    aug_pipeline=simple_widerface_val_pipeline,
                                                                    classification_threshold=classification_threshold,
                                                                    nms_threshold=nms_threshold,
                                                                    class_agnostic=True)
            temp = dict(id=image_id, objects=list())
            for result in results:
                cat = label_indexes_to_category_names[result[0]]
                score = result[1] * 100  # make score in [0, 100]
                xmin = result[2]
                ymin = result[3]
                xmax = result[4] + result[2]
                ymax = result[5] + result[3]
                temp_bbox = dict(
                    bbox={'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax},
                    category=cat,
                    score=score
                )
                temp['objects'].append(temp_bbox)

            results_json['imgs'][image_id] = temp
            print('[%5d] image is predicted.' % i)

        if not os.path.exists(os.path.dirname(results_json_save_path)):
            os.makedirs(os.path.dirname(results_json_save_path))
        json.dump(results_json, open(results_json_save_path, 'w'), indent=4, ensure_ascii=False)

    # evaluate
    gt_annotation_json_path = '/home/yonghaohe/datasets/TT100K/data/annotations.json'
    gt_json = json.load(open(gt_annotation_json_path, 'r'))

    results_json = json.load(open(results_json_save_path, 'r'))

    eval_result = official_eval.eval_annos(annos_gd=gt_json,
                                           annos_rt=results_json,
                                           iou=0.5,
                                           imgids=None,
                                           check_type=True,
                                           types=official_eval.type45,
                                           minscore=90,
                                           minboxsize=0,
                                           maxboxsize=400,
                                           match_same=True)
    print(eval_result['report'])


if __name__ == '__main__':
    evaluate()
