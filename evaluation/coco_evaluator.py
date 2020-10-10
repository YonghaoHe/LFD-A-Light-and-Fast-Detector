# -*- coding: utf-8 -*-
# author: Yonghao He
# description: evaluator for coco dataset

import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from .base_evaluator import Evaluator

__all__ = ['COCOEvaluator']


class COCOEvaluator(Evaluator):

    def __init__(self,
                 annotation_path,
                 label_indexes_to_category_ids):
        """

        :param annotation_path: the path of annotation json file, typically xxxx/instances_val2017.json
        :param label_indexes_to_category_ids: dict for converting label index to category id
        """
        assert os.path.isfile(annotation_path), 'annotation file does not exist!!!'
        assert isinstance(label_indexes_to_category_ids, dict), 'label index to category id must be a dict!!!'
        self._coco_gt = COCO(annotation_path)
        self._label_indexes_to_category_ids = label_indexes_to_category_ids
        self._detection_results = list()
        self._eval_display_str = ''

    def update(self, results):
        """

        :param results: tuple(predict_bboxes, meta_batch)
        predict_bboxes is a list, in which each element is a prediction for each image
        :return:
        """
        assert isinstance(results, tuple) and len(results) == 2, 'update info should contain two parts: predict bboxes and meta info.'
        predict_bboxes, meta_batch = results

        # reformat to satisfy COCO requirement
        for i in range(len(meta_batch)):
            meta_single = meta_batch[i]
            image_id = meta_single['image_id']
            resize_scale = meta_single['resize_scale']

            predict_bboxes_single = predict_bboxes[i]
            for j in range(len(predict_bboxes_single)):
                predict_item = dict()
                predict_item['image_id'] = image_id
                predict_item['bbox'] = [max(0, value / resize_scale) for value in predict_bboxes_single[j][2:]]
                predict_item['score'] = predict_bboxes_single[j][1]
                predict_item['category_id'] = self._label_indexes_to_category_ids[predict_bboxes_single[j][0]]
                self._detection_results.append(predict_item)

    def evaluate(self):
        self._eval_display_str = '\n'
        if len(self._detection_results) == 0:
            self._eval_display_str += 'No bboxes detected! Evaluation abort!\n'
            return

        coco_predict = self._coco_gt.loadRes(self._detection_results)

        coco_eval = COCOeval(self._coco_gt, coco_predict, iouType='bbox')

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # construct display str

        metric_items = ['mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l']
        for i, metric in enumerate(metric_items):
            self._eval_display_str += '{:<10}:{:.5f}\n'.format(metric, coco_eval.stats[i])

        self._detection_results.clear()

    def get_eval_display_str(self):
        return self._eval_display_str
