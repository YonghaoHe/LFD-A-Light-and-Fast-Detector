# -*- coding: utf-8 -*-

"""
作者: 何泳澔
日期: 2020-05-18
模块文件: coco_parser.py
模块描述: 
"""
import os
from pycocotools.coco import COCO
from .sample import Sample
__all__ = ['COCOParser']


class COCOParser(object):

    def __init__(self, coco_annotation_path, image_root):
        assert os.path.exists(coco_annotation_path)
        assert os.path.exists(image_root)

        self._image_root = image_root

        self._coco = COCO(annotation_file=coco_annotation_path)

        category_ids = self._coco.getCatIds()
        category_ids.sort()
        self._category_ids_to_label_indexes = dict()
        self._label_indexes_to_category_ids = dict()
        self._category_ids_to_category_names = dict()
        for i, cat_id in enumerate(category_ids):
            self._category_ids_to_label_indexes[cat_id] = i  # CAUTION: label index is 0-based!!!!
            self._label_indexes_to_category_ids[i] = cat_id
            self._category_ids_to_category_names[cat_id] = self._coco.loadCats(cat_id)[0]['name']

    @property
    def category_ids_to_label_indexes(self):
        return self._category_ids_to_label_indexes

    @property
    def label_indexes_to_category_ids(self):
        return self._label_indexes_to_category_ids

    @property
    def category_idx_to_category_names(self):
        return self._category_ids_to_category_names

    def generate_sample(self):
        image_ids = self._coco.getImgIds()

        for image_id in image_ids:
            image_info = self._coco.loadImgs(image_id)

            annotations = self._coco.loadAnns(self._coco.getAnnIds(image_id))

            bboxes = []
            bbox_category_ids = []
            for annotation in annotations:
                bbox = annotation['bbox']
                # filter some bad cases
                if min(bbox[2:]) <= 0 or min(bbox[:2]) < 0:
                    continue
                bboxes.append(bbox)
                bbox_category_ids.append(annotation['category_id'])

            # images without any bboxes are ignored
            if len(bboxes) == 0:
                continue

            sample = Sample()
            sample['image_id'] = image_id  # image_id is not in the Sample key words, and serves as the meta info for COCO samples. It will be used by evaluator.
            sample['image_path'] = os.path.join(self._image_root, image_info[0]['file_name'])
            sample['image_type'] = image_info[0]['file_name'].split('.')[-1].lower()
            sample['bboxes'] = bboxes
            sample['bbox_labels'] = [self._category_ids_to_label_indexes[bbox_cat_id] for bbox_cat_id in bbox_category_ids]

            yield sample
