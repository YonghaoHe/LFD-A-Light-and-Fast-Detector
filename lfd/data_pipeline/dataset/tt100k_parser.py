# -*- coding: utf-8 -*-

import os
import json
from .sample import Sample
from .base_parser import Parser
__all__ = ['TT100KParser']

type45 = "i2,i4,i5,il100,il60,il80,io,ip,p10,p11,p12,p19,p23,p26,p27,p3,p5,p6,pg,ph4,ph4.5,ph5,pl100,pl120,pl20,pl30,pl40,pl5,pl50,pl60,pl70,pl80,pm20,pm30,pm55,pn,pne,po,pr40,w13,w32,w55,w57,w59,wo"
type45 = type45.split(',')


class TT100KParser(Parser):

    def __init__(self,
                 data_root,
                 annotation_json_file_path,
                 id_file_path,
                 neg_image_root=None
                 ):
        assert os.path.exists(data_root)
        assert os.path.exists(annotation_json_file_path)
        assert os.path.exists(id_file_path)
        if neg_image_root is not None:
            assert os.path.exists(neg_image_root)

        self._data_root = data_root
        self._image_id_list = open(os.path.join(id_file_path)).read().splitlines()
        self._annotations = json.load(open(annotation_json_file_path, 'r'))['imgs']
        if neg_image_root is not None:
            self._neg_image_paths_list = [os.path.join(neg_image_root, file_name) for file_name in os.listdir(neg_image_root) if file_name.lower().endswith('.jpg')]
        else:
            self._neg_image_paths_list = list()

        self._category_names_to_label_indexes = dict()
        self._label_indexes_to_category_names = dict()
        for i, t in enumerate(type45):
            self._category_names_to_label_indexes[t] = i
            self._label_indexes_to_category_names[i] = t

    def get_meta_info(self):
        return {
            'category_names_to_label_indexes': self._category_names_to_label_indexes,
            'label_indexes_to_category_names': self._label_indexes_to_category_names
        }

    def generate_sample(self):

        for identity in self._image_id_list:
            annotation = self._annotations[identity]
            bboxes = list()
            labels = list()
            for obj in annotation['objects']:
                if obj['category'] not in type45:
                    continue

                #  to [x, y, w ,h]
                x = obj['bbox']['xmin']
                y = obj['bbox']['ymin']
                w = obj['bbox']['xmax'] - obj['bbox']['xmin'] + 1
                h = obj['bbox']['ymax'] - obj['bbox']['ymin'] + 1
                if x < 0 or y < 0 or w <= 2 or h <= 2:  # filter invalid bbox
                    continue
                bboxes.append([x, y, w, h])
                labels.append(self._category_names_to_label_indexes[obj['category']])

            sample = Sample()
            sample['image_path'] = os.path.join(self._data_root, annotation['path'])
            if len(bboxes) > 0:
                sample['bboxes'] = bboxes
                sample['bbox_labels'] = labels
            yield sample

        # neg images
        for neg_image_path in self._neg_image_paths_list:
            sample = Sample()
            sample['image_path'] = neg_image_path

            yield sample

