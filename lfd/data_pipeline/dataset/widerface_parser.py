# -*- coding: utf-8 -*-

import os
from .sample import Sample
from .base_parser import Parser
__all__ = ['WIDERFACEParser']


class WIDERFACEParser(Parser):

    def __init__(self, annotation_file_path, image_root, neg_image_root=None):
        assert os.path.exists(annotation_file_path)
        assert os.path.exists(image_root)
        if neg_image_root is not None:
            assert os.path.exists(neg_image_root)

        self._annotation_file_path = annotation_file_path
        self._image_root = image_root
        self._neg_image_root = neg_image_root

    def get_meta_info(self):
        return None

    def generate_sample(self):

        fin = open(self._annotation_file_path, 'r')
        line = fin.readline()
        while line:

            line = line.strip('\n')
            if line.endswith('.jpg'):
                image_path = os.path.join(self._image_root, line)
                line = fin.readline()
                continue

            num_bboxes = int(line)
            bboxes = list()
            if num_bboxes == 0:  # skip 0 0 0 0 0 line
                num_bboxes += 1

            for i in range(num_bboxes):
                line = fin.readline()
                line = line.strip('\n').split(' ')
                x = int(line[0])
                y = int(line[1])
                w = int(line[2])
                h = int(line[3])
                if x < 0 or y < 0 or w <= 0 or h <= 0:  # filter invalid bbox
                    continue
                bboxes.append([x, y, w, h])  # x y w h

            sample = Sample()
            sample['image_path'] = image_path
            with open(image_path, 'rb') as fin_im:
                image_bytes = fin_im.read()
            sample['image_bytes'] = image_bytes
            if len(bboxes) > 0:
                sample['bboxes'] = bboxes
                sample['bbox_labels'] = [0 for _ in range(len(bboxes))]  # only one class, the label index is 0

            yield sample

            line = fin.readline()
        fin.close()

        # read neg images
        if self._neg_image_root is not None:
            neg_image_paths_list = [os.path.join(self._neg_image_root, image_name) for image_name in os.listdir(self._neg_image_root) if image_name.lower().endswith('jpg')]
            for neg_image_path in neg_image_paths_list:
                sample = Sample()
                sample['image_path'] = neg_image_path
                with open(neg_image_path, 'rb') as fin_im:
                    image_bytes = fin_im.read()
                sample['image_bytes'] = image_bytes

                yield sample

