# -*- coding: utf-8 -*-

import os
from ..dataset.widerface_parser import WIDERFACEParser
from ..dataset.dataset import Dataset
__all__ = ['pack_widerface']


def pack_widerface(image_root_path, annotation_path, pack_save_path, neg_image_root_path=None):
    assert os.path.exists(image_root_path), 'image root path does not exist!'
    assert os.path.exists(annotation_path), 'annotation path does not exist!'
    if neg_image_root_path is not None:
        assert os.path.exists(neg_image_root_path), 'neg image root path does not exist!'

    if not os.path.exists(os.path.dirname(pack_save_path)):
        os.makedirs(os.path.dirname(pack_save_path))

    parser = WIDERFACEParser(
        annotation_file_path=annotation_path,
        image_root=image_root_path,
        neg_image_root=neg_image_root_path
    )

    dataset = Dataset(parser=parser, save_path=pack_save_path)

    print(dataset)
