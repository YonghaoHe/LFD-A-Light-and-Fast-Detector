# -*- coding: utf-8 -*-
# author: Yonghao He
# description:
import os
from ..dataset.widerface_parser import WIDERFACEParser
from ..dataset.dataset import Dataset


def pack(image_root_path, annotation_path, pack_save_path, neg_image_root_path=None):
    assert os.path.exists(image_root_path)
    assert os.path.exists(annotation_path)
    if neg_image_root_path is not None:
        assert os.path.exists(neg_image_root_path)

    if not os.path.exists(os.path.dirname(pack_save_path)):
        os.makedirs(os.path.dirname(pack_save_path))

    parser = WIDERFACEParser(
        annotation_file_path=annotation_path,
        image_root=image_root_path,
        neg_image_root=neg_image_root_path
    )

    dataset = Dataset(parser=parser, save_path=pack_save_path)

    print(dataset)
