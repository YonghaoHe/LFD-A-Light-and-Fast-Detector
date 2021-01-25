# -*- coding: utf-8 -*-
import os
from ..dataset.tt100k_parser import TT100KParser
from ..dataset.dataset import Dataset

__all__ = ['pack']


def pack(data_root,
         annotation_json_file_path,
         id_file_path,
         pack_save_path,
         neg_image_root_path=None,
         ):

    if not os.path.exists(os.path.dirname(pack_save_path)):
        os.makedirs(os.path.dirname(pack_save_path))

    parser = TT100KParser(
        data_root=data_root,
        annotation_json_file_path=annotation_json_file_path,
        id_file_path=id_file_path,
        neg_image_root=neg_image_root_path,
    )

    dataset = Dataset(parser=parser, save_path=pack_save_path)

    print(dataset)
