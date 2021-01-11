# -*- coding: utf-8 -*-
import sys
sys.path.append('../..')
from lfd.data_pipeline.pack.pack_coco import pack, check_by_show


def pack_dataset():
    image_root_path = '/home/yonghaohe/datasets/COCO/val2017'
    annotation_pth = '/home/yonghaohe/datasets/COCO/annotations/instances_val2017.json'
    pack_save_path = './COCO_pack/coco_val2017.pkl'

    pack(image_root_path, annotation_pth, pack_save_path, filter_no_gt=True, filter_min_size=32)


def check_dataset():
    pkl_path = './COCO_pack/coco_val2017.pkl'
    check_by_show(pkl_path)


if __name__ == '__main__':
    # pack_dataset()
    check_dataset()
