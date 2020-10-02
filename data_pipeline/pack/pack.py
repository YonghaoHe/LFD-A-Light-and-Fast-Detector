# -*- coding: utf-8 -*-

"""
作者: 何泳澔
日期: 2020-05-18
模块文件: pack.py
模块描述: 
"""
import random
import pickle
import cv2
from data_pipeline.dataset.coco_parser import COCOParser
from data_pipeline.dataset.dataset import Dataset


def pack_dataset():
    image_root = '/media/yonghaohe/HYH-4T-WD/public_dataset/COCO/val2017'
    annotation_path = '../../datasets/coco/instances_val2017.json'
    parser = COCOParser(image_root=image_root, coco_annotation_path=annotation_path)

    dataset = Dataset(parser, save_path='./coco_val2017.pkl')

    print(dataset)


def get_mini_dataset_for_debug():
    """
    get a mini dataset for debug based on existed pkl files
    :return:
    """
    pkl_file_path = './coco_val2017.pkl'
    new_pkl_file_save_path = 'mini_coco_trainval2017.pkl'
    meta_info, index_annotation_dict, dataset = pickle.load(open(pkl_file_path, 'rb'))

    new_index_annotation_dict = dict()
    new_dataset = dict()

    keys = list(index_annotation_dict.keys())
    random.shuffle(keys)
    selected_keys = keys[:320]

    for key in selected_keys:
        new_index_annotation_dict[key] = index_annotation_dict[key]
        new_dataset[key] = dataset[key]

    pickle.dump([meta_info, new_index_annotation_dict, new_dataset], open(new_pkl_file_save_path, 'wb'), pickle.HIGHEST_PROTOCOL)


def load_dataset():
    dataset = Dataset(load_path='./coco_train2017.pkl')
    print(dataset)

    index_annotation_dict = dataset.index_annotation_dict
    category_ids_to_label_indexes, label_indexes_to_category_ids, category_ids_to_category_names = dataset.meta_info

    for index, annotation in index_annotation_dict.items():
        sample = dataset[index]
        image = cv2.imread(sample['image_path'], cv2.IMREAD_COLOR)
        if 'bboxes' in sample:
            bboxes = sample['bboxes']
            bbox_labels = sample['bbox_labels']
            for i, bbox in enumerate(bboxes):
                cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])), (0, 255, 0), 2)
        cv2.imshow('image', image)
        cv2.waitKey()


if __name__ == '__main__':
    # pack_dataset()
    # load_dataset()
    get_mini_dataset_for_debug()
