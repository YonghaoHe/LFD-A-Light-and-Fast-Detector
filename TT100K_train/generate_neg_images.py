# -*- coding: utf-8 -*-
# generate pure neg images from train set

import os
import json
import cv2
import numpy

type45 = "i2,i4,i5,il100,il60,il80,io,ip,p10,p11,p12,p19,p23,p26,p27,p3,p5,p6,pg,ph4,ph4.5,ph5,pl100,pl120,pl20,pl30,pl40,pl5,pl50,pl60,pl70,pl80,pm20,pm30,pm55,pn,pne,po,pr40,w13,w32,w55,w57,w59,wo"
type45 = type45.split(',')

dataset_root = '/home/yonghaohe/datasets/TT100K/data'

annotation_json_file = os.path.join(dataset_root, 'annotations.json')

train_image_id_list = open(os.path.join(dataset_root, 'train/ids.txt')).read().splitlines()

neg_image_save_root = os.path.join(dataset_root, 'train_neg')
if not os.path.exists(neg_image_save_root):
    os.makedirs(neg_image_save_root)

annotation_json = json.load(open(annotation_json_file, 'r'))
image_annotations = annotation_json['imgs']

neg_image_counter = 0
min_size_threshold = 512
for identity in train_image_id_list:
    annotation = image_annotations[identity]
    bboxes = list()
    for obj in annotation['objects']:
        if obj['category'] not in type45:
            continue
        #  store as [x1,y1,x2,y2]
        bboxes.append([
            int(obj['bbox']['xmin']),
            int(obj['bbox']['ymin']),
            int(obj['bbox']['xmax']),
            int(obj['bbox']['ymax']),
        ])

    image = cv2.imread(os.path.join(dataset_root, annotation['path']), cv2.IMREAD_UNCHANGED)

    if len(bboxes) == 0:  # in case of no bboxes, save the whole image as a neg image
        neg_image_counter += 1
        cv2.imwrite(os.path.join(neg_image_save_root, str(neg_image_counter) + '.jpg'), image)
        print('[%5d] neg image saved!' % neg_image_counter)
    else:
        bboxes = numpy.array(bboxes)
        left = numpy.min(bboxes[:, 0])
        right = numpy.max(bboxes[:, 2])
        top = numpy.min(bboxes[:, 1])
        bottom = numpy.max(bboxes[:, 3])

        if left >= min_size_threshold:
            temp_image = image[:, :left]
            neg_image_counter += 1
            cv2.imwrite(os.path.join(neg_image_save_root, str(neg_image_counter) + '.jpg'), temp_image)
            print('[%5d] neg image saved!' % neg_image_counter)
        if top >= min_size_threshold:
            temp_image = image[:top, :]
            neg_image_counter += 1
            cv2.imwrite(os.path.join(neg_image_save_root, str(neg_image_counter) + '.jpg'), temp_image)
            print('[%5d] neg image saved!' % neg_image_counter)
        if image.shape[1] - right >= min_size_threshold:
            temp_image = image[:, right:]
            neg_image_counter += 1
            cv2.imwrite(os.path.join(neg_image_save_root, str(neg_image_counter) + '.jpg'), temp_image)
            print('[%5d] neg image saved!' % neg_image_counter)
        if image.shape[0] - bottom >= min_size_threshold:
            temp_image = image[bottom:, :]
            neg_image_counter += 1
            cv2.imwrite(os.path.join(neg_image_save_root, str(neg_image_counter) + '.jpg'), temp_image)
            print('[%5d] neg image saved!' % neg_image_counter)
