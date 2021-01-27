# -*- coding: utf-8 -*-
import os
import cv2
import numpy


image_root = '/home/yonghaohe/datasets/WIDER_FACE/WIDER_train/images'
annotation_path = '/home/yonghaohe/datasets/WIDER_FACE/wider_face_split/wider_face_train_bbx_gt.txt'
neg_image_save_root = '/home/yonghaohe/datasets/WIDER_FACE/WIDER_train/neg_images'
if not os.path.exists(neg_image_save_root):
    os.makedirs(neg_image_save_root)

neg_image_counter = 0
min_size_threshold = 100

with open(annotation_path, 'r') as fin:
    line = fin.readline()
    while line:

        line = line.strip('\n')
        if line.endswith('.jpg'):
            image_path = os.path.join(image_root, line)
            line = fin.readline()
            continue

        num_bboxes = int(line)
        bboxes = list()

        if num_bboxes == 0:
            num_bboxes += 1

        for i in range(num_bboxes):
            line = fin.readline()
            line = line.strip('\n').split(' ')
            x1 = int(line[0])
            y1 = int(line[1])
            x2 = int(line[2]) + int(line[0])
            y2 = int(line[3]) + int(line[1])
            if x1 < 0 or y1 < 0 or x2-x1 <= 0 or y2-y1 <= 0:  # filter invalid bbox
                continue
            bboxes.append([x1, y1, x2, y2])

        if len(bboxes) == 0:
            line = fin.readline()
            continue

        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        bboxes = numpy.array(bboxes)
        left = numpy.min(bboxes[:, 0])
        top = numpy.min(bboxes[:, 1])
        right = numpy.max(bboxes[:, 2])
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

        line = fin.readline()
