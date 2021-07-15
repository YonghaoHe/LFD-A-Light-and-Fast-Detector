# -*- coding: utf-8 -*-

import math
import sys
sys.path.append('..')
from lfd.data_pipeline import Dataset


dataset_path = './debug_data/train.pkl'

dataset = Dataset(load_path=dataset_path)

print(dataset)

indexes = dataset.get_indexes()

# 图像总数
num_images = len(dataset)

# 框的总数
num_bboxes = 0

# 每个类别框的总数
num_bboxes_per_label = dict()

# 所有框的大小分布，主要根据三个量进行统计：短边长度，长边长度和开方长度
num_bboxes_per_shorter_unit = dict()
num_bboxes_per_longer_unit = dict()
num_bboxes_per_sqrt_unit = dict()

for index in indexes:
    sample = dataset[index]
    labels = sample.get('bbox_labels', [])
    bboxes = sample.get('bboxes', [])
    assert len(labels) == len(bboxes)

    num_bboxes += len(labels)
    for label in labels:
        if label in num_bboxes_per_label:
            num_bboxes_per_label[label] += 1
        else:
            num_bboxes_per_label[label] = 1

    for bbox in bboxes:
        width = int(bbox[2] + 0.5)  # 四舍五入
        height = int(bbox[3] + 0.5)

        if min(width, height) in num_bboxes_per_shorter_unit:
            num_bboxes_per_shorter_unit[min(width, height)] += 1
        else:
            num_bboxes_per_shorter_unit[min(width, height)] = 1

        if max(width, height) in num_bboxes_per_longer_unit:
            num_bboxes_per_longer_unit[max(width, height)] += 1
        else:
            num_bboxes_per_longer_unit[max(width, height)] = 1

        sqrt_length = math.floor(math.sqrt(width * height) + 0.5)
        if sqrt_length in num_bboxes_per_sqrt_unit:
            num_bboxes_per_sqrt_unit[sqrt_length] += 1
        else:
            num_bboxes_per_sqrt_unit[sqrt_length] = 1

# 类别总数
num_categories = len(num_bboxes_per_label)

print('整体分析------------------------------')
print('图像总数： %d' % num_images)
print('类别总数： %d' % num_categories)
print('框的总数： %d' % num_bboxes)
print('每个类别框的总数：')
labels = list(num_bboxes_per_label.keys())
labels.sort()
for label in labels:
    print('类别[%d] ---- %d' % (label, num_bboxes_per_label[label]))

print('以框的短边长度为变量，统计每个长度下，框的总数：')
shorter_lengths = list(num_bboxes_per_shorter_unit.keys())
shorter_lengths.sort()
for length in shorter_lengths:
    print('边长[%4d] ---- %4d' % (length, num_bboxes_per_shorter_unit[length]))

print('以框的长边长度为变量，统计每个长度下，框的总数：')
longer_lengths = list(num_bboxes_per_longer_unit.keys())
longer_lengths.sort()
for length in longer_lengths:
    print('边长[%4d] ---- %4d' % (length, num_bboxes_per_longer_unit[length]))

print('以框的开方长度为变量，统计每个长度下，框的总数：')
sqrt_lengths = list(num_bboxes_per_sqrt_unit.keys())
sqrt_lengths.sort()
for length in sqrt_lengths:
    print('边长[%4d] ---- %4d' % (length, num_bboxes_per_sqrt_unit[length]))
