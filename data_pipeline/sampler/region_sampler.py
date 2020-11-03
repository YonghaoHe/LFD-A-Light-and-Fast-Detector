# -*- coding: utf-8 -*-

"""
作者: 何泳澔
日期: 2020-05-18
模块文件: region_sampler.py
模块描述: 
"""
import math
import random
import numpy
import cv2

__all__ = ['BaseRegionSampler', 'TypicalCOCOTrainingRegionSampler']


class BaseRegionSampler(object):
    """
    base class for region sampler
    """

    def __call__(self, sample):
        """
        given sample dict, replace 'image' key with certain region
        :param sample:
        :return:
        """
        raise NotImplementedError


class TypicalCOCOTrainingRegionSampler(BaseRegionSampler):
    """
    this region sampler implements typical processing for coco dataset:
    1) resize the image while keeping the aspect ratio, so that the shorter edge is changed to 800. If the longer edge exceeds 1333, keep the longer edge to 1333
    2) the default output region size is (h, w) = (1333, 1333), make sure to contain all the cases
    """

    def __init__(self, resize_shorter_range=(800,), resize_longer_limit=1333, pad_divisor=32):
        """

        """
        assert isinstance(resize_shorter_range, tuple)
        assert max(resize_shorter_range) <= resize_longer_limit
        assert pad_divisor > 0
        self._pad_divisor = pad_divisor
        self._resize_shorter_min = min(resize_shorter_range)
        self._resize_shorter_max = max(resize_shorter_range)
        self._resize_longer_limit = resize_longer_limit

    def __call__(self, sample):
        assert 'image' in sample
        im = sample['image']
        im_height, im_width = im.shape[0], im.shape[1]
        shorter_target = random.randint(self._resize_shorter_min, self._resize_shorter_max)
        resize_scale = min(self._resize_longer_limit / max(im_height, im_width), shorter_target / min(im_height, im_width))

        im_resized = cv2.resize(im, (0, 0), fx=resize_scale, fy=resize_scale)
        if 'bboxes' in sample:
            bboxes = sample['bboxes']
            # 这里需要保证新的bbox长宽大于等于1，并且bbox的范围不能超过图像的边界（让x，y向下取整）
            bboxes_resized = [[int(bbox[0] * resize_scale), int(bbox[1] * resize_scale), max(int(bbox[2] * resize_scale), 1), max(int(bbox[3] * resize_scale), 1)]
                              for bbox in bboxes]
            sample['bboxes'] = bboxes_resized

        # 将缩放后的图像放入输入大小的左上角。 这里采用了crop的方式。
        target_height = math.ceil(im_resized.shape[0] / self._pad_divisor) * self._pad_divisor
        target_width = math.ceil(im_resized.shape[1] / self._pad_divisor) * self._pad_divisor
        crop_region = [0, 0, target_width, target_height]
        im_resized = crop_from_image(im_resized, crop_region)

        sample['image'] = im_resized
        sample['resize_scale'] = resize_scale
        sample['resized_height'] = int(im_height * resize_scale)
        sample['resized_width'] = int(im_width * resize_scale)

        return sample


def crop_from_image(image, crop_region):
    """
    从图像中裁剪相应的区域
    支持超过图像边界的裁剪，超过的部分补0
    :param image:
    :param crop_region:
    :return:
    """
    im_w = image.shape[1]
    im_h = image.shape[0]

    crop_x, crop_y, crop_w, crop_h = crop_region

    if image.ndim == 3:
        image_crop = numpy.zeros((crop_h, crop_w, 3), dtype=image.dtype)
    else:
        image_crop = numpy.zeros((crop_h, crop_w), dtype=image.dtype)

    image_crop[max(0, -crop_y):min(crop_h, im_h - crop_y), max(0, -crop_x):min(crop_w, im_w - crop_x)] = \
        image[max(0, crop_y):min(im_h, crop_h + crop_y), max(0, crop_x):min(im_w, crop_w + crop_x)]

    return image_crop
