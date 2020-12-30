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

__all__ = ['BaseRegionSampler', 'TypicalCOCOTrainingRegionSampler', 'RandomBBoxCropRegionSampler', 'RandomBBoxCropWithScaleSelectionRegionSampler', 'IdleRegionSampler']


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


class RandomBBoxCropRegionSampler(BaseRegionSampler):
    """
    workflow:
    1, randomly resize the image according to resize_range
    2, (pos sample) randomly select a bbox, randomly choose a region with crop_size ,trying to contain the selected bbox
       (neg sample) randomly crop a region with crop_size
    """

    def __init__(self, crop_size, resize_range=(0.5, 1.5)):
        assert isinstance(crop_size, int)
        assert isinstance(resize_range, (tuple, list))

        self._crop_size = crop_size
        self._resize_range = resize_range

    def __call__(self, sample):
        assert 'image' in sample

        resize_scale = random.random() * (self._resize_range[1] - self._resize_range[0]) + self._resize_range[0]
        image = sample['image']
        image = cv2.resize(image, (0, 0), fx=resize_scale, fy=resize_scale)

        bboxes = sample['bboxes'] if 'bboxes' in sample else []
        labels = sample['bbox_labels'] if 'bbox_labels' in sample else []

        # rescale bboxes
        scaled_bboxes = []
        for bbox in bboxes:
            scaled_x = int(bbox[0] * resize_scale)
            scaled_y = int(bbox[1] * resize_scale)
            scaled_w = math.ceil(bbox[2] * resize_scale)
            scaled_h = math.ceil(bbox[3] * resize_scale)
            scaled_bboxes.append([scaled_x, scaled_y, scaled_w, scaled_h])

        target_bbox = random.choice(scaled_bboxes) if len(scaled_bboxes) > 0 else [0, 0, image.shape[1], image.shape[0]]

        w_range = self._crop_size - target_bbox[2]
        h_range = self._crop_size - target_bbox[3]

        crop_x = target_bbox[0] - random.randint(min(0, w_range), max(0, w_range))
        crop_y = target_bbox[1] - random.randint(min(0, h_range), max(0, h_range))

        crop_region = (crop_x, crop_y, self._crop_size, self._crop_size)

        new_bboxes = []
        new_labels = []
        for i, bbox in enumerate(scaled_bboxes):
            new_x = max(0, bbox[0] - crop_x)
            new_y = max(0, bbox[1] - crop_y)
            new_w = min(self._crop_size, bbox[0] + bbox[2] - crop_x) - new_x - 1
            new_h = min(self._crop_size, bbox[1] + bbox[3] - crop_y) - new_y - 1
            if new_w <= 1 or new_x >= self._crop_size or new_h <= 1 or new_y >= self._crop_size:
                continue
            new_bboxes.append([new_x, new_y, new_w, new_h])
            new_labels.append(labels[i])

        sample['image'] = crop_from_image(image, crop_region)
        if len(new_bboxes) > 0:
            sample['bboxes'] = new_bboxes
            sample['bbox_labels'] = new_labels
        else:  # if 'bboxes' is originally in sample, it should be deleted here when len(new_bboxes)==0
            if 'bboxes' in sample:
                del sample['bboxes'], sample['bbox_labels']

        return sample


class RandomBBoxCropWithScaleSelectionRegionSampler(BaseRegionSampler):
    """
    workflow:
    1, randomly select a bbox
    2, select a scale for detecting the selected bbox based on certain rules

    """

    def __init__(self, crop_size, detection_scales, scale_selection_probs=None, lock_threshold=None):
        assert isinstance(crop_size, int)
        assert isinstance(detection_scales, (tuple, list))
        if scale_selection_probs is not None:
            assert len(detection_scales) == len(scale_selection_probs)
        if lock_threshold is not None:
            assert isinstance(lock_threshold, int)

        self._crop_size = crop_size
        self._detection_scales = detection_scales
        self._scale_lower_bound = self._detection_scales[0][0]
        self._scale_selection_probs = scale_selection_probs
        if self._scale_selection_probs is None:
            self._scale_selection_probs = [1./len(self._detection_scales) for _ in range(len(self._detection_scales))]
        else:
            self._scale_selection_probs = [p/sum(self._scale_selection_probs) for p in self._scale_selection_probs]

        self._lock_threshold = lock_threshold

    def __call__(self, sample):
        assert 'image' in sample

        image = sample['image']
        bboxes = sample['bboxes'] if 'bboxes' in sample else []
        labels = sample['bbox_labels'] if 'bbox_labels' in sample else []

        # determine target scale
        target_bbox_index = -1
        if len(bboxes) > 0:
            target_bbox_index = random.randint(0, len(bboxes)-1)
            selected_bbox = bboxes[target_bbox_index]
            longer_side = max(selected_bbox[-2:])
            longer_side = max(self._scale_lower_bound, longer_side)
            if self._lock_threshold and longer_side <= self._lock_threshold:
                target_length = random.randint(self._scale_lower_bound, longer_side)
                resize_scale = target_length / longer_side
            else:
                target_scale = random.choices(self._detection_scales, self._scale_selection_probs)[0]
                target_length = random.randint(target_scale[0], target_scale[1])
                resize_scale = target_length / longer_side
        else:
            resize_scale = random.random() + 0.5  # [0.5, 1.5]

        image = cv2.resize(image, (0, 0), fx=resize_scale, fy=resize_scale)
        # rescale bboxes
        scaled_bboxes = []
        for bbox in bboxes:
            scaled_x = int(bbox[0] * resize_scale)
            scaled_y = int(bbox[1] * resize_scale)
            scaled_w = math.ceil(bbox[2] * resize_scale)
            scaled_h = math.ceil(bbox[3] * resize_scale)
            scaled_bboxes.append([scaled_x, scaled_y, scaled_w, scaled_h])

        target_bbox = scaled_bboxes[target_bbox_index] if len(scaled_bboxes) > 0 else [0, 0, image.shape[1], image.shape[0]]

        w_range = self._crop_size - target_bbox[2]
        h_range = self._crop_size - target_bbox[3]

        crop_x = target_bbox[0] - random.randint(min(0, w_range), max(0, w_range))
        crop_y = target_bbox[1] - random.randint(min(0, h_range), max(0, h_range))

        crop_region = (crop_x, crop_y, self._crop_size, self._crop_size)

        new_bboxes = []
        new_labels = []
        for i, bbox in enumerate(scaled_bboxes):
            new_x = max(0, bbox[0] - crop_x)
            new_y = max(0, bbox[1] - crop_y)
            new_w = min(self._crop_size, bbox[0] + bbox[2] - crop_x) - new_x - 1
            new_h = min(self._crop_size, bbox[1] + bbox[3] - crop_y) - new_y - 1
            if new_w <= 1 or new_x >= self._crop_size or new_h <= 1 or new_y >= self._crop_size:
                continue
            new_bboxes.append([new_x, new_y, new_w, new_h])
            new_labels.append(labels[i])

        sample['image'] = crop_from_image(image, crop_region)
        if len(new_bboxes) > 0:
            sample['bboxes'] = new_bboxes
            sample['bbox_labels'] = new_labels
        else:  # if 'bboxes' is originally in sample, it should be deleted here when len(new_bboxes)==0
            if 'bboxes' in sample:
                del sample['bboxes'], sample['bbox_labels']

        return sample


class IdleRegionSampler(BaseRegionSampler):
    """
    this region sampler does not make any changes to the sample
    in most cases, it's used for evaluation
    """

    def __init__(self):
        pass

    def __call__(self, sample):
        assert 'image' in sample

        sample['resize_scale'] = 1.
        sample['resized_height'] = sample['image'].shape[0]
        sample['resized_width'] = sample['image'].shape[1]

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
