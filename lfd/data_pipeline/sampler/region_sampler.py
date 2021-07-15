# -*- coding: utf-8 -*-

import math
import random
import numpy
import cv2

__all__ = ['BaseRegionSampler',
           'TypicalCOCOTrainingRegionSampler',
           'RandomBBoxCropRegionSampler',
           'RandomBBoxCropWithRangeSelectionRegionSampler',
           'IdleRegionSampler']


class BaseRegionSampler(object):
    """
    base class for region sampler
    """

    def __call__(self, sample):
        """
        given sample dict, replace 'image' key with a certain region
        :param sample:
        :return:
        """
        raise NotImplementedError


class TypicalCOCOTrainingRegionSampler(BaseRegionSampler):
    """
    this region sampler implements typical processing for coco dataset:
    resize the image while keeping the aspect ratio, so that the shorter edge is changed to 800. If the longer edge exceeds 1333, keep the longer edge to 1333
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

            bboxes_resized = [[int(bbox[0] * resize_scale), int(bbox[1] * resize_scale), max(int(bbox[2] * resize_scale), 1), max(int(bbox[3] * resize_scale), 1)]
                              for bbox in bboxes]
            sample['bboxes'] = bboxes_resized

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
    1, randomly resize the image according to resize_range, resize_prob controls the probability of performing resize
    2, (pos sample) randomly select a bbox, randomly choose a region with crop_size, containing the selected bbox
       (neg sample) randomly crop a region with crop_size
    """

    def __init__(self, crop_size, resize_range=(0.5, 1.5), resize_prob=1.0):
        assert isinstance(crop_size, int)
        assert isinstance(resize_range, (tuple, list))
        assert 0 <= resize_prob <= 1.

        self._crop_size = crop_size
        self._resize_range = resize_range
        self._resize_prob = resize_prob

    def __call__(self, sample):
        assert 'image' in sample

        image = sample['image']
        if random.random() < self._resize_prob:
            resize_scale = random.random() * (self._resize_range[1] - self._resize_range[0]) + self._resize_range[0]
        else:
            resize_scale = 1.0
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


class RandomBBoxCropWithRangeSelectionRegionSampler(BaseRegionSampler):
    """
    workflow:
    1, randomly select a bbox
    2, select a proper scale for the box, and get the resize_scale
    3, resize the image all bboxes according to eh resize_scale
    4, crop a region containing the selected bbox with crop_size

    """

    def __init__(self, crop_size, detection_ranges, range_mode='longer', neg_resize_range=(0.5, 3), range_selection_probs=None, lock_threshold=None):
        assert isinstance(crop_size, int)
        assert isinstance(detection_ranges, (tuple, list))
        assert range_mode in ['shorter', 'longer', 'sqrt']
        assert isinstance(neg_resize_range, (tuple, list)) and len(neg_resize_range) == 2
        if range_selection_probs is not None:
            assert len(detection_ranges) == len(range_selection_probs)
        if lock_threshold is not None:
            assert isinstance(lock_threshold, int)

        self._crop_size = crop_size
        self._detection_ranges = detection_ranges
        self._range_mode = range_mode
        self._range_lower_bound = self._detection_ranges[0][0]
        self._range_upper_bound = self._detection_ranges[-1][1]
        self._range_selection_probs = range_selection_probs
        self._neg_resize_range = neg_resize_range
        if self._range_selection_probs is None:
            self._range_selection_probs = [1. / len(self._detection_ranges) for _ in range(len(self._detection_ranges))]
        else:
            self._range_selection_probs = [p / sum(self._range_selection_probs) for p in self._range_selection_probs]

        self._lock_threshold = lock_threshold

    def __call__(self, sample):
        assert 'image' in sample

        image = sample['image']
        bboxes = sample['bboxes'] if 'bboxes' in sample else []
        labels = sample['bbox_labels'] if 'bbox_labels' in sample else []

        # determine target scale
        target_bbox_index = -1
        if len(bboxes) > 0:
            target_bbox_index = random.randint(0, len(bboxes) - 1)
            selected_bbox = bboxes[target_bbox_index]
            if self._range_mode == 'shorter':
                determine_side = min(selected_bbox[-2:])
            elif self._range_mode == 'longer':
                determine_side = max(selected_bbox[-2:])
            elif self._range_mode == 'sqrt':
                determine_side = (selected_bbox[-2] * selected_bbox[-1]) ** 0.5
            else:
                raise ValueError

            if determine_side <= self._range_lower_bound:
                resize_scale = 1.0
            elif self._lock_threshold and determine_side <= self._lock_threshold:
                target_length = random.randint(self._range_lower_bound, determine_side)
                resize_scale = target_length / determine_side
            else:
                if determine_side >= self._range_upper_bound and random.random() > 0.9:
                    target_length = self._range_upper_bound + random.randint(0, self._range_upper_bound * 0.5)
                    resize_scale = target_length / determine_side
                else:
                    target_range = random.choices(self._detection_ranges, self._range_selection_probs)[0]
                    target_length = random.randint(target_range[0], target_range[1])
                    resize_scale = target_length / determine_side
        else:
            resize_scale = random.random() * (self._neg_resize_range[1] - self._neg_resize_range[0]) + self._neg_resize_range[0]

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
    crop a region from the given image
    :param image:
    :param crop_region: x, y , w, h
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
