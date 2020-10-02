# -*- coding: utf-8 -*-

"""
作者: 何泳澔
日期: 2020-05-18
模块文件: region_sampler.py
模块描述: 
"""
import numpy
import cv2
__all__ = ['TypicalCOCOTrainingRegionSampler']


class TypicalCOCOTrainingRegionSampler(object):
    """
    这个图像区域采样器将实现目前绝大多数文章中对于COCO数据集的单幅图像处理
    1，保持长宽比的前提下，缩放原图，使得短边等于800，如果此时长边大于了1333，那么以将长边缩放为1333为准
    2，输出的图像尺寸为1333x1333
    """

    def __init__(self, output_size=(1333, 1333), resize_scales=(800, 1333),):
        """

        :param output_size:
        :param resize_scales:
        """
        assert max(resize_scales) <= min(output_size)
        self._output_height = output_size[0]
        self._output_width = output_size[1]
        self._shorter_scale = min(resize_scales)
        self._longer_scale = max(resize_scales)

    def __call__(self, sample):
        assert 'image' in sample
        im = sample['image']
        im_height, im_width = im.shape[0], im.shape[1]
        resize_scale = min(self._longer_scale/max(im_height, im_width), self._shorter_scale/min(im_height, im_width))

        im_resized = cv2.resize(im, (0, 0), fx=resize_scale, fy=resize_scale)
        if 'bboxes' in sample:
            bboxes = sample['bboxes']
            # 这里需要保证新的bbox长宽大于等于1，并且bbox的范围不能超过图像的边界（让x，y向下取整）
            bboxes_resized = [[int(bbox[0]*resize_scale), int(bbox[1]*resize_scale), max(bbox[2]*resize_scale, 1), max(bbox[3]*resize_scale, 1)]
                              for bbox in bboxes]
            sample['bboxes'] = bboxes_resized

        # 将缩放后的图像放入输入大小的左上角。 这里采用了crop的方式。
        crop_region = [0, 0, self._output_width, self._output_height]
        im_resized = crop_from_image(im_resized, crop_region)

        sample['image'] = im_resized
        sample['resize_scale'] = resize_scale  # 'resize_scale' will be used as meta info by evaluator

        return sample

    @property
    def output_size(self):
        return self._output_height, self._output_width


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
