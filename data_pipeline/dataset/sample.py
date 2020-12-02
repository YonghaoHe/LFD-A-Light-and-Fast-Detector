# -*- coding: utf-8 -*-

__all__ = ['Sample', 'reserved_keys']

reserved_keys = ['image_bytes', 'image_type', 'image_path', 'image', 'bboxes', 'bbox_labels']


class Sample(dict):
    """
    Sample class is a subclass of dict, storing information of a single sample
    The following keys are reserved:
    'image_bytes' image data in bytes format
    'image_type' image type, such as jpg, png, bmp
    'image_path' image path for loading
    'image' image data in numpy.ndarray
    'bboxes' bbox coordinates info for detection
    'bbox_labels' bbox label for detection
    """
    def __str__(self):
        info_str = 'The sample includes the following keys: \n'

        for key in self.keys():
            info_str += '[' + str(key) + ']\t'
        return info_str
