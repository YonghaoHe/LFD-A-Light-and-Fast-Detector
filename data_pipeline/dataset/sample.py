# -*- coding: utf-8 -*-

__all__ = ['Sample', 'reserved_keys']

reserved_keys = ['image_bytes', 'image_type', 'image_path', 'image', 'bboxes', 'bbox_labels']


class Sample(dict):
    """
    Sample class is for storing required information of a single sample
    Sample works in a dict-like way, and some reserved key wards are defined：
    'image_bytes' image data in bytes format
    'image_type' image type, such as jpg, png, bmp
    'image_path' image path for loading
    'image' image data in numpy.ndarray
    'bboxes' bbox coordinates info for detection
    'bbox_labels' bbox label for detection
    """

    def __init__(self, sample_dict=None):
        """

        :param sample_dict: predefined sample dict
        """
        super(Sample, self).__init__()
        if sample_dict is not None:
            assert isinstance(sample_dict, dict)
            self._sample = sample_dict
        else:
            self._sample = dict()

    def __setitem__(self, key, value):
        """
        add new item
        :param key: str
        :param value:
        :return:
        """
        self._sample[key] = value

    def __getitem__(self, key):
        """
        visit content by key
        :param key:
        :return:
        """
        return self._sample[key]

    def __delitem__(self, key):
        if key in self._sample:
            del self._sample[key]

    def __str__(self):
        if self._sample:
            sample_str = 'The sample includes the following keys: \n'
            for key in self._sample.keys():
                sample_str += '[' + key + ']'
        else:
            sample_str = 'The sample is empty!'
        return sample_str

    def __contains__(self, item):
        if item in self._sample:
            return True
        else:
            return False

    def update(self, update_dict):
        """
        update new content
        :param update_dict:
        :return:
        """
        assert isinstance(update_dict, dict), 'the input arg should be a dict!'
        self._sample.update(update_dict)

    def keys(self):
        """
        return current keys as a list
        :return:
        """
        return list(self._sample.keys())
