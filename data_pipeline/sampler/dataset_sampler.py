# -*- coding: utf-8 -*-

import random

__all__ = ['RandomDatasetSampler']


class RandomDatasetSampler(object):
    """
    shuffle the whole dataset, and then return sample indexes sequentially
    steps：
    1) randomize all sample indexes
    2) return batched indexes sequentially
    """

    def __init__(self, index_annotation_dict, batch_size=1, shuffle=True, ignore_last=False):
        """

        :param index_annotation_dict: key为sample的索引，value为样本的标签。通常由调用dataset的相关函数获取
        :param batch_size:
        :param shuffle: 开始返回前，是否打乱所有的indexes
        :param ignore_last: 当dataset中的sample个数无法被batch_size整除时，是否返回最后一个batch
        """
        assert isinstance(index_annotation_dict, dict)

        self._indexes = list(index_annotation_dict.keys())
        self._num_samples = len(self._indexes)
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._ignore_last = ignore_last
        assert self._batch_size <= self._num_samples

        if not self._ignore_last and self._num_samples % self._batch_size != 0:
            self._loops = int(self._num_samples / self._batch_size) + 1
        else:
            self._loops = int(self._num_samples / self._batch_size)

    def __iter__(self):
        if self._shuffle:
            random.shuffle(self._indexes)

        for i in range(self._loops):
            if i == self._loops - 1:
                selected_indexes = self._indexes[i * self._batch_size:]
            else:
                selected_indexes = self._indexes[i * self._batch_size: (i + 1) * self._batch_size]
            yield selected_indexes

    def __len__(self):
        return self._loops

    def get_batch_size(self):
        return self._batch_size
