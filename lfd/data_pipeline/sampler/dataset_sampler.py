# -*- coding: utf-8 -*-

import random
import numpy
import math

__all__ = ['BaseDatasetSampler', 'RandomDatasetSampler', 'COCORandomDatasetSampler', 'RandomWithNegDatasetSampler']


class BaseDatasetSampler(object):
    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def get_batch_size(self):
        raise NotImplementedError


class RandomDatasetSampler(BaseDatasetSampler):
    """
    shuffle the whole dataset, and then return sample indexes sequentially
    steps：
    1) randomize all sample indexes
    2) return batched indexes sequentially
    """

    def __init__(self, dataset, batch_size=1, shuffle=True, ignore_last=False):
        """
        """
        assert len(dataset) > 0
        self._indexes = dataset.get_indexes()
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


class COCORandomDatasetSampler(BaseDatasetSampler):
    """
    group all samples by computing the aspect ratio (w/h > 1, or others)
    """

    def __init__(self, dataset, batch_size=1, shuffle=True):
        assert len(dataset) >= 1
        assert batch_size >= 1
        indexes = dataset.get_indexes()
        self._group_indexes = dict()
        for index in indexes:
            temp_sample = dataset[index]
            group_id = int(temp_sample['original_width'] / temp_sample['original_height'] < 1)
            if group_id in self._group_indexes:
                self._group_indexes[group_id].append(index)
            else:
                self._group_indexes[group_id] = [index]

        assert batch_size <= len(dataset)
        self._batch_size = batch_size
        self._shuffle = shuffle

        # pad group indexes
        num_samples = 0
        for group_id in self._group_indexes:
            temp_group_indexes = self._group_indexes[group_id]
            num_pad = math.ceil(len(temp_group_indexes) / self._batch_size) * self._batch_size - len(temp_group_indexes)
            temp_group_indexes += random.sample(temp_group_indexes, num_pad)
            self._group_indexes[group_id] = temp_group_indexes
            num_samples += len(temp_group_indexes)

        assert num_samples % self._batch_size == 0
        self._loop = num_samples // self._batch_size

    def __iter__(self):
        all_index_batches = list()
        for group_id in self._group_indexes:
            temp_group_indexes = self._group_indexes[group_id]
            if self._shuffle:
                random.shuffle(temp_group_indexes)
            temp_loop = len(temp_group_indexes) // self._batch_size
            all_index_batches += [temp_group_indexes[i * self._batch_size:(i + 1) * self._batch_size] for i in range(temp_loop)]

        random.shuffle(all_index_batches)

        for i in range(self._loop):
            yield all_index_batches[i]

    def __len__(self):
        return self._loop

    def get_batch_size(self):
        return self._batch_size


class RandomWithNegDatasetSampler(BaseDatasetSampler):
    """
    the dataset is divided into two parts: pos samples and neg samples.
    both parts are shuffled.
    for each batch, neg_ratio controls how many neg samples will be put in a batch
    """

    def __init__(self, dataset, batch_size=1, neg_ratio=0.1, shuffle=True, ignore_last=False):

        assert len(dataset) > 0, 'dataset is empty!'
        assert batch_size <= len(dataset), 'the number of samples should larger than batch size!'
        assert 0. <= neg_ratio <= 1, 'neg ratio should be in [0,1]!'

        self._batch_size = batch_size
        self._neg_ratio = neg_ratio
        self._shuffle = shuffle
        self._ignore_last = ignore_last

        self._pos_indexes = list()
        self._neg_indexes = list()
        self._all_indexes = dataset.get_indexes()
        for index in self._all_indexes:
            if 'bboxes' in dataset[index]:
                self._pos_indexes.append(index)
            else:
                self._neg_indexes.append(index)
        if len(self._neg_indexes) == 0:
            self._num_neg_samples_for_each_batch = 0
        else:
            self._num_neg_samples_for_each_batch = int(self._batch_size * self._neg_ratio)
        self._num_pos_samples_for_each_batch = self._batch_size - self._num_neg_samples_for_each_batch

        if not self._ignore_last and len(self._pos_indexes) % self._num_pos_samples_for_each_batch != 0:
            self._loop = int(len(self._pos_indexes) / self._num_pos_samples_for_each_batch) + 1
        else:
            self._loop = int(len(self._pos_indexes) / self._num_pos_samples_for_each_batch)

    def __len__(self):
        return self._loop

    def get_batch_size(self):
        return self._batch_size

    def __iter__(self):

        random.shuffle(self._pos_indexes)
        for i in range(self._loop):

            if i == self._loop - 1:
                yield self._pos_indexes[i * self._num_pos_samples_for_each_batch:] + numpy.random.choice(self._neg_indexes, self._num_neg_samples_for_each_batch).tolist()
            else:
                yield self._pos_indexes[i * self._num_pos_samples_for_each_batch:(i + 1) * self._num_pos_samples_for_each_batch] + numpy.random.choice(self._neg_indexes, self._num_neg_samples_for_each_batch).tolist()
