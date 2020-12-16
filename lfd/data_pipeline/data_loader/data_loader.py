# -*- coding: utf-8 -*-

import queue
import threading
import numpy
import cv2

from ..dataset import turbojpeg, reserved_keys


class DataLoader(object):
    """

    """
    def __init__(self,
                 dataset,
                 dataset_sampler,
                 region_sampler,
                 augmentation_pipeline=None,
                 num_workers=1):
        """

        :param dataset:
        :param dataset_sampler:
        :param region_sampler:
        :param augmentation_pipeline:
        :param num_workers:

        """

        self._dataset = dataset
        self._dataset_sampler = dataset_sampler
        self._loops = len(self._dataset_sampler)
        self._batch_size = dataset_sampler.get_batch_size()
        self._region_sampler = region_sampler
        self._augmentation_pipeline = augmentation_pipeline
        self._num_workers = num_workers

        self._index_queue = queue.Queue()
        self._batch_queue = queue.Queue(maxsize=self._num_workers)

        self.__start_workers()

    def __start_workers(self):
        for i in range(self._num_workers):
            worker = threading.Thread(target=self.__worker_func, args=(), daemon=True)
            worker.start()

    def _decode_image(self, sample):
        if 'image' in sample:  # the decoded image
            return sample['image']
        elif 'image_bytes' in sample:  # the encoded image bytes
            try:
                image = turbojpeg.decode(sample['image_bytes'])
            except:
                image = cv2.imdecode(numpy.frombuffer(sample['image_bytes'], dtype=numpy.uint8), cv2.IMREAD_UNCHANGED)
            return image
        elif 'image_path' in sample:  # the image path
            with open(sample['image_path'], 'rb') as fin:
                image_bytes = fin.read()
            try:
                image = turbojpeg.decode(image_bytes)
            except:
                image = cv2.imdecode(numpy.frombuffer(image_bytes, dtype=numpy.uint8), cv2.IMREAD_UNCHANGED)
            return image
        else:
            raise ValueError('sample does not have "image", "image_bytes" or "image_path"!')

    def _image_batch_postprocess(self, image_batch):
        assert isinstance(image_batch, list)

        # get max width and height in this batch
        height_list, width_list = list(), list()
        for image in image_batch:
            height_list.append(image.shape[0])
            width_list.append(image.shape[1])
        numpy_image_batch = numpy.zeros((len(image_batch), max(height_list), max(width_list), 3), dtype=numpy.float32) if image_batch[0].ndim == 3 else numpy.zeros((len(image_batch), max(height_list), max(width_list), 1), dtype=numpy.float32)

        # fill numpy_image_batch by putting each image to the left-top corner
        for i, image in enumerate(image_batch):
            numpy_image_batch[i, 0:image.shape[0], 0:image.shape[1]] = image

        numpy_image_batch = numpy_image_batch.transpose([0, 3, 1, 2])
        return numpy_image_batch

    def __worker_func(self):

        while True:
            # obtain indexes of a batch
            index_batch = self._index_queue.get()

            image_batch = list()
            annotation_batch = list()
            meta_batch = list()  # store some meta information related to the sample. for example, image id for each sample in COCO dataset

            for i, sample_index in enumerate(index_batch):

                sample = self._dataset[sample_index]

                # declare a new dict to store content of the current sample, avoiding memory consumption due to new content added to the original dataset dict (namely self._dataset)
                sample_temp = dict()
                if 'bboxes' in sample:
                    sample_temp['bboxes'] = sample['bboxes']
                    sample_temp['bbox_labels'] = sample['bbox_labels']

                meta_keys = set(sample.keys()) - set(reserved_keys)
                # add meta key to sample_temp
                for meta_key in meta_keys:
                    sample_temp[meta_key] = sample[meta_key]

                # image decode ----------------------------------------------------------
                image = self._decode_image(sample)
                assert image is not None
                sample_temp['image'] = image

                # region sampling -----------------------------------------------------------
                sample_temp = self._region_sampler(sample_temp)

                # data augmentation --------------------------------------------------------------
                if sample_temp['image'].ndim == 2:  # adjust the image according to the input channels
                    image = numpy.tile(sample_temp['image'], (3, 1, 1))
                    sample_temp['image'] = image.transpose([1, 2, 0])
                if self._augmentation_pipeline is not None:
                    sample_temp = self._augmentation_pipeline(sample_temp)

                # fill batch
                image_batch.append(sample_temp['image'])
                if 'bboxes' in sample_temp:  # transform annotation from list to numpy
                    annotation_batch.append((numpy.array(sample_temp['bboxes'], dtype=numpy.float32), numpy.array(sample_temp['bbox_labels'], dtype=numpy.int64)))
                else:
                    annotation_batch.append((numpy.empty((0, 4), dtype=numpy.float32), numpy.empty((0,), dtype=numpy.int64)))

                # fill meta_batch using sample_temp
                # during processing, new meta keys may be added, so it is necessary to get meta keys again
                meta_keys = set(sample_temp.keys()) - set(reserved_keys)
                if len(meta_keys) == 0:
                    meta_batch.append(None)
                else:
                    meta_batch.append({k: sample_temp[k] for k in meta_keys})

            # perform post image batch process
            image_batch = self._image_batch_postprocess(image_batch)

            self._batch_queue.put((image_batch, annotation_batch, meta_batch))

    def __iter__(self):
        # 首先将所有的 batch indexes 都放入queue中
        for index_batch in self._dataset_sampler:
            self._index_queue.put(index_batch)

        # 开始返回batch
        loop_counter = 0
        while loop_counter < self._loops:
            yield self._batch_queue.get()
            loop_counter += 1

    def __len__(self):
        # 返回dataloader的长度，即获取batch的数量，也就是一次epoch需要循环的次数，
        return self._loops

    @property
    def batch_size(self):
        return self._batch_size
