# -*- coding: utf-8 -*-

import pickle
import os


class Dataset(object):
    """
    a dataset wrapper for save and load datasets
    """

    def __init__(self, parser=None, save_path=None, load_path=None):
        """

        :param parser:
        :param save_path:
        :param load_path:
        """

        if load_path is not None:
            self._load_path = load_path
            assert os.path.exists(self._load_path), '[%s] path does not exist!' % self._load_path
            self.__load_dataset()
        else:
            self._parser = parser
            self._save_path = save_path
            assert self._save_path is not None, 'When parser is provided, the save_path must be set!'
            self.__build_dataset()

    def __build_dataset(self):
        """
        build dataset
        :return:
        """
        print('Start to pack samples.' + '-' * 20)
        print('save path: %s' % self._save_path)
        print('-' * 30)

        if not os.path.exists(os.path.dirname(self._save_path)):
            os.makedirs(os.path.dirname(self._save_path))

        self._dataset = dict()
        self._index_annotation_dict = dict()
        self._meta_info = [self._parser.category_ids_to_label_indexes, self._parser.label_indexes_to_category_ids, self._parser.category_idx_to_category_names]

        for index, sample in enumerate(self._parser.generate_sample()):
            if 'bboxes' in sample:
                self._index_annotation_dict[index] = (sample['bboxes'], sample['bbox_labels'])
            else:
                self._index_annotation_dict[index] = None
            self._dataset[index] = sample

            print('Sample [%d] is processed.' % index)

        print('Save dataset ' + '-' * 20)
        pickle.dump([self._meta_info, self._index_annotation_dict, self._dataset], open(self._save_path, 'wb'), pickle.HIGHEST_PROTOCOL)

    def __load_dataset(self):
        """
        load dataset
        :return:
        """
        self._meta_info, self._index_annotation_dict, self._dataset = pickle.load(open(self._load_path, 'rb'))

    def __getitem__(self, index):
        """

        :param index:
        :return:
        """
        return self._dataset[index]

    def __len__(self):
        """
        :return:
        """
        return len(self._index_annotation_dict)

    def __str__(self):
        return self.get_dataset_statistics()

    @property
    def index_annotation_dict(self):
        """
        :return:
        """
        return self._index_annotation_dict

    @property
    def meta_info(self):
        return self._meta_info

    def get_dataset_statistics(self):
        """
        :return:
        """
        num_samples_with_bbox = 0
        temp_label_bboxes_dict = dict()
        for index, annotation in self._index_annotation_dict.items():
            if annotation is None:
                continue
            bboxes, labels = annotation
            for i, label in enumerate(labels):
                if label in temp_label_bboxes_dict:
                    temp_label_bboxes_dict[label] += 1
                else:
                    temp_label_bboxes_dict[label] = 1
            num_samples_with_bbox += 1

        statistics = 'Dataset statistics:--------------\n' \
                     'The total number of samples: %d\n' \
                     'The total number of classes: %d\n' \
                     'The total number of bboxes: %d\n' \
                     'The total number of neg samples: %d\n' \
                     % (self.__len__(), len(temp_label_bboxes_dict.keys()), sum(temp_label_bboxes_dict.values()), self.__len__() - num_samples_with_bbox)
        statistics += 'For each class:\n'
        for label, num_bboxes in temp_label_bboxes_dict.items():
            statistics += 'class {:>3} includes {:>9} bboxes\n'.format(label, num_bboxes)
        return statistics
