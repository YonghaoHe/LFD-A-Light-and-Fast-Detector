# -*- coding: utf-8 -*-

from lfd.data_pipeline import Dataset
from lfd.data_pipeline.pack.pack_tt100k import pack


def pack_dataset():
    data_root = '/home/yonghaohe/datasets/TT100K/data'
    annotation_json_file_path = '/home/yonghaohe/datasets/TT100K/data/annotations.json'
    id_file_path = '/home/yonghaohe/datasets/TT100K/data/train/ids.txt'
    pack_save_path = './TT100K_pack/train.pkl'
    neg_image_root_path = '/home/yonghaohe/datasets/TT100K/data/train_neg'

    pack(data_root=data_root,
         annotation_json_file_path=annotation_json_file_path,
         id_file_path=id_file_path,
         pack_save_path=pack_save_path,
         neg_image_root_path=neg_image_root_path,
         )


def check_dataset():
    dataset_path = './TT100K_pack/train.pkl'

    dataset = Dataset(load_path=dataset_path)

    label_indexes_to_category_names = dataset.meta_info['label_indexes_to_category_names']

    print(dataset)

    indexes = dataset.get_indexes()

    import random, cv2
    random.shuffle(indexes)

    for index in indexes:
        sample = dataset[index]
        if 'bboxes' in sample:
            im = cv2.imread(sample['image_path'], cv2.IMREAD_UNCHANGED)
            for label, bbox in zip(sample['bbox_labels'], sample['bboxes']):
                cv2.rectangle(im, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), (0, 255, 0), 2)
                cv2.putText(im, label_indexes_to_category_names[label], (int(bbox[0] - 10), int(bbox[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255))

            cv2.imshow('im', im)
            cv2.waitKey()


if __name__ == '__main__':
    pack_dataset()
    # check_dataset()
