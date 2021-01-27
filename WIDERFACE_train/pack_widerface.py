# -*- coding: utf-8 -*-

from lfd.data_pipeline import Dataset
from lfd.data_pipeline.pack.pack_widerface import pack


def pack_dataset():
    image_root_path = '/home/yonghaohe/datasets/WIDER_FACE/WIDER_train/images'
    annotation_path = '/home/yonghaohe/datasets/WIDER_FACE/wider_face_split/wider_face_train_bbx_gt.txt'
    neg_image_root_path = '/home/yonghaohe/datasets/WIDER_FACE/WIDER_train/neg_images'
    pack_save_path = './WIDERFACE_pack/widerface_train.pkl'

    pack(image_root_path=image_root_path,
         annotation_path=annotation_path,
         pack_save_path=pack_save_path,
         neg_image_root_path=neg_image_root_path)


def check_dataset():
    dataset_path = './WIDERFACE_pack/widerface_train.pkl'

    dataset = Dataset(load_path=dataset_path)

    print(dataset)

    indexes = dataset.get_indexes()

    import random, cv2, numpy
    random.shuffle(indexes)

    for index in indexes:
        sample = dataset[index]
        if 'bboxes' in sample:
            im = cv2.imdecode(numpy.frombuffer(sample['image_bytes'], dtype=numpy.uint8), cv2.IMREAD_UNCHANGED)
            for bbox in sample['bboxes']:
                cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)

            cv2.imshow('im', im)
            cv2.waitKey()


if __name__ == '__main__':
    pack_dataset()
    # check_dataset()

