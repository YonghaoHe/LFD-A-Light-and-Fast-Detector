# -*- coding: utf-8 -*-

from lfd.execution.utils import load_checkpoint
from lfd.data_pipeline.dataset import Sample
from lfd.model.utils import multiclass_nms
import torch
import numpy
import cv2
import os

from TL_augmentation_pipeline import *
from TL_LFD_L_work_dir_20210714_173824.TL_LFD_L import config_dict, prepare_model

prepare_model()
param_file_path = './TL_LFD_L_work_dir_20210714_173824/epoch_100.pth'
load_checkpoint(config_dict['model'], load_path=param_file_path, strict=True)


def predict(model,
            image,
            aug_pipeline,
            classification_threshold=None,
            nms_threshold=None,
            class_agnostic=False,
            cuda_device_index=0):
    assert isinstance(image, str) or isinstance(image, numpy.ndarray)
    if isinstance(image, str):
        image = cv2.imread(image, cv2.IMREAD_UNCHANGED)
        assert image is not None, 'image is None, confirm that the path is valid!'

    sample = Sample()
    sample['image'] = image
    sample = aug_pipeline(sample)
    data_batch = sample['image']
    data_batch = data_batch[None]
    data_batch = data_batch.transpose([0, 3, 1, 2])
    data_batch = torch.from_numpy(data_batch)

    image_width = data_batch.size(3)
    image_height = data_batch.size(2)
    data_batch = data_batch.cuda(cuda_device_index)
    model.cuda(cuda_device_index)
    model.eval()

    with torch.no_grad():
        predicted_classification, predicted_regression = model.forward(data_batch)
    predicted_classification = predicted_classification[0]
    predicted_regression = predicted_regression[0]

    all_point_coordinates_list = model.generate_point_coordinates(model._head_indexes_to_feature_map_sizes)
    expanded_regression_ranges_list = [all_point_coordinates_list[i].new_tensor(model._regression_ranges[i])[None].expand_as(all_point_coordinates_list[i])
                                       for i in range(model._num_heads)]

    concat_point_coordinates = torch.cat(all_point_coordinates_list, dim=0)
    concat_regression_ranges = torch.cat(expanded_regression_ranges_list, dim=0)

    predicted_classification = predicted_classification.sigmoid()

    concat_point_coordinates = concat_point_coordinates.to(predicted_regression.device)
    concat_regression_ranges = concat_regression_ranges.to(predicted_regression.device)

    max_scores = predicted_classification.max(dim=1)[0]
    selected_indexes = torch.where(max_scores > classification_threshold)[0]
    if selected_indexes.numel() == 0:
        return []

    predicted_classification = predicted_classification[selected_indexes]
    predicted_regression = predicted_regression[selected_indexes]
    concat_point_coordinates = concat_point_coordinates[selected_indexes]
    concat_regression_ranges = concat_regression_ranges[selected_indexes]

    #  calculate bboxes' x1 y1 x2 y2
    concat_regression_ranges_max = concat_regression_ranges.max(dim=-1)[0]
    predicted_regression = predicted_regression.sigmoid() * concat_regression_ranges_max[..., None]
    predicted_bboxes = model.distance2bbox(concat_point_coordinates, predicted_regression, max_shape=(image_height, image_width))

    # add BG label for multi class nms
    bg_label_padding = predicted_classification.new_zeros(predicted_classification.size(0), 1)
    predicted_classification = torch.cat([predicted_classification, bg_label_padding], dim=1)

    if nms_threshold:
        model._nms_cfg.update({'iou_thr': nms_threshold})
    if class_agnostic:
        model._nms_cfg.update({'class_agnostic': class_agnostic})
    nms_bboxes, nms_labels = multiclass_nms(
        multi_bboxes=predicted_bboxes,
        multi_scores=predicted_classification,
        score_thr=classification_threshold,
        nms_cfg=model._nms_cfg,
        max_num=-1,
        score_factors=None
    )

    if nms_bboxes.size(0) == 0:
        return []

    # [x1, y1, x2, y2, score] -> [x1, y1, w, h, score]
    nms_bboxes[:, 2] = nms_bboxes[:, 2] - nms_bboxes[:, 0] + 1
    nms_bboxes[:, 3] = nms_bboxes[:, 3] - nms_bboxes[:, 1] + 1
    # each row : [class_label, score, x1, y1, w, h]
    results = torch.cat([nms_labels[:, None].to(nms_bboxes), nms_bboxes[:, [4, 0, 1, 2, 3]]], dim=1)
    # from tensor to list
    results = results.tolist()
    results = [[int(temp_result[0])] + temp_result[1:] for temp_result in results]

    return results


classification_threshold = 0.5
nms_threshold = 0.3
image_path = './test-imgs'

if os.path.isfile(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    results = predict(config_dict['model'], image, aug_pipeline=val_pipeline, classification_threshold=classification_threshold, nms_threshold=nms_threshold, class_agnostic=True)

    for bbox in results:
        print(bbox)
        cv2.rectangle(image, (int(bbox[2]), int(bbox[3])), (int(bbox[2] + bbox[4]), int(bbox[3] + bbox[5])), (0, 255, 0), 2)
    print('%d lights are detected!' % len(results))
    cv2.imshow('im', image)
    cv2.waitKey()

    # cv2.imwrite(os.path.join(os.path.dirname(image_path), os.path.basename(image_path).replace('.jpg', '_result.jpg')), image)
else:
    image_path_list = [os.path.join(image_path, file_name) for file_name in os.listdir(image_path) if file_name.endswith(('jpg', 'png'))]

    for image_path in image_path_list:
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        results = predict(config_dict['model'], image, aug_pipeline=val_pipeline, classification_threshold=classification_threshold, nms_threshold=nms_threshold, class_agnostic=True)

        for bbox in results:
            print(bbox)
            cv2.rectangle(image, (int(bbox[2]), int(bbox[3])), (int(bbox[2] + bbox[4]), int(bbox[3] + bbox[5])), (0, 255, 0), 2)
        print('%d lights are detected!' % len(results))
        cv2.imshow('im', image)
        cv2.waitKey()

        # cv2.imwrite(os.path.join(os.path.dirname(image_path), ''.join(os.path.basename(image_path).split('.')[:-1]) + '_result.jpg'), image)
