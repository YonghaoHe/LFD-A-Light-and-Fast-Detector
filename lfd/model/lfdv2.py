# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy
import cv2
import math
from ..data_pipeline.dataset import Sample
from .utils import multiclass_nms
import pycuda.driver as cuda

__all__ = ['LFDv2']


def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6):
    """Calculate overlap between two set of bboxes.

    If ``is_aligned `` is ``False``, then calculate the overlaps between each
    bbox of bboxes1 and bboxes2, otherwise the overlaps between each aligned
    pair of bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (B, m, 4) in <x1, y1, x2, y2> format or empty.
        bboxes2 (Tensor): shape (B, n, 4) in <x1, y1, x2, y2> format or empty.
            B indicates the batch dim, in shape (B1, B2, ..., Bn).
            If ``is_aligned `` is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union), "iof" (intersection over
            foreground) or "giou" (generalized intersection over union).
            Default "iou".
        is_aligned (bool, optional): If True, then m and n must be equal.
            Default False.
        eps (float, optional): A value added to the denominator for numerical
            stability. Default 1e-6.

    Returns:
        Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)

    Example:
        >>> bboxes1 = torch.FloatTensor([
        >>>     [0, 0, 10, 10],
        >>>     [10, 10, 20, 20],
        >>>     [32, 32, 38, 42],
        >>> ])
        >>> bboxes2 = torch.FloatTensor([
        >>>     [0, 0, 10, 20],
        >>>     [0, 10, 10, 19],
        >>>     [10, 10, 20, 20],
        >>> ])
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2)
        >>> assert overlaps.shape == (3, 3)
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2, is_aligned=True)
        >>> assert overlaps.shape == (3, )

    Example:
        >>> empty = torch.empty(0, 4)
        >>> nonempty = torch.FloatTensor([[0, 0, 10, 9]])
        >>> assert tuple(bbox_overlaps(empty, nonempty).shape) == (0, 1)
        >>> assert tuple(bbox_overlaps(nonempty, empty).shape) == (1, 0)
        >>> assert tuple(bbox_overlaps(empty, empty).shape) == (0, 0)
    """

    assert mode in ['iou', 'iof', 'giou'], f'Unsupported mode {mode}'
    # Either the boxes are empty or the length of boxes's last dimenstion is 4
    assert (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0)

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows,))
        else:
            return bboxes1.new(batch_shape + (rows, cols))

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (
            bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (
            bboxes2[..., 3] - bboxes2[..., 1])

    if is_aligned:
        lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])  # [B, rows, 2]
        rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, rows, 2]

        wh = (rb - lt).clamp(min=0)  # [B, rows, 2]
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
    else:
        lt = torch.max(bboxes1[..., :, None, :2],
                       bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
        rb = torch.min(bboxes1[..., :, None, 2:],
                       bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]

        wh = (rb - lt).clamp(min=0)  # [B, rows, cols, 2]
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            union = area1[..., None]
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :, None, :2],
                                    bboxes2[..., None, :, :2])
            enclosed_rb = torch.max(bboxes1[..., :, None, 2:],
                                    bboxes2[..., None, :, 2:])

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union
    if mode in ['iou', 'iof']:
        return ious
    # calculate gious
    enclose_wh = (enclosed_rb - enclosed_lt).clamp(min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    enclose_area = torch.max(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious


class LFDv2(nn.Module):

    def __init__(self,
                 backbone=None,
                 neck=None,
                 head=None,
                 num_classes=80,
                 regression_ranges=((0, 64), (64, 128), (128, 256), (256, 512), (512, 1024)),
                 gray_range_factors=(0.9, 1.1),
                 range_assign_mode='longer',  # determine how to assign bbox to which range
                 point_strides=(8, 16, 32, 64, 128),
                 classification_loss_func=None,
                 regression_loss_func=None,
                 distance_to_bbox_mode='exp',
                 enable_classification_weight=False,
                 enable_regression_weight=False,
                 classification_threshold=0.05,
                 nms_threshold=0.5,
                 pre_nms_bbox_limit=1000,
                 post_nms_bbox_limit=100,
                 ):
        super(LFDv2, self).__init__()
        assert len(regression_ranges) == len(point_strides)
        assert range_assign_mode in ['longer', 'shorter', 'sqrt', 'dist']
        assert distance_to_bbox_mode in ['exp', 'sigmoid']

        self._backbone = backbone
        self._neck = neck
        self._head = head
        self._num_classes = num_classes
        self._regression_ranges = regression_ranges
        self._range_assign_mode = range_assign_mode
        if self._range_assign_mode in ['shorter', 'sqrt']:
            assert type(regression_loss_func).__name__ in ['IoULoss', 'GIoULoss', 'DIoULoss', 'CIoULoss'], 'when range assign mode is "shorter" or "sqrt", regression loss should be IOU losses!'
            assert distance_to_bbox_mode == 'exp', 'when range assign mode is "shorter" or "sqrt", distance_to_bbox_mode must be "exp"!'

        self._gray_range_factors = (min(gray_range_factors), max(gray_range_factors))
        self._gray_ranges = [(int(low * self._gray_range_factors[0]), int(up * self._gray_range_factors[1])) for (low, up) in self._regression_ranges]
        self._num_heads = len(point_strides)
        self._point_strides = point_strides

        # currently, classification losses support BCEWithLogitsLoss, CrossEntropyLoss, FocalLoss, QualityFocalLoss
        # we find that FocalLoss is not suitable for train-from-scratch
        if classification_loss_func is not None:
            assert type(classification_loss_func).__name__ in ['BCEWithLogitsLoss', 'FocalLoss', 'CrossEntropyLoss', 'QualityFocalLoss']
        self._classification_loss_func = classification_loss_func

        # currently, regression losses support SmoothL1Loss, MSELoss, IoULoss, GIoULoss, DIoULoss, CIoULoss
        # regression losses are divided into two categories: independent(SmoothL1Loss, MSELoss) and
        # union(IoULoss, GIoULoss, DIoULoss, CIoULoss)
        if regression_loss_func is not None:
            assert type(regression_loss_func).__name__ in ['SmoothL1Loss', 'MSELoss', 'IoULoss', 'GIoULoss', 'DIoULoss', 'CIoULoss']
            if type(regression_loss_func).__name__ in ['SmoothL1Loss', 'MSELoss']:
                self._regression_loss_type = 'independent'
            else:
                self._regression_loss_type = 'union'
        self._regression_loss_func = regression_loss_func
        assert distance_to_bbox_mode in ['exp', 'sigmoid']
        self._distance_to_bbox_mode = distance_to_bbox_mode
        self._enable_classification_weight = enable_classification_weight
        self._enable_regression_weight = enable_regression_weight

        self._classification_threshold = classification_threshold
        self._nms_cfg = dict(type='nms', iou_thr=nms_threshold)
        self._pre_nms_bbox_limit = pre_nms_bbox_limit
        self._post_nms_bbox_limit = post_nms_bbox_limit

        self._head_indexes_to_feature_map_sizes = dict()

    @property
    def head_indexes_to_feature_map_sizes(self):
        return self._head_indexes_to_feature_map_sizes

    def generate_point_coordinates(self, feature_map_sizes):
        """
        transform feature map points to locations in original input image
        :param feature_map_sizes:
        :return:
        """

        def generate_for_single_feature_map(func_feature_map_size, func_stride):
            height, width = func_feature_map_size
            x_coordinates = torch.arange(0, width * func_stride, func_stride)
            y_coordinates = torch.arange(0, height * func_stride, func_stride)

            y_mesh, x_mesh = torch.meshgrid(y_coordinates, x_coordinates)

            point_coordinates = torch.stack((x_mesh.reshape(-1), y_mesh.reshape(-1)), dim=-1)

            return point_coordinates

        assert len(feature_map_sizes) == len(self._point_strides)
        all_point_coordinates_list = []
        for i in range(len(self._point_strides)):
            all_point_coordinates_list.append(generate_for_single_feature_map(feature_map_sizes[i], self._point_strides[i]))

        return all_point_coordinates_list

    def annotation_to_target(self, all_point_coordinates_list, gt_bboxes_list, gt_labels_list, *args):

        expanded_regression_ranges_list = [all_point_coordinates_list[i].new_tensor(self._regression_ranges[i])[None].expand_as(all_point_coordinates_list[i])
                                           for i in range(self._num_heads)]
        expanded_gray_ranges_list = [all_point_coordinates_list[i].new_tensor(self._gray_ranges[i])[None].expand_as(all_point_coordinates_list[i])
                                     for i in range(self._num_heads)]
        expanded_strides_list = [all_point_coordinates_list[i].new_tensor(self._point_strides[i]).expand(all_point_coordinates_list[i].size(0))
                                 for i in range(self._num_heads)]

        concat_point_coordinates = torch.cat(all_point_coordinates_list, dim=0)
        concat_regression_ranges = torch.cat(expanded_regression_ranges_list, dim=0)
        concat_gray_ranges = torch.cat(expanded_gray_ranges_list, dim=0)
        concat_strides = torch.cat(expanded_strides_list, dim=0)

        classification_targets_list = list()
        regression_targets_list = list()
        for i, gt_bboxes in enumerate(gt_bboxes_list):
            temp_classification_targets, temp_regression_targets = self._generate_target_for_single_image(gt_bboxes=gt_bboxes,
                                                                                                          gt_labels=gt_labels_list[i],
                                                                                                          concat_point_coordinates=concat_point_coordinates,
                                                                                                          concat_regression_ranges=concat_regression_ranges,
                                                                                                          concat_gray_ranges=concat_gray_ranges,
                                                                                                          concat_strides=concat_strides)

            # display for debug---------------------------------------------------------------------------------------
            # split_classification_targets_list = temp_classification_targets.split([coord.size(0) for coord in all_point_coordinates_list], dim=0)
            # split_classification_targets_list = [target.reshape(int(math.sqrt(target.size(0))), int(math.sqrt(target.size(0))), -1) for target in split_classification_targets_list]
            #
            # display_class_label = 1
            # for j, target in enumerate(split_classification_targets_list):
            #     display_map = target[..., display_class_label].numpy()
            #     display_map = display_map * 255
            #     display_map[display_map < 0] = 127
            #     display_map = display_map.astype(dtype=numpy.uint8)
            #
            #     cv2.imshow(str(j), display_map)
            # cv2.waitKey()
            # ---------------------------------------------------------------------------------------------------------

            classification_targets_list.append(temp_classification_targets)
            regression_targets_list.append(temp_regression_targets)

        stack_classification_targets_tensor = torch.stack(classification_targets_list, dim=0)
        stack_regression_targets_tensor = torch.stack(regression_targets_list, dim=0)
        return stack_classification_targets_tensor, stack_regression_targets_tensor

    def _generate_target_for_single_image(self,
                                          gt_bboxes,
                                          gt_labels,
                                          concat_point_coordinates,
                                          concat_regression_ranges,
                                          concat_gray_ranges,
                                          concat_strides):

        assert gt_bboxes.size(0) == gt_labels.size(0)

        num_points = concat_point_coordinates.size(0)
        num_gt_bboxes = gt_bboxes.size(0)

        classification_targets = gt_bboxes.new_full((num_points, self._num_classes), 0)
        regression_targets = gt_bboxes.new_zeros((num_points, 4))

        if num_gt_bboxes == 0:
            return classification_targets, regression_targets

        gt_bboxes = gt_bboxes[None].expand(num_points, num_gt_bboxes, 4)
        gt_labels = gt_labels[None].expand(num_points, num_gt_bboxes)

        concat_regression_ranges = concat_regression_ranges[:, None, :].expand(num_points, num_gt_bboxes, 2)
        concat_gray_ranges = concat_gray_ranges[:, None, :].expand(num_points, num_gt_bboxes, 2)

        point_x_coordinates, point_y_corrdinates = concat_point_coordinates[:, 0], concat_point_coordinates[:, 1]
        point_x_coordinates = point_x_coordinates[:, None].expand(num_points, num_gt_bboxes)
        point_y_corrdinates = point_y_corrdinates[:, None].expand(num_points, num_gt_bboxes)
        gt_bboxes_center_x = gt_bboxes[..., 0] + gt_bboxes[..., 2] / 2.
        gt_bboxes_center_y = gt_bboxes[..., 1] + gt_bboxes[..., 3] / 2.

        concat_strides = concat_strides[:, None]

        # calculate regression values
        delta_x1 = point_x_coordinates - gt_bboxes[..., 0]
        delta_y1 = point_y_corrdinates - gt_bboxes[..., 1]
        delta_x2 = (gt_bboxes[..., 0] + gt_bboxes[..., 2] - 1) - point_x_coordinates
        delta_y2 = (gt_bboxes[..., 1] + gt_bboxes[..., 3] - 1) - point_y_corrdinates
        regression_delta = torch.stack((delta_x1, delta_y1, delta_x2, delta_y2), dim=-1)  # distance to left, top, right, bottom
        hit_condition = regression_delta.min(dim=-1)[0] >= 0

        # calculate scores in [0, 1] for classification
        # the closer the point near the center, the higher score it will get
        # abs_to_center_x = torch.abs(point_x_coordinates - gt_bboxes_center_x)
        # abs_to_center_y = torch.abs(point_y_corrdinates - gt_bboxes_center_y)
        # x_scores = abs_to_center_x / (concat_strides / 2.)
        # x_scores = x_scores * (x_scores >= 1) + (x_scores < 1)
        # x_scores = torch.sqrt(1. / x_scores)
        # y_scores = abs_to_center_y / (concat_strides / 2.)
        # y_scores = y_scores * (y_scores >= 1) + (y_scores < 1)
        # y_scores = torch.sqrt(1. / y_scores)
        # point_scores = x_scores * y_scores  # P x N

        filtered_regression_delta = regression_delta * hit_condition[..., None].expand((num_points, num_gt_bboxes, 4))  # filter cases that points do not hit
        left_right = filtered_regression_delta[..., [0, 2]]
        top_bottom = filtered_regression_delta[..., [1, 3]]
        # clamp to avoid zero-divisor
        point_scores = ((left_right.min(dim=-1)[0]).clamp(min=0.0) / (left_right.max(dim=-1)[0]).clamp(min=0.01)) * \
                       ((top_bottom.min(dim=-1)[0]).clamp(min=0.0) / (top_bottom.max(dim=-1)[0]).clamp(min=0.01))
        point_scores = torch.sqrt(point_scores)

        strides = concat_strides.expand((num_points, num_gt_bboxes))
        core_zone_left = gt_bboxes_center_x - strides / 2
        core_zone_right = gt_bboxes_center_x + strides / 2
        core_zone_top = gt_bboxes_center_y - strides / 2
        core_zone_bottom = gt_bboxes_center_y + strides / 2

        inside_core_zone = (point_x_coordinates >= core_zone_left) & (point_x_coordinates <= core_zone_right) & \
                           (point_y_corrdinates >= core_zone_top) & (point_y_corrdinates <= core_zone_bottom)
        inside_core_zone = inside_core_zone & hit_condition  # in case that the point is out of bbox
        point_scores = point_scores * (~inside_core_zone) + inside_core_zone  # assign 1 to the scores of points that in core zone

        # determine pos/gray/neg points P x N
        # compute determine side according to range_assign_mode
        if self._range_assign_mode == 'longer':
            assign_measure = torch.max(gt_bboxes[..., 2], gt_bboxes[..., 3])
        elif self._range_assign_mode == 'shorter':
            assign_measure = torch.min(gt_bboxes[..., 2], gt_bboxes[..., 3])
        elif self._range_assign_mode == 'sqrt':
            assign_measure = torch.sqrt(gt_bboxes[..., 2] * gt_bboxes[..., 3])
        elif self._range_assign_mode == 'dist':
            assign_measure = regression_delta.max(dim=-1)[0]
        else:
            raise ValueError('Unsupported range assign mode!')

        if self._regression_loss_type == 'independent':
            regression_delta = regression_delta / concat_regression_ranges[..., 1, None]  # P x N x 4

        head_selection_condition = (concat_regression_ranges[..., 0] <= assign_measure) & (assign_measure <= concat_regression_ranges[..., 1])

        green_condition = head_selection_condition & hit_condition

        gray_condition1 = (concat_gray_ranges[..., 0] <= assign_measure) & (assign_measure < concat_regression_ranges[..., 0])
        gray_condition2 = (concat_regression_ranges[..., 1] < assign_measure) & (assign_measure <= concat_gray_ranges[..., 1])
        gray_condition = (gray_condition1 | gray_condition2) & hit_condition

        # rank scores in ascending order for each point
        # why rank here: for a certain class, multiple objects may cover the same point, putting the largest score at the end will make
        # the classification_targets assigned with this largest score.
        # 对于单个类别来说，某个point可能落入多个这个类别目标的bbox中，那classification target应该被赋值为这个类别最大的那个得分，所以这里通过把
        # 最大的分数排在最后来实现的，具体影响的代码是：classification_targets[index1, green_label_index] = sorted_point_scores[index1, index2]
        sorted_point_scores, sorted_indexes = point_scores.sort(dim=1)
        intermediate_indexes = sorted_indexes.new_tensor(range(sorted_indexes.size(0)))[..., None].expand(sorted_indexes.size(0), sorted_indexes.size(1))

        # reranking
        sorted_gt_labels = gt_labels[intermediate_indexes, sorted_indexes]
        sorted_green_condition = green_condition[intermediate_indexes, sorted_indexes]
        sorted_gray_condition = gray_condition[intermediate_indexes, sorted_indexes]

        # set green positions
        index1, index2 = torch.where(sorted_green_condition)
        green_label_index = sorted_gt_labels[index1, index2]
        classification_targets[index1, green_label_index] = sorted_point_scores[index1, index2]

        # set gray positions
        index3, index4 = torch.where(sorted_gray_condition)
        gray_label_index = sorted_gt_labels[index3, index4]
        classification_targets[index3, gray_label_index] = -1

        # for each point, select the regression target with the highest score (affected by green and gray conditions)
        filtered_sorted_point_scores = sorted_point_scores * (sorted_green_condition & ~sorted_gray_condition)
        _, select_indexes = filtered_sorted_point_scores.max(dim=1)
        sorted_regression_delta = regression_delta[intermediate_indexes, sorted_indexes]
        regression_targets = sorted_regression_delta[range(num_points), select_indexes]

        return classification_targets, regression_targets

    def distance2bbox(self, points, distance, max_shape=None):
        """Decode distance prediction to bounding box.

        Args:
            points (Tensor): Shape (n, 2), [x, y].
            distance (Tensor): Distance from the given point to 4
                boundaries (left, top, right, bottom).
            max_shape (tuple): Shape of the image.

        Returns:
            Tensor: Decoded bboxes.
        """
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        if max_shape is not None:
            x1 = x1.clamp(min=0, max=max_shape[1])
            y1 = y1.clamp(min=0, max=max_shape[0])
            x2 = x2.clamp(min=0, max=max_shape[1])
            y2 = y2.clamp(min=0, max=max_shape[0])
        return torch.stack([x1, y1, x2, y2], -1)

    def get_loss(self, predict_outputs, annotation_batch, *args):

        predict_classification_tensor, predict_regression_tensor = predict_outputs

        gt_bboxes_list, gt_labels_list = list(), list()
        for annotation in annotation_batch:
            bboxes_numpy, labels_numpy = annotation
            gt_bboxes_list.append(torch.from_numpy(bboxes_numpy))
            gt_labels_list.append(torch.from_numpy(labels_numpy))

        # 获取所有level上的feature map locations在原图中的坐标位置
        all_point_coordinates_list = self.generate_point_coordinates(self._head_indexes_to_feature_map_sizes)

        # 进行annotation 到 target 的转换
        classification_target_tensor, regression_target_tensor = self.annotation_to_target(all_point_coordinates_list, gt_bboxes_list, gt_labels_list)

        batch_size = predict_classification_tensor.size(0)
        # CAUTION: the number of channels for CrossEntropyLoss is num_classes + 1 (additional one channel for bg)
        if type(self._classification_loss_func).__name__ == 'CrossEntropyLoss':
            flatten_predict_classification_tensor = predict_classification_tensor.reshape(-1, self._num_classes + 1)
        else:
            flatten_predict_classification_tensor = predict_classification_tensor.reshape(-1, self._num_classes)
        flatten_predict_regression_tensor = predict_regression_tensor.reshape(-1, 4)
        flatten_classification_target_tensor = classification_target_tensor.reshape(-1, self._num_classes)  # (N*P,C)
        flatten_regression_target_tensor = regression_target_tensor.reshape(-1, 4)  # (N*P, 4)

        flatten_classification_target_tensor = flatten_classification_target_tensor.to(flatten_predict_classification_tensor.device)
        flatten_regression_target_tensor = flatten_regression_target_tensor.to(flatten_predict_regression_tensor.device)

        # ignore gray positions
        min_scores = flatten_classification_target_tensor.min(dim=-1)[0]
        green_indexes = torch.where(min_scores >= 0)[0]

        flatten_predict_classification_tensor = flatten_predict_classification_tensor[green_indexes]
        flatten_predict_regression_tensor = flatten_predict_regression_tensor[green_indexes]
        flatten_classification_target_tensor = flatten_classification_target_tensor[green_indexes]
        flatten_regression_target_tensor = flatten_regression_target_tensor[green_indexes]

        max_scores, max_score_indexes = flatten_classification_target_tensor.max(dim=-1)
        pos_indexes = torch.where(max_scores >= 0.001)[0]
        weight = max_scores[pos_indexes]
        # targets for FocalLoss/CrossEntropyLoss are label indexes
        if type(self._classification_loss_func).__name__ in ['FocalLoss', 'CrossEntropyLoss', 'QualityFocalLoss']:
            # assign background label
            flatten_classification_target_label_tensor = max_score_indexes * (max_scores >= 0.001) + self._num_classes * (max_scores < 0.001)
            flatten_classification_target_score_tensor = max_scores
            if type(self._classification_loss_func).__name__ == 'QualityFocalLoss':
                # get classification loss
                classification_loss = self._classification_loss_func(flatten_predict_classification_tensor,
                                                                     [flatten_classification_target_label_tensor, flatten_classification_target_score_tensor],
                                                                     avg_factor=weight.sum() if self._enable_classification_weight else pos_indexes.nelement() + 1,
                                                                     )
            else:
                # get classification loss
                classification_loss = self._classification_loss_func(flatten_predict_classification_tensor,
                                                                     flatten_classification_target_label_tensor,
                                                                     avg_factor=weight.sum() if self._enable_classification_weight else pos_indexes.nelement() + 1,
                                                                     )
        else:  # BCEWithLogitsLoss
            # get classification loss
            classification_loss = self._classification_loss_func(flatten_predict_classification_tensor,
                                                                 flatten_classification_target_tensor,
                                                                 avg_factor=weight.sum() if self._enable_classification_weight else pos_indexes.nelement() + 1,
                                                                 )

        # get regression loss
        flatten_predict_regression_tensor = flatten_predict_regression_tensor[pos_indexes]
        flatten_regression_target_tensor = flatten_regression_target_tensor[pos_indexes]

        if pos_indexes.nelement() > 0:
            if self._regression_loss_type == 'independent':
                regression_loss = self._regression_loss_func(flatten_predict_regression_tensor,
                                                             flatten_regression_target_tensor,
                                                             avg_factor=weight.sum() if self._enable_regression_weight else pos_indexes.nelement(),
                                                             weight=weight if self._enable_regression_weight else None)
            else:
                flatten_all_point_coordinates = (torch.cat(all_point_coordinates_list, dim=0)).repeat(batch_size, 1)
                flatten_all_point_coordinates = flatten_all_point_coordinates.to(flatten_predict_regression_tensor.device)
                flatten_all_point_coordinates = flatten_all_point_coordinates[green_indexes][pos_indexes]

                flatten_xyxy_regression_target_tensor = self.distance2bbox(flatten_all_point_coordinates, flatten_regression_target_tensor)

                if self._distance_to_bbox_mode == 'exp':
                    flatten_predict_regression_tensor = flatten_predict_regression_tensor.float().exp()
                    flatten_xyxy_predict_regression_tensor = self.distance2bbox(flatten_all_point_coordinates, flatten_predict_regression_tensor)
                elif self._distance_to_bbox_mode == 'sigmoid':
                    expanded_regression_ranges_list = [all_point_coordinates_list[i].new_tensor(self._regression_ranges[i])[None].expand_as(all_point_coordinates_list[i])
                                                       for i in range(self._num_heads)]
                    concat_regression_ranges = (torch.cat(expanded_regression_ranges_list, dim=0)).repeat(batch_size, 1)
                    concat_regression_ranges = concat_regression_ranges.to(flatten_predict_regression_tensor.device)
                    concat_regression_ranges = concat_regression_ranges[green_indexes][pos_indexes]
                    concat_regression_ranges_max = concat_regression_ranges.max(dim=-1)[0]
                    flatten_predict_regression_tensor = flatten_predict_regression_tensor.sigmoid() * concat_regression_ranges_max[..., None]
                    flatten_xyxy_predict_regression_tensor = self.distance2bbox(flatten_all_point_coordinates, flatten_predict_regression_tensor)
                else:
                    raise ValueError('Unknown distance_to_bbox mode!')

                regression_loss = self._regression_loss_func(flatten_xyxy_predict_regression_tensor,
                                                             flatten_xyxy_regression_target_tensor,
                                                             avg_factor=weight.sum() if self._enable_regression_weight else pos_indexes.nelement(),
                                                             weight=weight if self._enable_regression_weight else None)

        else:
            regression_loss = flatten_predict_regression_tensor.sum()

        loss = classification_loss + regression_loss
        loss_values = dict(loss=loss.item(),
                           classification_loss=classification_loss.item(),
                           regression_loss=regression_loss.item())

        return dict(loss=loss,
                    loss_values=loss_values)

    def get_results(self, predict_outputs, *args):
        """
        for online evaluation
        :param predict_outputs:
        :param args:
        :return:
        """
        predict_classification_tensor, predict_regression_tensor = predict_outputs
        num_samples = predict_classification_tensor.size(0)
        all_point_coordinates_list = self.generate_point_coordinates(self._head_indexes_to_feature_map_sizes)
        expanded_regression_ranges_list = [all_point_coordinates_list[i].new_tensor(self._regression_ranges[i])[None].expand_as(all_point_coordinates_list[i])
                                           for i in range(self._num_heads)]
        meta_batch = args[0]

        results = []
        for i in range(num_samples):
            nms_bboxes, nms_labels = self._get_results_for_single_image(predict_classification_tensor[i],
                                                                        predict_regression_tensor[i],
                                                                        all_point_coordinates_list,
                                                                        expanded_regression_ranges_list,
                                                                        meta_batch[i])
            if nms_bboxes.size(0) == 0:
                results.append([])
                continue

            # [x1, y1, x2, y2, score] -> [x1, y1, w, h, score]
            nms_bboxes[:, 2] = nms_bboxes[:, 2] - nms_bboxes[:, 0] + 1
            nms_bboxes[:, 3] = nms_bboxes[:, 3] - nms_bboxes[:, 1] + 1
            # each row : [class_label, score, x1, y1, w, h]
            temp_results = torch.cat([nms_labels[:, None].to(nms_bboxes), nms_bboxes[:, [4, 0, 1, 2, 3]]], dim=1)
            # from tensor to list
            temp_results = temp_results.tolist()
            temp_results = [[int(temp_result[0])] + temp_result[1:] for temp_result in temp_results]

            results.append(temp_results)
        return results

    def _get_results_for_single_image(self,
                                      predicted_classification,
                                      predicted_regression,
                                      all_point_coordinates_list,
                                      expanded_regression_ranges_list,
                                      meta_info):
        split_list = [point_coordinates_per_level.size(0) for point_coordinates_per_level in all_point_coordinates_list]
        predicted_classification_split = predicted_classification.split(split_list, dim=0)
        predicted_regression_split = predicted_regression.split(split_list, dim=0)
        image_resized_height = meta_info['resized_height']
        image_resized_width = meta_info['resized_width']

        predicted_classification_merge = list()
        predicted_bboxes_merge = list()

        for i in range(len(split_list)):
            if type(self._classification_loss_func).__name__ in ['CrossEntropyLoss']:
                temp_predicted_classification = predicted_classification_split[i].softmax(dim=1)
                temp_predicted_classification = temp_predicted_classification[:, :-1]  # remove bg
            else:
                temp_predicted_classification = predicted_classification_split[i].sigmoid()
            temp_predicted_regression = predicted_regression_split[i]
            temp_point_coordinates = all_point_coordinates_list[i].to(temp_predicted_regression.device)
            temp_expanded_regression_ranges = expanded_regression_ranges_list[i].to(temp_predicted_regression.device)

            if 0 < self._pre_nms_bbox_limit < temp_predicted_classification.size(0):
                temp_max_scores = temp_predicted_classification.max(dim=1)[0]
                topk_indexes = temp_max_scores.topk(self._pre_nms_bbox_limit)[1]
                temp_predicted_classification = temp_predicted_classification[topk_indexes]
                temp_predicted_regression = temp_predicted_regression[topk_indexes]
                temp_point_coordinates = temp_point_coordinates[topk_indexes]
                temp_expanded_regression_ranges = temp_expanded_regression_ranges[topk_indexes]

            #  calculate bboxes' x1 y1 x2 y2
            if self._regression_loss_type == 'independent':
                temp_predicted_regression = temp_predicted_regression * temp_expanded_regression_ranges[..., 1, None]
                x1 = temp_point_coordinates[:, 0] - temp_predicted_regression[:, 0]
                x1 = x1.clamp(min=0, max=image_resized_width)
                y1 = temp_point_coordinates[:, 1] - temp_predicted_regression[:, 1]
                y1 = y1.clamp(min=0, max=image_resized_height)
                x2 = temp_point_coordinates[:, 0] + temp_predicted_regression[:, 2]
                x2 = x2.clamp(min=0, max=image_resized_width)
                y2 = temp_point_coordinates[:, 1] + temp_predicted_regression[:, 3]
                y2 = y2.clamp(min=0, max=image_resized_height)
                temp_bboxes = torch.stack([x1, y1, x2, y2], -1)
            else:
                if self._distance_to_bbox_mode == 'exp':
                    temp_predicted_regression = temp_predicted_regression.float().exp()
                    temp_bboxes = self.distance2bbox(temp_point_coordinates, temp_predicted_regression, max_shape=(image_resized_height, image_resized_width))
                elif self._distance_to_bbox_mode == 'sigmoid':
                    temp_expanded_regression_ranges_max = temp_expanded_regression_ranges.max(dim=-1)[0]
                    temp_predicted_regression = temp_predicted_regression.sigmoid() * temp_expanded_regression_ranges_max[..., None]
                    temp_bboxes = self.distance2bbox(temp_point_coordinates, temp_predicted_regression, max_shape=(image_resized_height, image_resized_width))
                else:
                    raise ValueError('Unknown distance_to_bbox mode!')

            predicted_classification_merge.append(temp_predicted_classification)
            predicted_bboxes_merge.append(temp_bboxes)

        predicted_classification_merge = torch.cat(predicted_classification_merge)
        # add BG label for multi class nms
        bg_label_padding = predicted_classification_merge.new_zeros(predicted_classification_merge.size(0), 1)
        predicted_classification_merge = torch.cat([predicted_classification_merge, bg_label_padding], dim=1)

        predicted_bboxes_merge = torch.cat(predicted_bboxes_merge)
        predicted_bboxes_merge = predicted_bboxes_merge / meta_info['resize_scale']

        nms_bboxes, nms_labels = multiclass_nms(
            multi_bboxes=predicted_bboxes_merge,
            multi_scores=predicted_classification_merge,
            score_thr=self._classification_threshold,
            nms_cfg=self._nms_cfg,
            max_num=self._post_nms_bbox_limit,
            score_factors=None
        )

        return nms_bboxes, nms_labels

    def forward(self, x):

        backbone_outputs = self._backbone(x)

        neck_outputs = self._neck(backbone_outputs)

        head_outputs = self._head(neck_outputs)  # in case of outputs > 2, like fcos head (cls, reg, centerness)

        classification_outputs = head_outputs[0]
        regression_outputs = head_outputs[1]

        #  变换输出的dim和shape，转化成tensor输出
        #  tensor 中的dim n必须要保留，为了DP能够正常多卡
        classification_reformat_outputs = []
        regression_reformat_outputs = []
        for i, classification_output in enumerate(classification_outputs):
            n, c, h, w = classification_output.shape
            classification_output = classification_output.permute([0, 2, 3, 1])
            classification_output = classification_output.reshape((n, h * w, c))
            classification_reformat_outputs.append(classification_output)

            self._head_indexes_to_feature_map_sizes[i] = (h, w)

            n, c, h, w = regression_outputs[i].shape
            regression_output = regression_outputs[i].permute([0, 2, 3, 1])
            regression_output = regression_output.reshape((n, h * w, c))
            regression_reformat_outputs.append(regression_output)

        classification_output_tensor = torch.cat(classification_reformat_outputs, dim=1)
        regression_output_tensor = torch.cat(regression_reformat_outputs, dim=1)

        return classification_output_tensor, regression_output_tensor

    def predict_for_single_image(self, image, aug_pipeline, classification_threshold=None, nms_threshold=None, class_agnostic=False):
        """
        for easy prediction
        :param image: image can be string path or numpy array
        :param aug_pipeline: image pre-processing like flip, normalization
        :param classification_threshold: higher->higher precision, lower->higher recall
        :param nms_threshold:
        :param class_agnostic:
        """
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
        data_batch = data_batch.cuda()
        self.cuda()
        self.eval()

        with torch.no_grad():
            predicted_classification, predicted_regression = self.forward(data_batch)
        predicted_classification = predicted_classification[0]
        predicted_regression = predicted_regression[0]

        all_point_coordinates_list = self.generate_point_coordinates(self._head_indexes_to_feature_map_sizes)
        expanded_regression_ranges_list = [all_point_coordinates_list[i].new_tensor(self._regression_ranges[i])[None].expand_as(all_point_coordinates_list[i])
                                           for i in range(self._num_heads)]

        concat_point_coordinates = torch.cat(all_point_coordinates_list, dim=0)
        concat_regression_ranges = torch.cat(expanded_regression_ranges_list, dim=0)

        if type(self._classification_loss_func).__name__ in ['CrossEntropyLoss']:
            predicted_classification = predicted_classification.softmax(dim=1)
            predicted_classification = predicted_classification[:, :-1]  # remove bg
        else:
            predicted_classification = predicted_classification.sigmoid()

        concat_point_coordinates = concat_point_coordinates.to(predicted_regression.device)
        concat_regression_ranges = concat_regression_ranges.to(predicted_regression.device)

        classification_threshold = classification_threshold if classification_threshold is not None else self._classification_threshold
        max_scores = predicted_classification.max(dim=1)[0]
        selected_indexes = torch.where(max_scores > classification_threshold)[0]
        if selected_indexes.numel() == 0:
            return []

        predicted_classification = predicted_classification[selected_indexes]
        predicted_regression = predicted_regression[selected_indexes]
        concat_point_coordinates = concat_point_coordinates[selected_indexes]
        concat_regression_ranges = concat_regression_ranges[selected_indexes]

        #  calculate bboxes' x1 y1 x2 y2
        if self._regression_loss_type == 'independent':
            predicted_regression = predicted_regression * concat_regression_ranges[..., 1, None]
            x1 = concat_point_coordinates[:, 0] - predicted_regression[:, 0]
            x1 = x1.clamp(min=0, max=image_width)
            y1 = concat_point_coordinates[:, 1] - predicted_regression[:, 1]
            y1 = y1.clamp(min=0, max=image_height)
            x2 = concat_point_coordinates[:, 0] + predicted_regression[:, 2]
            x2 = x2.clamp(min=0, max=image_width)
            y2 = concat_point_coordinates[:, 1] + predicted_regression[:, 3]
            y2 = y2.clamp(min=0, max=image_height)
            predicted_bboxes = torch.stack([x1, y1, x2, y2], -1)
        else:
            if self._distance_to_bbox_mode == 'exp':
                predicted_regression = predicted_regression.float().exp()
                predicted_bboxes = self.distance2bbox(concat_point_coordinates, predicted_regression, max_shape=(image_height, image_width))
            elif self._distance_to_bbox_mode == 'sigmoid':
                concat_regression_ranges_max = concat_regression_ranges.max(dim=-1)[0]
                predicted_regression = predicted_regression.sigmoid() * concat_regression_ranges_max[..., None]
                predicted_bboxes = self.distance2bbox(concat_point_coordinates, predicted_regression, max_shape=(image_height, image_width))
            else:
                raise ValueError('Unknown distance_to_bbox mode!')
        # add BG label for multi class nms
        bg_label_padding = predicted_classification.new_zeros(predicted_classification.size(0), 1)
        predicted_classification = torch.cat([predicted_classification, bg_label_padding], dim=1)

        if nms_threshold:
            self._nms_cfg.update({'iou_thr': nms_threshold})
        if class_agnostic:
            self._nms_cfg.update({'class_agnostic': class_agnostic})
        nms_bboxes, nms_labels = multiclass_nms(
            multi_bboxes=predicted_bboxes,
            multi_scores=predicted_classification,
            score_thr=classification_threshold,
            nms_cfg=self._nms_cfg,
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

    def predict_for_single_image_with_tensorrt(self,
                                               image,
                                               input_buffers,
                                               output_buffers,
                                               bindings,
                                               stream,
                                               engine,
                                               tensorrt_engine_context,
                                               aug_pipeline,
                                               classification_threshold=None,
                                               nms_threshold=None,
                                               class_agnostic=False):
        """
        for easy prediction, using tensorrt as inference engine instead
        :param image: image can be string path or numpy array
        :param input_buffers:
        :param output_buffers:
        :param bindings:
        :param stream:
        :param engine:
        :param tensorrt_engine_context: running context of deserialized tensorrt engine
        :param aug_pipeline: image pre-processing like flip, normalization
        :param classification_threshold: higher->higher precision, lower->higher recall
        :param nms_threshold:
        :param class_agnostic:
        """
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
        image_width = data_batch.shape[3]
        image_height = data_batch.shape[2]
        input_buffers[0].host = data_batch.astype(dtype=numpy.float32, order='C')

        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in input_buffers]
        tensorrt_engine_context.execute_async(batch_size=1, bindings=bindings, stream_handle=stream.handle)
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in output_buffers]
        stream.synchronize()

        output_shapes = []
        for binding in engine:
            if not engine.binding_is_input(binding):
                output_shapes.append([engine.max_batch_size] + list(engine.get_binding_shape(binding)))
        outputs = [out.host for out in output_buffers]
        outputs = [numpy.squeeze(output.reshape(shape), axis=(0, 1)) for output, shape in zip(outputs, output_shapes)]
        predicted_classification = torch.from_numpy(outputs[0]).cuda()
        predicted_regression = torch.from_numpy(outputs[1]).cuda()

        if len(self._head_indexes_to_feature_map_sizes) == 0:  # in self.forward(), self._head_indexes_to_feature_map_sizes is filled dynamically. but we have to compute manually
            for i, stride in enumerate(self._point_strides):
                loop = int(math.log2(stride))
                map_height = image_height
                map_width = image_width
                for l in range(loop):
                    map_height = int((map_height + 1) / 2)
                    map_width = int((map_width + 1) / 2)
                self._head_indexes_to_feature_map_sizes[i] = (map_height, map_width)

        all_point_coordinates_list = self.generate_point_coordinates(self._head_indexes_to_feature_map_sizes)
        expanded_regression_ranges_list = [all_point_coordinates_list[i].new_tensor(self._regression_ranges[i])[None].expand_as(all_point_coordinates_list[i])
                                           for i in range(self._num_heads)]

        concat_point_coordinates = torch.cat(all_point_coordinates_list, dim=0)
        concat_regression_ranges = torch.cat(expanded_regression_ranges_list, dim=0)

        if type(self._classification_loss_func).__name__ in ['CrossEntropyLoss']:
            predicted_classification = predicted_classification.softmax(dim=1)
            predicted_classification = predicted_classification[:, :-1]  # remove bg
        else:
            predicted_classification = predicted_classification.sigmoid()

        concat_point_coordinates = concat_point_coordinates.to(predicted_regression.device)
        concat_regression_ranges = concat_regression_ranges.to(predicted_regression.device)

        classification_threshold = classification_threshold if classification_threshold is not None else self._classification_threshold
        max_scores = predicted_classification.max(dim=1)[0]
        selected_indexes = torch.where(max_scores > classification_threshold)[0]
        if selected_indexes.numel() == 0:
            return []

        predicted_classification = predicted_classification[selected_indexes]
        predicted_regression = predicted_regression[selected_indexes]
        concat_point_coordinates = concat_point_coordinates[selected_indexes]
        concat_regression_ranges = concat_regression_ranges[selected_indexes]

        #  calculate bboxes' x1 y1 x2 y2
        if self._regression_loss_type == 'independent':
            predicted_regression = predicted_regression * concat_regression_ranges[..., 1, None]
            x1 = concat_point_coordinates[:, 0] - predicted_regression[:, 0]
            x1 = x1.clamp(min=0, max=image_width)
            y1 = concat_point_coordinates[:, 1] - predicted_regression[:, 1]
            y1 = y1.clamp(min=0, max=image_height)
            x2 = concat_point_coordinates[:, 0] + predicted_regression[:, 2]
            x2 = x2.clamp(min=0, max=image_width)
            y2 = concat_point_coordinates[:, 1] + predicted_regression[:, 3]
            y2 = y2.clamp(min=0, max=image_height)
            predicted_bboxes = torch.stack([x1, y1, x2, y2], -1)
        else:
            if self._distance_to_bbox_mode == 'exp':
                predicted_regression = predicted_regression.float().exp()
                predicted_bboxes = self.distance2bbox(concat_point_coordinates, predicted_regression, max_shape=(image_height, image_width))
            elif self._distance_to_bbox_mode == 'sigmoid':
                concat_regression_ranges_max = concat_regression_ranges.max(dim=-1)[0]
                predicted_regression = predicted_regression.sigmoid() * concat_regression_ranges_max[..., None]
                predicted_bboxes = self.distance2bbox(concat_point_coordinates, predicted_regression, max_shape=(image_height, image_width))
            else:
                raise ValueError('Unknown distance_to_bbox mode!')
        # add BG label for multi class nms
        bg_label_padding = predicted_classification.new_zeros(predicted_classification.size(0), 1)
        predicted_classification = torch.cat([predicted_classification, bg_label_padding], dim=1)

        if nms_threshold:
            self._nms_cfg.update({'iou_thr': nms_threshold})
        if class_agnostic:
            self._nms_cfg.update({'class_agnostic': class_agnostic})
        nms_bboxes, nms_labels = multiclass_nms(
            multi_bboxes=predicted_bboxes,
            multi_scores=predicted_classification,
            score_thr=classification_threshold,
            nms_cfg=self._nms_cfg,
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


class LFDv2_(nn.Module):
    """
    Based on LFDv1, LFDv2 has the following new features:
    1, new centerness (follows FCOS) with calibration (v1: original centerness with calibration)
    2, gray scale relaxation (v1: gray scale is totally ignored)
    3, iou score coupling
    """

    def __init__(self,
                 backbone=None,
                 neck=None,
                 head=None,
                 num_classes=80,
                 head_measure_ranges=((0, 64), (64, 128), (128, 256), (256, 512), (512, 1024)),

                 enable_head_measure_range_relaxation=False,
                 measure_range_relaxation_factor=0.2,
                 enable_centerness_calibration=False,
                 enable_iou_score_coupling=False,

                 head_assign_measure='longer',
                 point_strides=(8, 16, 32, 64, 128),
                 classification_loss_func=None,
                 regression_loss_func=None,
                 classification_threshold=0.05,
                 nms_threshold=0.5,
                 pre_nms_bbox_limit=1000,
                 post_nms_bbox_limit=100,
                 ):
        super(LFDv2, self).__init__()
        assert len(head_measure_ranges) == len(point_strides)
        assert head_assign_measure in ['longer', 'shorter', 'sqrt', 'dist']

        self._backbone = backbone
        self._neck = neck
        self._head = head
        self._num_classes = num_classes
        self._head_measure_ranges = head_measure_ranges
        self._head_assign_measure = head_assign_measure

        self._enable_head_measure_range_relaxation = enable_head_measure_range_relaxation
        self._measure_range_relaxation_factor = measure_range_relaxation_factor
        self._measure_range_relaxation_ranges = \
            [(int(low * (1 - self._measure_range_relaxation_factor)), int(up * (1 + self._measure_range_relaxation_factor))) for (low, up) in self._head_measure_ranges]
        self._enable_centerness_calibration = enable_centerness_calibration
        self._enable_iou_score_coupling = enable_iou_score_coupling
        self._num_heads = len(point_strides)
        self._point_strides = point_strides

        if classification_loss_func is not None:
            assert type(classification_loss_func).__name__ in ['QualityFocalLoss']
        self._classification_loss_func = classification_loss_func

        if regression_loss_func is not None:
            assert type(regression_loss_func).__name__ in ['IoULoss', 'GIoULoss', 'DIoULoss', 'CIoULoss']
        self._regression_loss_func = regression_loss_func

        self._classification_threshold = classification_threshold
        self._nms_cfg = dict(type='nms', iou_thr=nms_threshold)
        self._pre_nms_bbox_limit = pre_nms_bbox_limit
        self._post_nms_bbox_limit = post_nms_bbox_limit

        self._head_indexes_to_feature_map_sizes = dict()

    @property
    def head_indexes_to_feature_map_sizes(self):
        return self._head_indexes_to_feature_map_sizes

    def generate_point_coordinates(self, feature_map_sizes):
        """
        transform feature map points to locations in original input image
        :param feature_map_sizes:
        :return:
        """

        def generate_for_single_feature_map(func_feature_map_size, func_stride):
            height, width = func_feature_map_size
            x_coordinates = torch.arange(0, width * func_stride, func_stride)
            y_coordinates = torch.arange(0, height * func_stride, func_stride)

            y_mesh, x_mesh = torch.meshgrid(y_coordinates, x_coordinates)

            point_coordinates = torch.stack((x_mesh.reshape(-1), y_mesh.reshape(-1)), dim=-1)

            return point_coordinates

        assert len(feature_map_sizes) == len(self._point_strides)
        all_point_coordinates_list = []
        for i in range(len(self._point_strides)):
            all_point_coordinates_list.append(generate_for_single_feature_map(feature_map_sizes[i], self._point_strides[i]))

        return all_point_coordinates_list

    def annotation_to_target(self, all_point_coordinates_list, gt_bboxes_list, gt_labels_list, *args):

        expanded_head_measure_ranges_list = [all_point_coordinates_list[i].new_tensor(self._head_measure_ranges[i])[None].expand_as(all_point_coordinates_list[i])
                                             for i in range(self._num_heads)]
        expanded_measure_range_relaxation_ranges_list = [all_point_coordinates_list[i].new_tensor(self._measure_range_relaxation_ranges[i])[None].expand_as(all_point_coordinates_list[i])
                                                         for i in range(self._num_heads)]
        expanded_strides_list = [all_point_coordinates_list[i].new_tensor(self._point_strides[i]).expand(all_point_coordinates_list[i].size(0))
                                 for i in range(self._num_heads)]

        concat_point_coordinates = torch.cat(all_point_coordinates_list, dim=0)
        concat_head_measure_ranges = torch.cat(expanded_head_measure_ranges_list, dim=0)
        concat_measure_range_relaxation_ranges = torch.cat(expanded_measure_range_relaxation_ranges_list, dim=0)
        concat_strides = torch.cat(expanded_strides_list, dim=0)

        classification_targets_list = list()
        regression_targets_list = list()
        for i, gt_bboxes in enumerate(gt_bboxes_list):
            temp_classification_targets, temp_regression_targets = self._generate_target_for_single_image(gt_bboxes=gt_bboxes,
                                                                                                          gt_labels=gt_labels_list[i],
                                                                                                          concat_point_coordinates=concat_point_coordinates,
                                                                                                          concat_head_measure_ranges=concat_head_measure_ranges,
                                                                                                          concat_measure_range_relaxation_ranges=concat_measure_range_relaxation_ranges,
                                                                                                          concat_strides=concat_strides)

            # display for debug---------------------------------------------------------------------------------------
            # split_classification_targets_list = temp_classification_targets.split([coord.size(0) for coord in all_point_coordinates_list], dim=0)
            # split_classification_targets_list = [target.reshape(int(math.sqrt(target.size(0))), int(math.sqrt(target.size(0))), -1) for target in split_classification_targets_list]
            #
            # display_class_label = 1
            # for j, target in enumerate(split_classification_targets_list):
            #     display_map = target[..., display_class_label].numpy()
            #     display_map = display_map * 255
            #     display_map[display_map < 0] = 127
            #     display_map = display_map.astype(dtype=numpy.uint8)
            #
            #     cv2.imshow(str(j), display_map)
            # cv2.waitKey()
            # ---------------------------------------------------------------------------------------------------------

            classification_targets_list.append(temp_classification_targets)
            regression_targets_list.append(temp_regression_targets)

        stack_classification_targets_tensor = torch.stack(classification_targets_list, dim=0)
        stack_regression_targets_tensor = torch.stack(regression_targets_list, dim=0)
        return stack_classification_targets_tensor, stack_regression_targets_tensor

    def _generate_target_for_single_image(self,
                                          gt_bboxes,
                                          gt_labels,
                                          concat_point_coordinates,
                                          concat_head_measure_ranges,
                                          concat_measure_range_relaxation_ranges,
                                          concat_strides):

        assert gt_bboxes.size(0) == gt_labels.size(0)

        num_points = concat_point_coordinates.size(0)
        num_gt_bboxes = gt_bboxes.size(0)

        classification_targets = gt_bboxes.new_zeros((num_points, self._num_classes))
        regression_targets = gt_bboxes.new_zeros((num_points, 4))

        if num_gt_bboxes == 0:
            return classification_targets, regression_targets

        gt_bboxes = gt_bboxes[None].expand(num_points, num_gt_bboxes, 4)  # PxMx4 (P: number of points, M: number of bboxes)
        gt_labels = gt_labels[None].expand(num_points, num_gt_bboxes)

        concat_head_measure_ranges = concat_head_measure_ranges[:, None, :].expand(num_points, num_gt_bboxes, 2)  # PxMx2
        concat_measure_range_relaxation_ranges = concat_measure_range_relaxation_ranges[:, None, :].expand(num_points, num_gt_bboxes, 2)

        point_x_coordinates, point_y_corrdinates = concat_point_coordinates[:, 0], concat_point_coordinates[:, 1]
        point_x_coordinates = point_x_coordinates[:, None].expand(num_points, num_gt_bboxes)
        point_y_corrdinates = point_y_corrdinates[:, None].expand(num_points, num_gt_bboxes)

        # calculate regression values (each bbox is described as [x, y, width, height])
        delta_x1 = point_x_coordinates - gt_bboxes[..., 0]
        delta_y1 = point_y_corrdinates - gt_bboxes[..., 1]
        delta_x2 = (gt_bboxes[..., 0] + gt_bboxes[..., 2] - 1) - point_x_coordinates
        delta_y2 = (gt_bboxes[..., 1] + gt_bboxes[..., 3] - 1) - point_y_corrdinates
        regression_delta = torch.stack((delta_x1, delta_y1, delta_x2, delta_y2), dim=-1)  # distance to left, top, right, bottom

        hit_condition = regression_delta.min(dim=-1)[0] >= 0  # this means the point is in a certain bbox

        # centerness scores
        filtered_regression_delta = regression_delta * hit_condition[..., None].expand((num_points, num_gt_bboxes, 4))  # filter cases that points do not hit
        point_centerness_scores = self.centerness_score(filtered_regression_delta)
        if self._enable_centerness_calibration:
            gt_bboxes_center_x = gt_bboxes[..., 0] + gt_bboxes[..., 2] / 2.
            gt_bboxes_center_y = gt_bboxes[..., 1] + gt_bboxes[..., 3] / 2.
            strides = concat_strides[..., None].expand((num_points, num_gt_bboxes))
            core_zone_left = gt_bboxes_center_x - strides / 2
            core_zone_right = gt_bboxes_center_x + strides / 2
            core_zone_top = gt_bboxes_center_y - strides / 2
            core_zone_bottom = gt_bboxes_center_y + strides / 2

            inside_core_zone = (point_x_coordinates >= core_zone_left) & (point_x_coordinates <= core_zone_right) & \
                               (point_y_corrdinates >= core_zone_top) & (point_y_corrdinates <= core_zone_bottom)
            inside_core_zone = inside_core_zone & hit_condition  # in case that the point is out of bbox

            point_centerness_scores = point_centerness_scores * (~inside_core_zone) + inside_core_zone  # assign 1 to the scores of points that in core zone

        # determine pos/gray/neg points P x N
        # obtain measure
        if self._head_assign_measure == 'longer':
            assign_measure = torch.max(gt_bboxes[..., 2], gt_bboxes[..., 3])
        elif self._head_assign_measure == 'shorter':
            assign_measure = torch.min(gt_bboxes[..., 2], gt_bboxes[..., 3])
        elif self._head_assign_measure == 'sqrt':
            assign_measure = torch.sqrt(gt_bboxes[..., 2] * gt_bboxes[..., 3])
        elif self._head_assign_measure == 'dist':
            assign_measure = regression_delta.max(dim=-1)[0]
        else:
            raise ValueError('Unsupported range assign mode!')

        if self._enable_head_measure_range_relaxation:
            relaxation_scores = self.measure_range_relaxation_score(assign_measure, concat_head_measure_ranges, concat_measure_range_relaxation_ranges)
        else:
            relaxation_scores = (concat_head_measure_ranges[..., 0] <= assign_measure) & (assign_measure <= concat_head_measure_ranges[..., 1])

        final_score = point_centerness_scores * relaxation_scores
        positive_condition = final_score > 0

        # rank scores in ascending order for each point
        # why rank here: for a certain class, multiple objects may cover the same point, putting the largest score at the end will make
        # the classification_targets assigned with this largest score.
        # 对于单个类别来说，某个point可能落入多个这个类别目标的bbox中，那classification target应该被赋值为这个类别最大的那个得分，所以这里通过把
        # 最大的分数排在最后来实现的，具体影响的代码是：classification_targets[index1, green_label_index] = sorted_point_scores[index1, index2]
        sorted_final_scores, sorted_indexes = final_score.sort(dim=1)
        intermediate_indexes = sorted_indexes.new_tensor(range(sorted_indexes.size(0)))[..., None].expand(sorted_indexes.size(0), sorted_indexes.size(1))

        # reranking
        sorted_gt_labels = gt_labels[intermediate_indexes, sorted_indexes]
        sorted_positive_condition = positive_condition[intermediate_indexes, sorted_indexes]

        indexes_1, indexes_2 = torch.where(sorted_positive_condition)
        positive_label_indexes = sorted_gt_labels[indexes_1, indexes_2]
        classification_targets[indexes_1, positive_label_indexes] = sorted_final_scores[indexes_1, indexes_2]

        # for each point, select the regression target with the highest score (affected by green and gray conditions)
        _, select_indexes = sorted_final_scores.max(dim=1)
        sorted_regression_delta = regression_delta[intermediate_indexes, sorted_indexes]
        regression_targets = sorted_regression_delta[range(num_points), select_indexes]

        return classification_targets, regression_targets

    def centerness_score(self, bbox_targets):
        """Compute centerness targets.

        Args:
            bbox_targets (Tensor): BBox targets of all bboxes in shape
                (num_pos, num_bbox, 4)

        Returns:
            Tensor: Centerness target.
        """
        # only calculate pos centerness targets, otherwise there may be nan
        left_right = bbox_targets[..., [0, 2]]
        top_bottom = bbox_targets[..., [1, 3]]
        # clamp to avoid zero-divisor
        centerness_targets = ((left_right.min(dim=-1)[0]).clamp(min=0.0) / (left_right.max(dim=-1)[0]).clamp(min=0.01)) * \
                             ((top_bottom.min(dim=-1)[0]).clamp(min=0.0) / (top_bottom.max(dim=-1)[0]).clamp(min=0.01))
        return torch.sqrt(centerness_targets)

    def measure_range_relaxation_score(self, measure, measure_ranges, measure_relaxation_ranges):
        #  linear reduction
        left_relaxation_multiplier = (measure - measure_relaxation_ranges[..., 0]) / (measure_ranges[..., 0] - measure_relaxation_ranges[..., 0]).clamp(min=0.01)
        left_indicator = (measure_relaxation_ranges[..., 0] <= measure) & (measure < measure_ranges[..., 0])

        in_measure_range_indicator = (measure_ranges[..., 0] <= measure) & (measure <= measure_ranges[..., 1])

        right_relaxation_multiplier = (measure_relaxation_ranges[..., 1] - measure) / (measure_relaxation_ranges[..., 1] - measure_ranges[..., 1]).clamp(min=0.01)
        right_indicator = (measure_ranges[..., 1] < measure) & (measure <= measure_relaxation_ranges[..., 1])

        relaxation_score = left_relaxation_multiplier * left_indicator + in_measure_range_indicator + right_relaxation_multiplier * right_indicator

        return relaxation_score

    def distance2bbox(self, points, distance, max_shape=None):
        """Decode distance prediction to bounding box.

        Args:
            points (Tensor): Shape (n, 2), [x, y].
            distance (Tensor): Distance from the given point to 4
                boundaries (left, top, right, bottom).
            max_shape (tuple): Shape of the image.

        Returns:
            Tensor: Decoded bboxes.
        """
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        if max_shape is not None:
            x1 = x1.clamp(min=0, max=max_shape[1])
            y1 = y1.clamp(min=0, max=max_shape[0])
            x2 = x2.clamp(min=0, max=max_shape[1])
            y2 = y2.clamp(min=0, max=max_shape[0])
        return torch.stack([x1, y1, x2, y2], -1)

    def get_loss(self, predict_outputs, annotation_batch, *args):

        predict_classification_tensor, predict_regression_tensor = predict_outputs

        gt_bboxes_list, gt_labels_list = list(), list()
        for annotation in annotation_batch:
            bboxes_numpy, labels_numpy = annotation
            gt_bboxes_list.append(torch.from_numpy(bboxes_numpy))
            gt_labels_list.append(torch.from_numpy(labels_numpy))

        # 获取所有level上的feature map locations在原图中的坐标位置
        all_point_coordinates_list = self.generate_point_coordinates(self._head_indexes_to_feature_map_sizes)

        # 进行annotation 到 target 的转换
        classification_target_tensor, regression_target_tensor = self.annotation_to_target(all_point_coordinates_list, gt_bboxes_list, gt_labels_list)

        batch_size = predict_classification_tensor.size(0)
        flatten_predict_classification_tensor = predict_classification_tensor.reshape(-1, self._num_classes)
        flatten_predict_regression_tensor = predict_regression_tensor.reshape(-1, 4)
        flatten_classification_target_tensor = classification_target_tensor.reshape(-1, self._num_classes)
        flatten_regression_target_tensor = regression_target_tensor.reshape(-1, 4)
        flatten_all_point_coordinates = (torch.cat(all_point_coordinates_list, dim=0)).repeat(batch_size, 1)

        flatten_classification_target_tensor = flatten_classification_target_tensor.to(flatten_predict_classification_tensor.device)
        flatten_regression_target_tensor = flatten_regression_target_tensor.to(flatten_predict_regression_tensor.device)

        max_score_targets, max_score_indexes = flatten_classification_target_tensor.max(dim=-1)
        pos_indexes = (max_score_targets > 0).nonzero().reshape(-1)

        # get regression loss
        pos_flatten_predict_regression_tensor = flatten_predict_regression_tensor[pos_indexes]
        iou_score_targets = flatten_predict_classification_tensor.new_zeros(max_score_targets.shape)

        if len(pos_indexes) > 0:
            pos_points = flatten_all_point_coordinates[pos_indexes]
            pos_points = pos_points.to(flatten_predict_regression_tensor.device)
            pos_flatten_regression_target_tensor = flatten_regression_target_tensor[pos_indexes]

            pos_flatten_predict_regression_tensor = pos_flatten_predict_regression_tensor.float().exp()
            pos_decoded_bbox_preds = self.distance2bbox(pos_points, pos_flatten_predict_regression_tensor)
            pos_decoded_bbox_targets = self.distance2bbox(pos_points, pos_flatten_regression_target_tensor)

            bbox_reg_weights = flatten_predict_classification_tensor.detach().sigmoid()
            bbox_reg_weights = bbox_reg_weights[pos_indexes][range(len(pos_indexes)), max_score_indexes[pos_indexes]]
            bbox_reg_weights_denorm = max(bbox_reg_weights.sum(), 1.0)

            regression_loss = self._regression_loss_func(pos_decoded_bbox_preds, pos_decoded_bbox_targets,
                                                         weight=bbox_reg_weights,
                                                         avg_factor=bbox_reg_weights_denorm)

            iou_score_targets[pos_indexes] = bbox_overlaps(pos_decoded_bbox_preds.detach(), pos_decoded_bbox_targets, is_aligned=True)
        else:
            regression_loss = flatten_predict_regression_tensor.sum()

        if self._enable_iou_score_coupling:
            max_score_targets = max_score_targets * iou_score_targets

        cls_weights_denorm = max(max_score_targets.sum(), 1.0)
        label_targets = max_score_indexes * (max_score_targets > 0) + self._num_classes * (max_score_targets <= 0)
        classification_loss = self._classification_loss_func(flatten_predict_classification_tensor,
                                                             [label_targets, max_score_targets],
                                                             avg_factor=cls_weights_denorm)

        loss = classification_loss + regression_loss
        loss_values = dict(loss=loss.item(),
                           classification_loss=classification_loss.item(),
                           regression_loss=regression_loss.item())

        return dict(loss=loss,
                    loss_values=loss_values)

    def get_results(self, predict_outputs, *args):
        """
        for online evaluation
        :param predict_outputs:
        :param args:
        :return:
        """
        predict_classification_tensor, predict_regression_tensor = predict_outputs
        num_samples = predict_classification_tensor.size(0)
        all_point_coordinates_list = self.generate_point_coordinates(self._head_indexes_to_feature_map_sizes)

        meta_batch = args[0]

        results = []
        for i in range(num_samples):
            nms_bboxes, nms_labels = self._get_results_for_single_image(predict_classification_tensor[i],
                                                                        predict_regression_tensor[i],
                                                                        all_point_coordinates_list,
                                                                        meta_batch[i])
            if nms_bboxes.size(0) == 0:
                results.append([])
                continue

            # [x1, y1, x2, y2, score] -> [x1, y1, w, h, score]
            nms_bboxes[:, 2] = nms_bboxes[:, 2] - nms_bboxes[:, 0] + 1
            nms_bboxes[:, 3] = nms_bboxes[:, 3] - nms_bboxes[:, 1] + 1
            # each row : [class_label, score, x1, y1, w, h]
            temp_results = torch.cat([nms_labels[:, None].to(nms_bboxes), nms_bboxes[:, [4, 0, 1, 2, 3]]], dim=1)
            # from tensor to list
            temp_results = temp_results.tolist()
            temp_results = [[int(temp_result[0])] + temp_result[1:] for temp_result in temp_results]

            results.append(temp_results)
        return results

    def _get_results_for_single_image(self,
                                      predicted_classification,
                                      predicted_regression,
                                      all_point_coordinates_list,
                                      meta_info):
        split_list = [point_coordinates_per_level.size(0) for point_coordinates_per_level in all_point_coordinates_list]
        predicted_classification_split = predicted_classification.split(split_list, dim=0)
        predicted_regression_split = predicted_regression.split(split_list, dim=0)
        image_resized_height = meta_info['resized_height']
        image_resized_width = meta_info['resized_width']

        predicted_classification_merge = list()
        predicted_bboxes_merge = list()

        for i in range(len(split_list)):

            temp_predicted_classification = predicted_classification_split[i].sigmoid()
            temp_predicted_regression = predicted_regression_split[i]
            temp_point_coordinates = all_point_coordinates_list[i].to(temp_predicted_regression.device)

            if 0 < self._pre_nms_bbox_limit < temp_predicted_classification.size(0):
                temp_max_scores = temp_predicted_classification.max(dim=1)[0]
                topk_indexes = temp_max_scores.topk(self._pre_nms_bbox_limit)[1]
                temp_predicted_classification = temp_predicted_classification[topk_indexes]
                temp_predicted_regression = temp_predicted_regression[topk_indexes]
                temp_point_coordinates = temp_point_coordinates[topk_indexes]

                #  calculate bboxes' x1 y1 x2 y2
                temp_predicted_regression = temp_predicted_regression.float().exp()
                temp_bboxes = self.distance2bbox(temp_point_coordinates, temp_predicted_regression, max_shape=(image_resized_height, image_resized_width))

            predicted_classification_merge.append(temp_predicted_classification)
            predicted_bboxes_merge.append(temp_bboxes)

        predicted_classification_merge = torch.cat(predicted_classification_merge)
        # add BG label for multi class nms
        bg_label_padding = predicted_classification_merge.new_zeros(predicted_classification_merge.size(0), 1)
        predicted_classification_merge = torch.cat([predicted_classification_merge, bg_label_padding], dim=1)

        predicted_bboxes_merge = torch.cat(predicted_bboxes_merge)
        predicted_bboxes_merge = predicted_bboxes_merge / meta_info['resize_scale']

        nms_bboxes, nms_labels = multiclass_nms(
            multi_bboxes=predicted_bboxes_merge,
            multi_scores=predicted_classification_merge,
            score_thr=self._classification_threshold,
            nms_cfg=self._nms_cfg,
            max_num=self._post_nms_bbox_limit,
            score_factors=None
        )

        return nms_bboxes, nms_labels

    def forward(self, x):

        backbone_outputs = self._backbone(x)

        neck_outputs = self._neck(backbone_outputs)

        head_outputs = self._head(neck_outputs)  # in case of outputs > 2, like fcos head (cls, reg, centerness)

        classification_outputs = head_outputs[0]
        regression_outputs = head_outputs[1]

        #  变换输出的dim和shape，转化成tensor输出
        #  tensor 中的dim n必须要保留，为了DP能够正常多卡
        classification_reformat_outputs = []
        regression_reformat_outputs = []
        for i, classification_output in enumerate(classification_outputs):
            n, c, h, w = classification_output.shape
            classification_output = classification_output.permute([0, 2, 3, 1])
            classification_output = classification_output.reshape((n, h * w, c))
            classification_reformat_outputs.append(classification_output)

            self._head_indexes_to_feature_map_sizes[i] = (h, w)

            n, c, h, w = regression_outputs[i].shape
            regression_output = regression_outputs[i].permute([0, 2, 3, 1])
            regression_output = regression_output.reshape((n, h * w, c))
            regression_reformat_outputs.append(regression_output)

        classification_output_tensor = torch.cat(classification_reformat_outputs, dim=1)
        regression_output_tensor = torch.cat(regression_reformat_outputs, dim=1)

        return classification_output_tensor, regression_output_tensor

    def predict_for_single_image(self, image, aug_pipeline, classification_threshold=None, nms_threshold=None, class_agnostic=False):
        """
        for easy prediction
        :param image: image can be string path or numpy array
        :param aug_pipeline: image pre-processing like flip, normalization
        :param classification_threshold: higher->higher precision, lower->higher recall
        :param nms_threshold:
        :param class_agnostic:
        """
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
        data_batch = data_batch.cuda()
        self.cuda()
        self.eval()

        with torch.no_grad():
            predicted_classification, predicted_regression = self.forward(data_batch)
        predicted_classification = predicted_classification[0]
        predicted_regression = predicted_regression[0]

        all_point_coordinates_list = self.generate_point_coordinates(self._head_indexes_to_feature_map_sizes)

        concat_point_coordinates = torch.cat(all_point_coordinates_list, dim=0)

        predicted_classification = predicted_classification.sigmoid()

        concat_point_coordinates = concat_point_coordinates.to(predicted_regression.device)

        classification_threshold = classification_threshold if classification_threshold is not None else self._classification_threshold
        max_scores = predicted_classification.max(dim=1)[0]
        selected_indexes = torch.where(max_scores > classification_threshold)[0]
        if selected_indexes.numel() == 0:
            return []

        predicted_classification = predicted_classification[selected_indexes]
        predicted_regression = predicted_regression[selected_indexes]
        concat_point_coordinates = concat_point_coordinates[selected_indexes]

        #  calculate bboxes' x1 y1 x2 y2
        predicted_regression = predicted_regression.float().exp()
        predicted_bboxes = self.distance2bbox(concat_point_coordinates, predicted_regression, max_shape=(image_height, image_width))

        # add BG label for multi class nms
        bg_label_padding = predicted_classification.new_zeros(predicted_classification.size(0), 1)
        predicted_classification = torch.cat([predicted_classification, bg_label_padding], dim=1)

        if nms_threshold:
            self._nms_cfg.update({'iou_thr': nms_threshold})
        if class_agnostic:
            self._nms_cfg.update({'class_agnostic': class_agnostic})
        nms_bboxes, nms_labels = multiclass_nms(
            multi_bboxes=predicted_bboxes,
            multi_scores=predicted_classification,
            score_thr=classification_threshold,
            nms_cfg=self._nms_cfg,
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

    def predict_for_single_image_with_tensorrt(self,
                                               image,
                                               input_buffers,
                                               output_buffers,
                                               bindings,
                                               stream,
                                               engine,
                                               tensorrt_engine_context,
                                               aug_pipeline,
                                               classification_threshold=None,
                                               nms_threshold=None,
                                               class_agnostic=False):
        """
        for easy prediction, using tensorrt as inference engine instead
        :param image: image can be string path or numpy array
        :param input_buffers:
        :param output_buffers:
        :param bindings:
        :param stream:
        :param engine:
        :param tensorrt_engine_context: running context of deserialized tensorrt engine
        :param aug_pipeline: image pre-processing like flip, normalization
        :param classification_threshold: higher->higher precision, lower->higher recall
        :param nms_threshold:
        :param class_agnostic:
        """
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
        image_width = data_batch.shape[3]
        image_height = data_batch.shape[2]
        input_buffers[0].host = data_batch.astype(dtype=numpy.float32, order='C')

        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in input_buffers]
        tensorrt_engine_context.execute_async(batch_size=1, bindings=bindings, stream_handle=stream.handle)
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in output_buffers]
        stream.synchronize()

        output_shapes = []
        for binding in engine:
            if not engine.binding_is_input(binding):
                output_shapes.append([engine.max_batch_size] + list(engine.get_binding_shape(binding)))
        outputs = [out.host for out in output_buffers]
        outputs = [numpy.squeeze(output.reshape(shape), axis=(0, 1)) for output, shape in zip(outputs, output_shapes)]
        predicted_classification = torch.from_numpy(outputs[0]).cuda()
        predicted_regression = torch.from_numpy(outputs[1]).cuda()

        if len(self._head_indexes_to_feature_map_sizes) == 0:  # in self.forward(), self._head_indexes_to_feature_map_sizes is filled dynamically. but we have to compute manually
            for i, stride in enumerate(self._point_strides):
                loop = int(math.log2(stride))
                map_height = image_height
                map_width = image_width
                for l in range(loop):
                    map_height = int((map_height + 1) / 2)
                    map_width = int((map_width + 1) / 2)
                self._head_indexes_to_feature_map_sizes[i] = (map_height, map_width)

        all_point_coordinates_list = self.generate_point_coordinates(self._head_indexes_to_feature_map_sizes)

        concat_point_coordinates = torch.cat(all_point_coordinates_list, dim=0)

        predicted_classification = predicted_classification.sigmoid()

        concat_point_coordinates = concat_point_coordinates.to(predicted_regression.device)

        classification_threshold = classification_threshold if classification_threshold is not None else self._classification_threshold
        max_scores = predicted_classification.max(dim=1)[0]
        selected_indexes = torch.where(max_scores > classification_threshold)[0]
        if selected_indexes.numel() == 0:
            return []

        predicted_classification = predicted_classification[selected_indexes]
        predicted_regression = predicted_regression[selected_indexes]
        concat_point_coordinates = concat_point_coordinates[selected_indexes]

        #  calculate bboxes' x1 y1 x2 y2
        predicted_regression = predicted_regression.float().exp()
        predicted_bboxes = self.distance2bbox(concat_point_coordinates, predicted_regression, max_shape=(image_height, image_width))

        # add BG label for multi class nms
        bg_label_padding = predicted_classification.new_zeros(predicted_classification.size(0), 1)
        predicted_classification = torch.cat([predicted_classification, bg_label_padding], dim=1)

        if nms_threshold:
            self._nms_cfg.update({'iou_thr': nms_threshold})
        if class_agnostic:
            self._nms_cfg.update({'class_agnostic': class_agnostic})
        nms_bboxes, nms_labels = multiclass_nms(
            multi_bboxes=predicted_bboxes,
            multi_scores=predicted_classification,
            score_thr=classification_threshold,
            nms_cfg=self._nms_cfg,
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
