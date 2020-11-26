# -*- coding: utf-8 -*-
# author: Yonghao He
# description: 

import torch
import torch.nn as nn
import numpy
import cv2
import math
from .utils import multiclass_nms

__all__ = ['LFD']

INF = 1e8


class LFD(nn.Module):
    """

    """

    def __init__(self,
                 backbone=None,
                 neck=None,
                 head=None,
                 num_classes=80,
                 regression_ranges=((0, 64), (64, 128), (128, 256), (256, 512), (512, INF)),
                 gray_range_factors=(0.9, 1.1),
                 point_strides=(8, 16, 32, 64, 128),
                 classification_loss_func=None,
                 regression_loss_func=None,
                 classification_threshold=0.05,
                 nms_threshold=0.5,
                 pre_nms_bbox_limit=1000,
                 post_nms_bbox_limit=100,
                 ):
        super(LFD, self).__init__()
        assert len(regression_ranges) == len(point_strides)

        self._backbone = backbone
        self._neck = neck
        self._head = head
        self._num_classes = num_classes
        self._regression_ranges = regression_ranges
        self._gray_range_factors = (min(gray_range_factors), max(gray_range_factors))
        self._gray_ranges = [(low * self._gray_range_factors[0], up * self._gray_range_factors[1]) for (low, up) in self._regression_ranges]
        self._num_heads = len(point_strides)
        self._point_strides = point_strides

        if classification_loss_func is not None:
            assert type(classification_loss_func).__name__ in ['BCEWithLogitsLoss', 'FocalLoss', 'CrossEntropyLoss']
        self._classification_loss_func = classification_loss_func

        if regression_loss_func is not None:
            assert type(regression_loss_func).__name__ in ['SmoothL1Loss', 'MSELoss']
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

            # 得到点的坐标矩阵，shape为（n,2）, n= height x width, 第一列为x的坐标，第二列为y的坐标
            # CAUTION：FCOS将feature map的locations映射回原图上的points，加上了 stride // 2，从感受野中心的计算上看，这里是不用加的，加了反而产生了
            # 偏差，但是为了忠于原文的逻辑，这里加上了
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
        gt_bboxes_larger_side = torch.max(gt_bboxes[..., 2], gt_bboxes[..., 3])
        concat_regression_ranges = concat_regression_ranges[:, None, :].expand(num_points, num_gt_bboxes, 2)
        concat_gray_ranges = concat_gray_ranges[:, None, :].expand(num_points, num_gt_bboxes, 2)

        point_x_coordinates, point_y_corrdinates = concat_point_coordinates[:, 0], concat_point_coordinates[:, 1]
        point_x_coordinates = point_x_coordinates[:, None].expand(num_points, num_gt_bboxes)
        point_y_corrdinates = point_y_corrdinates[:, None].expand(num_points, num_gt_bboxes)
        gt_bboxes_center_x = gt_bboxes[..., 0] + gt_bboxes[..., 2] / 2.
        gt_bboxes_center_y = gt_bboxes[..., 1] + gt_bboxes[..., 3] / 2.

        concat_strides = concat_strides[:, None]

        # calculate scores for classification
        # the closer the point near the center, the higher score it will get
        abs_to_center_x = torch.abs(point_x_coordinates - gt_bboxes_center_x)
        abs_to_center_y = torch.abs(point_y_corrdinates - gt_bboxes_center_y)
        x_scores = abs_to_center_x / (concat_strides / 2.)
        x_scores = x_scores * (x_scores >= 1) + (x_scores < 1)
        x_scores = torch.sqrt(1. / x_scores)
        y_scores = abs_to_center_y / (concat_strides / 2.)
        y_scores = y_scores * (y_scores >= 1) + (y_scores < 1)
        y_scores = torch.sqrt(1. / y_scores)
        point_scores = x_scores * y_scores  # P x N

        # calculate regression values
        delta_x1 = point_x_coordinates - gt_bboxes[..., 0]
        delta_y1 = point_y_corrdinates - gt_bboxes[..., 1]
        delta_x2 = (gt_bboxes[..., 0] + gt_bboxes[..., 2] - 1) - point_x_coordinates
        delta_y2 = (gt_bboxes[..., 1] + gt_bboxes[..., 3] - 1) - point_y_corrdinates
        regression_delta = torch.stack((delta_x1, delta_y1, delta_x2, delta_y2), dim=-1)
        regression_delta = regression_delta / concat_regression_ranges[..., 1, None]  # P x N x 4

        # determine learnable points P x N
        head_selection_condition = (concat_regression_ranges[..., 0] <= gt_bboxes_larger_side) & (gt_bboxes_larger_side <= concat_regression_ranges[..., 1])
        hit_condition = regression_delta.min(dim=-1)[0] >= 0
        green_condition = head_selection_condition & hit_condition

        gray_condition1 = (concat_gray_ranges[..., 0] <= gt_bboxes_larger_side) & (gt_bboxes_larger_side < concat_regression_ranges[..., 0])
        gray_condition2 = (concat_regression_ranges[..., 1] < gt_bboxes_larger_side) & (gt_bboxes_larger_side <= concat_gray_ranges[..., 1])
        gray_condition = (gray_condition1 | gray_condition2) & hit_condition

        # rank scores in ascending order for each point
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

        #
        if type(self._classification_loss_func).__name__ == 'CrossEntropyLoss':
            flatten_predict_classification_tensor = predict_classification_tensor.reshape(-1, self._num_classes + 1)
        else:
            flatten_predict_classification_tensor = predict_classification_tensor.reshape(-1, self._num_classes)
        flatten_predict_regression_tensor = predict_regression_tensor.reshape(-1, 4)
        flatten_classification_target_tensor = classification_target_tensor.reshape(-1, self._num_classes)  # (N*P,C)
        flatten_regression_target_tensor = regression_target_tensor.reshape(-1, 4)  # (N*P, 4)

        flatten_classification_target_tensor = flatten_classification_target_tensor.to(flatten_predict_classification_tensor.device)
        flatten_regression_target_tensor = flatten_regression_target_tensor.to(flatten_predict_regression_tensor.device)

        # ignore gray positions (in the future, we can only ignore gray classes within a position)
        min_scores = flatten_classification_target_tensor.min(dim=-1)[0]
        green_indexes = torch.where(min_scores >= 0)[0]

        flatten_predict_classification_tensor = flatten_predict_classification_tensor[green_indexes]
        flatten_predict_regression_tensor = flatten_predict_regression_tensor[green_indexes]
        flatten_classification_target_tensor = flatten_classification_target_tensor[green_indexes]
        flatten_regression_target_tensor = flatten_regression_target_tensor[green_indexes]

        max_scores, max_score_indexes = flatten_classification_target_tensor.max(dim=-1)
        pos_indexes = torch.where(max_scores >= 0.001)[0]
        # targets for FocalLoss
        if type(self._classification_loss_func).__name__ in ['FocalLoss', 'CrossEntropyLoss']:
            # assign background label
            max_score_indexes = max_score_indexes * (max_scores >= 0.001) + self._num_classes * (max_scores < 0.001)
            flatten_classification_target_tensor = max_score_indexes

        flatten_predict_regression_tensor = flatten_predict_regression_tensor[pos_indexes]
        flatten_regression_target_tensor = flatten_regression_target_tensor[pos_indexes]

        classification_loss = self._classification_loss_func(flatten_predict_classification_tensor, flatten_classification_target_tensor)
        if pos_indexes.nelement() > 0:
            regression_loss = self._regression_loss_func(flatten_predict_regression_tensor, flatten_regression_target_tensor)
        else:
            regression_loss = flatten_predict_regression_tensor.sum()

        loss = classification_loss + regression_loss
        loss_values = dict(loss=loss.item(),
                           classification_loss=classification_loss.item(),
                           regression_loss=regression_loss.item())

        return dict(loss=loss,
                    loss_values=loss_values)

    def get_results(self, predict_outputs, *args):
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
            temp_results = torch.cat([nms_labels[:, None].to(nms_bboxes), nms_bboxes[:, [4, 0, 1, 2, 3]]], dim=1)  # each row : [class_label, score, x1, y1, w, h]
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

        classification_outputs, regression_outputs = self._head(neck_outputs)

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
