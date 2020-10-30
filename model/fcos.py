# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from .utils import multiclass_nms, batched_nms

__all__ = ['FCOS']

INF = 1e8


class FCOS(nn.Module):

    def __init__(self,
                 backbone=None,
                 neck=None,
                 head=None,
                 num_classes=80,
                 regress_ranges=((0, 64), (64, 128), (128, 256), (256, 512), (512, INF)),
                 point_strides=(8, 16, 32, 64, 128),
                 classification_loss_func=None,
                 regression_loss_func=None,
                 centerness_loss_func=None,
                 classification_threshold=0.05,
                 nms_threshold=0.5,
                 pre_nms_bbox_limit=1000,
                 post_nms_bbox_limit=100,
                 param_groups_cfg=None):
        super(FCOS, self).__init__()
        assert len(regress_ranges) == len(point_strides), 'the length should be the same!'
        self._backbone = backbone
        self._neck = neck
        self._head = head
        # foreground label index ∈ [0,num_classes-1], background label index = num_classes
        self._num_classes = num_classes
        self._regress_ranges = regress_ranges
        self._point_strides = point_strides
        self._num_levels = len(point_strides)  # equal to the number of detection heads
        self._classification_loss_func = classification_loss_func
        self._regression_loss_func = regression_loss_func
        self._centerness_loss_func = centerness_loss_func
        self._classification_threshold = classification_threshold
        self._nms_cfg = dict(type='nms', iou_thr=nms_threshold)
        self._pre_nms_bbox_limit = pre_nms_bbox_limit
        self._post_nms_bbox_limit = post_nms_bbox_limit
        self._param_groups_cfg = param_groups_cfg
        self._head_indexes_to_feature_map_sizes = dict()  # dynamically record the feature map size for each head, and it will be used to obtain point coordinates

    @property
    def head_indexes_to_feature_map_sizes(self):
        return self._head_indexes_to_feature_map_sizes

    def get_param_groups_for_optimizer(self):
        """
        only support bias related config: bias_lr, bias_weight_decay
        :return:
        """
        if self._param_groups_cfg is not None:
            assert isinstance(self._param_groups_cfg, dict)
            bias_parameters = []
            other_parameters = []
            for name, module in self.named_modules():
                if isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):  # norm bias is ignored
                    for _, param in module.named_parameters(recurse=False):
                        other_parameters.append(param)
                else:
                    for param_name, param in module.named_parameters(recurse=False):
                        if 'bias' in param_name:
                            bias_parameters.append(param)
                        else:
                            other_parameters.append(param)
            bias_group = dict(params=bias_parameters)
            if 'bias_lr' in self._param_groups_cfg:
                bias_group['lr'] = self._param_groups_cfg['bias_lr']
            if 'bias_weight_decay' in self._param_groups_cfg:
                bias_group['weight_decay'] = self._param_groups_cfg['bias_weight_decay']
            param_groups = [dict(params=other_parameters), bias_group]
            return param_groups
        else:
            return self.parameters()

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
            point_coordinates = torch.stack((x_mesh.reshape(-1), y_mesh.reshape(-1)), dim=-1) + func_stride // 2

            return point_coordinates

        assert len(feature_map_sizes) == len(self._point_strides)
        all_point_coordinates_list = []
        for i in range(len(self._point_strides)):
            all_point_coordinates_list.append(generate_for_single_feature_map(feature_map_sizes[i], self._point_strides[i]))

        return all_point_coordinates_list

    def annotation_to_target(self, all_point_coordinates_list, gt_bboxes_list, gt_labels_list):
        """
        根据每幅图像的gt bboxes和labels生成可以直接用于loss计算的targets
        :param all_point_coordinates_list: 由 generate_point_coordinates生成的 point coordinates list
        :param gt_bboxes_list: list，其中每个元素为每幅图像的gt bboxes信息，格式为torch.Tensor([[x,y,w,h],...])
        :param gt_labels_list: lsit,其中每个元素为每幅图像的gt bboxes的类别标签，其格式为torch.Tensor([label1, label2,...])
        :return:
        """

        def generate_target_for_single_image(func_gt_bboxes, func_gt_labels, func_concat_point_coordinates, func_concat_regress_ranges):
            """
            注意，这里的所有前景类别标签∈[0,num_classes-1],背景标签类别为num_classes
            注意，这里的所有前景类别标签∈[0,num_classes-1],背景标签类别为num_classes
            注意，这里的所有前景类别标签∈[0,num_classes-1],背景标签类别为num_classes
            :param func_gt_bboxes: 一幅图像的所有gt bboxes, [[x,y,w,h],...]
            :param func_gt_labels: 一幅图像的所有gt bboxes labels[label1, labels2]
            :param func_concat_point_coordinates:
            :param func_concat_regress_ranges:
            :return:
            """
            assert func_gt_bboxes.size(0) == func_gt_labels.size(0)
            num_points = func_concat_point_coordinates.size(0)  # 注意，这是所有level的所有points
            num_gt_bboxes = func_gt_labels.size(0)
            if num_gt_bboxes == 0:  # 当该幅图像没有gt_bboxes时，直接返回label index = num_classes
                return func_gt_labels.new_full((num_points,), self._num_classes), \
                       func_gt_bboxes.new_zeros((num_points, 4))

            # 计算每个gt_bboxes的面积，并且将其扩展到num_points那么多个
            gt_bbox_areas = func_gt_bboxes[:, 2] * func_gt_bboxes[:, 3]
            gt_bbox_areas = gt_bbox_areas[None].repeat(num_points, 1)

            # 扩展regress_ranges
            func_concat_regress_ranges = func_concat_regress_ranges[:, None, :].expand(num_points, num_gt_bboxes, 2)
            # 扩展gt_bboxes
            func_gt_bboxes = func_gt_bboxes[None].expand(num_points, num_gt_bboxes, 4)

            # 单独获取points的x和y坐标，并扩展
            point_x_coordinates, point_y_coordinates = func_concat_point_coordinates[:, 0], func_concat_point_coordinates[:, 1]
            point_x_coordinates = point_x_coordinates[:, None].expand(num_points, num_gt_bboxes)
            point_y_coordinates = point_y_coordinates[:, None].expand(num_points, num_gt_bboxes)

            # 计算x和y分别到所有gt_bboxes的四个边的距离
            distance_to_left = point_x_coordinates - func_gt_bboxes[:, :, 0]
            distance_to_right = (func_gt_bboxes[:, :, 0] + func_gt_bboxes[:, :, 2] - 1) - point_x_coordinates
            distance_to_top = point_y_coordinates - func_gt_bboxes[:, :, 1]
            distance_to_bottom = (func_gt_bboxes[:, :, 1] + func_gt_bboxes[:, :, 3] - 1) - point_y_coordinates

            regress_targets = torch.stack((distance_to_left,
                                           distance_to_top,
                                           distance_to_right,
                                           distance_to_bottom), dim=-1)

            # 这里还有centerness sampling没有实现
            # TBD

            # 获取points是否落在gt_bboxes中的指示变量
            # 这里不同level中的points都有可能落进gt_bboxes中，后续还需要通过regress_ranges进一步剔除
            inside_gt_bbox_indicator = regress_targets.min(dim=-1)[0] > 0

            # 根据points到四个边的最大距离，判断gt_bboxes是否需要在某个point被学习，进一步过滤不需要学习的points
            max_regress_distance = regress_targets.max(dim=-1)[0]
            inside_regress_range_indicator = (max_regress_distance >= func_concat_regress_ranges[:, :, 0]) & \
                                             (max_regress_distance <= func_concat_regress_ranges[:, :, 1])

            # 根据indcator重置areas中的未命中的为INF，便于后续根据面积消除歧义
            # 这里使用更加高效的计算方式，替代mmdet中的计算方式
            # 在 mmdet 中：
            # >> gt_bbox_areas[inside_gt_bbox_indicator == 0] = INF
            # >> gt_bbox_areas[inside_regress_range_indicator == 0] = INF
            # 注意：以下的float类型的数据和bool型数据乘法，torch 1.2是不支持的
            valid_indicator = inside_gt_bbox_indicator & inside_regress_range_indicator
            gt_bbox_areas = gt_bbox_areas * valid_indicator + INF * (~valid_indicator)

            min_areas, min_area_indexes = gt_bbox_areas.min(dim=1)

            # 生成 targets
            classification_targets = func_gt_labels[min_area_indexes]
            # >>label_targets[min_areas == INF] = self._num_classes  # 由于很多points都是未命中的，所以areas都是INF，这里它们将会被当做背景 (注：背景的index设置为num_classes)
            # 替换为以下表达式
            classification_targets = classification_targets * (min_areas != INF) + self._num_classes * (min_areas == INF)

            # 这里在所有point的回归都是选择了最小面积的gt_bbox
            regress_targets = regress_targets[range(num_points), min_area_indexes]

            return classification_targets, regress_targets

        # 扩展regress ranges便于后续判断gt bboxes对于某个level是否是正样本.
        # 其中每一个expanded_regeress_ranges 的 shape = （n, 2）, n是对应level的points数量
        expanded_regress_ranges_list = [all_point_coordinates_list[i].new_tensor(self._regress_ranges[i])[None].expand_as(all_point_coordinates_list[i])
                                        for i in range(self._num_levels)]

        # 把所有的的point_coordinates和regress_ranges进行拼接
        concat_regress_ranges = torch.cat(expanded_regress_ranges_list, dim=0)
        concat_point_coordinates = torch.cat(all_point_coordinates_list, dim=0)

        # 针对每幅图像计算label_targets和regress_targets
        classification_targets_list = []
        regress_targets_list = []
        for i, gt_bboxes in enumerate(gt_bboxes_list):
            temp_classification_targets, temp_regress_targets = \
                generate_target_for_single_image(gt_bboxes, gt_labels_list[i], concat_point_coordinates, concat_regress_ranges)

            classification_targets_list.append(temp_classification_targets)
            regress_targets_list.append(temp_regress_targets)

        stack_classification_targets_tensor = torch.stack(classification_targets_list, dim=0)
        stack_regression_targets_tensor = torch.stack(regress_targets_list, dim=0)
        return stack_classification_targets_tensor, stack_regression_targets_tensor

    def centerness_target(self, pos_flatten_regress_targets):
        left_right = pos_flatten_regress_targets[:, [0, 2]]
        top_bottom = pos_flatten_regress_targets[:, [1, 3]]
        centerness_targets = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)

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
        """
        in this function, meta_batch (delivered by args) is not used
        in your own method, args may serve as a provider of extra data
        :param predict_outputs:
        :param annotation_batch:
        :return:
        """
        predict_classification_tensor, predict_regress_tensor, predict_centerness_tensor = predict_outputs

        gt_bboxes_list, gt_labels_list = list(), list()
        for annotation in annotation_batch:
            bboxes_numpy, labels_numpy = annotation
            gt_bboxes_list.append(torch.from_numpy(bboxes_numpy))
            gt_labels_list.append(torch.from_numpy(labels_numpy))

        # 获取所有level上的feature map locations在原图中的坐标位置
        all_point_coordinates_list = self.generate_point_coordinates(self._head_indexes_to_feature_map_sizes)

        # 进行annotation 到 target 的转换
        classification_target_tensor, regress_target_tensor = self.annotation_to_target(all_point_coordinates_list, gt_bboxes_list, gt_labels_list)

        # 对预测的三类feature map进行调整，使其能够和target对齐
        batch_size = predict_classification_tensor.shape[0]
        flatten_predict_classification_tensor = predict_classification_tensor.reshape(-1, self._num_classes)
        flatten_predict_regress_tensor = predict_regress_tensor.reshape(-1, 4)
        flatten_predict_centerness_tensor = predict_centerness_tensor.reshape(-1)

        flatten_classification_target_tensor = classification_target_tensor.reshape(-1)
        flatten_regress_target_tensor = regress_target_tensor.reshape(-1, 4)

        # flatten_all_point_coordinates = torch.cat([point_coordinates.repeat(batch_size, 1) for point_coordinates in all_point_coordinates_list])
        flatten_all_point_coordinates = (torch.cat(all_point_coordinates_list, dim=0)).repeat(batch_size, 1)

        if flatten_classification_target_tensor.get_device() != flatten_predict_classification_tensor.get_device():
            flatten_classification_target_tensor = flatten_classification_target_tensor.to(flatten_predict_classification_tensor.device)
        if flatten_regress_target_tensor.get_device() != flatten_predict_classification_tensor.get_device():
            flatten_regress_target_tensor = flatten_regress_target_tensor.to(flatten_predict_classification_tensor.device)
        if flatten_all_point_coordinates.get_device() != flatten_predict_classification_tensor.get_device():
            flatten_all_point_coordinates = flatten_all_point_coordinates.to(flatten_predict_classification_tensor.device)

        # 获取前景样本的点索引
        # 注意：这里的背景样本的label index为num_classes，所以正样本[0-79]。在focal loss中需要所有正样本为[0-79]
        # pos_indexes = flatten_label_targets.nonzero().reshape(-1)
        pos_indexes = (flatten_classification_target_tensor != self._num_classes).nonzero().reshape(-1)
        num_pos = pos_indexes.nelement()

        classification_loss = self._classification_loss_func(flatten_predict_classification_tensor, flatten_classification_target_tensor, avg_factor=num_pos + batch_size)

        pos_flatten_predict_regress_tensor = flatten_predict_regress_tensor[pos_indexes]
        pos_flatten_predict_centerness_tensor = flatten_predict_centerness_tensor[pos_indexes]

        if num_pos > 0:
            pos_flatten_regress_target_tensor = flatten_regress_target_tensor[pos_indexes]
            pos_flatten_centerness_target_tensor = self.centerness_target(pos_flatten_regress_target_tensor)
            pos_flatten_all_point_coordinates = flatten_all_point_coordinates[pos_indexes]

            pos_xyxy_predict_bboxes = self.distance2bbox(pos_flatten_all_point_coordinates, pos_flatten_predict_regress_tensor)
            pos_xyxy_target_bboxes = self.distance2bbox(pos_flatten_all_point_coordinates, pos_flatten_regress_target_tensor)

            regression_loss = self._regression_loss_func(pos_xyxy_predict_bboxes,
                                                         pos_xyxy_target_bboxes,
                                                         weight=pos_flatten_centerness_target_tensor,
                                                         avg_factor=pos_flatten_centerness_target_tensor.sum())
            centerness_loss = self._centerness_loss_func(pos_flatten_predict_centerness_tensor,
                                                         pos_flatten_centerness_target_tensor)
        else:
            regression_loss = pos_flatten_predict_regress_tensor.sum()
            centerness_loss = pos_flatten_predict_centerness_tensor.sum()

        loss = classification_loss + regression_loss + centerness_loss
        loss_values = dict(loss=loss.item(),
                           classification_loss=classification_loss.item(),
                           regression_loss=regression_loss.item(),
                           centerness_loss=centerness_loss.item())

        return dict(loss=loss,
                    loss_values=loss_values)

    def get_results(self, predict_outputs, *args):
        """
        get predicted bboxes from output feature maps
        :param predict_outputs:
        :param args:
        :return:
        """

        predict_classification_tensor, predict_regress_tensor, predict_centerness_tensor = predict_outputs
        num_samples = predict_classification_tensor.size(0)
        all_point_coordinates_list = self.generate_point_coordinates(self._head_indexes_to_feature_map_sizes)
        meta_batch = args[0]

        results = []
        for i in range(num_samples):
            nms_bboxes, nms_labels = self._get_results_for_single_image(predict_classification_tensor[i],
                                                                        predict_regress_tensor[i],
                                                                        predict_centerness_tensor[i],
                                                                        all_point_coordinates_list,
                                                                        meta_batch[i])

            # if empty, append []
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
                                      predicted_centerness,
                                      all_point_coordinates_list,
                                      meta_info):
        # 还需要加上图像长宽的限制，同时把bbox的尺度改变也纳入到这里
        split_list = [point_coordinates_per_level.size(0) for point_coordinates_per_level in all_point_coordinates_list]
        predicted_classification_split = predicted_classification.split(split_list, dim=0)
        predicted_regression_split = predicted_regression.split(split_list, dim=0)
        predicted_centerness_split = predicted_centerness.split(split_list, dim=0)
        image_resized_height = meta_info['resized_height']
        image_resized_width = meta_info['resized_width']

        predicted_classification_merge = list()
        predicted_bboxes_merge = list()
        predicted_centerness_merge = list()

        for i in range(len(split_list)):
            temp_predicted_classification = predicted_classification_split[i].sigmoid()
            temp_predicted_centerness = predicted_centerness_split[i].sigmoid()
            temp_predicted_regression = predicted_regression_split[i]
            temp_point_coordinates = all_point_coordinates_list[i].to(temp_predicted_regression.device)

            #  apply pre nms bbox limit to each head
            if 0 < self._pre_nms_bbox_limit < temp_predicted_classification.size(0):
                temp_max_scores, _ = (temp_predicted_classification * temp_predicted_centerness).max(dim=1)  # sort after centerness mask
                _, topk_indexes = temp_max_scores.topk(self._pre_nms_bbox_limit)
                temp_predicted_classification = temp_predicted_classification[topk_indexes]
                temp_predicted_centerness = temp_predicted_centerness[topk_indexes]
                temp_predicted_regression = temp_predicted_regression[topk_indexes]
                temp_point_coordinates = temp_point_coordinates[topk_indexes]

            predicted_classification_merge.append(temp_predicted_classification)
            predicted_centerness_merge.append(temp_predicted_centerness)
            predicted_bboxes_merge.append(self.distance2bbox(temp_point_coordinates, temp_predicted_regression, max_shape=(image_resized_height, image_resized_width)))

        predicted_classification_merge = torch.cat(predicted_classification_merge)
        # add BG label
        bg_label_padding = predicted_classification_merge.new_zeros(predicted_classification_merge.size(0), 1)
        predicted_classification_merge = torch.cat([predicted_classification_merge, bg_label_padding], dim=1)

        predicted_bboxes_merge = torch.cat(predicted_bboxes_merge)
        # change to original sizes
        predicted_bboxes_merge = predicted_bboxes_merge/meta_info['resize_scale']
        predicted_centerness_merge = torch.cat(predicted_centerness_merge).squeeze()

        nms_bboxes, nms_labels = multiclass_nms(
            multi_bboxes=predicted_bboxes_merge,
            multi_scores=predicted_classification_merge,
            score_thr=self._classification_threshold,
            nms_cfg=self._nms_cfg,
            max_num=self._post_nms_bbox_limit,
            score_factors=predicted_centerness_merge
        )

        return nms_bboxes, nms_labels

    def forward(self, x):

        backbone_outputs = self._backbone(x)

        neck_outputs = self._neck(backbone_outputs)

        classification_outputs, regression_outputs, centerness_outputs = self._head(neck_outputs)

        #  变换输出的dim和shape，转化成tensor输出
        #  tensor 中的dim n必须要保留，为了DP能够正常多卡
        classification_reformat_outputs = []
        regression_reformat_outputs = []
        centerness_reformat_outputs = []
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

            n, c, h, w = centerness_outputs[i].shape
            centerness_output = centerness_outputs[i].permute([0, 2, 3, 1])
            centerness_output = centerness_output.reshape((n, h * w, c))
            centerness_reformat_outputs.append(centerness_output)

        classification_output_tensor = torch.cat(classification_reformat_outputs, dim=1)
        regression_output_tensor = torch.cat(regression_reformat_outputs, dim=1)
        centerness_output_tensor = torch.cat(centerness_reformat_outputs, dim=1)

        return classification_output_tensor, regression_output_tensor, centerness_output_tensor
