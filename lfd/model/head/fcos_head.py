# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy

__all__ = ['FCOSHead']


class Scale(nn.Module):
    def __init__(self,
                 scale_factor=1.0):
        super(Scale, self).__init__()
        self._scale = nn.Parameter(torch.tensor(scale_factor, dtype=torch.float))

    def forward(self, x):
        return x * self._scale


class FCOSHead(nn.Module):

    def __init__(self,
                 num_classes,
                 num_input_channels,
                 num_head_channels=256,
                 num_heads=5,
                 num_layers=4,
                 norm_cfg=None):
        super(FCOSHead, self).__init__()
        if norm_cfg is not None:
            assert isinstance(norm_cfg, dict) and 'type' in norm_cfg
            assert norm_cfg['type'] in ['BatchNorm2d', 'GroupNorm']
            if norm_cfg['type'] == 'GroupNorm':
                assert 'num_groups' in norm_cfg

        self._num_classes = num_classes
        self._num_input_channels = num_input_channels
        self._num_head_channels = num_head_channels
        self._num_heads = num_heads
        self._num_layers = num_layers
        self._norm_cfg = norm_cfg

        self._classification_path = nn.ModuleList()
        self._regression_path = nn.ModuleList()
        for i in range(self._num_layers):

            num_in_channels = self._num_input_channels if i == 0 else self._num_head_channels
            if self._norm_cfg is not None:
                self._classification_path.append(nn.Conv2d(num_in_channels, self._num_head_channels, kernel_size=3, stride=1, padding=1, bias=False))

                if self._norm_cfg['type'] == 'BatchNorm2d':
                    norm = nn.BatchNorm2d(num_features=self._num_head_channels)
                elif self._norm_cfg['type'] == 'GroupNorm':
                    norm = nn.GroupNorm(num_groups=self._norm_cfg['num_groups'], num_channels=self._num_head_channels)
                else:
                    raise ValueError
                self._classification_path.append(norm)

                self._regression_path.append(nn.Conv2d(num_in_channels, self._num_head_channels, kernel_size=3, stride=1, padding=1, bias=False))

                if self._norm_cfg['type'] == 'BatchNorm2d':
                    norm = nn.BatchNorm2d(num_features=self._num_head_channels)
                elif self._norm_cfg['type'] == 'GroupNorm':
                    norm = nn.GroupNorm(num_groups=self._norm_cfg['num_groups'], num_channels=self._num_head_channels)
                else:
                    raise ValueError
                self._regression_path.append(norm)
            else:
                self._classification_path.append(nn.Conv2d(num_in_channels, self._num_head_channels, kernel_size=3, stride=1, padding=1, bias=True))

                self._regression_path.append(nn.Conv2d(num_in_channels, self._num_head_channels, kernel_size=3, stride=1, padding=1, bias=True))

            self._classification_path.append(nn.ReLU(inplace=True))
            self._regression_path.append(nn.ReLU(inplace=True))

        self._classification = nn.Conv2d(self._num_head_channels, self._num_classes, kernel_size=3, stride=1, padding=1, bias=True)
        self._centerness = nn.Conv2d(self._num_head_channels, 1, kernel_size=3, stride=1, padding=1, bias=True)
        self._regression = nn.Conv2d(self._num_head_channels, 4, kernel_size=3, stride=1, padding=1, bias=True)
        self._scales = nn.ModuleList([Scale(1.0) for _ in range(self._num_heads)])

        self.__init_weights()

    def __bias_init_with_prob(self, prior_prob):
        """
        ported from mmdetection
        :param prior_prob:
        :return:
        """
        bias_init = float(-numpy.log((1 - prior_prob) / prior_prob))
        return bias_init

    def __init_weights(self):
        for m in self._classification_path:
            if isinstance(m, nn.Conv2d):
                if hasattr(m, 'weight'):
                    nn.init.normal_(m.weight, mean=0, std=0.01)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if hasattr(m, 'weight'):
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        for m in self._regression_path:
            if isinstance(m, nn.Conv2d):
                if hasattr(m, 'weight'):
                    nn.init.normal_(m.weight, mean=0, std=0.01)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if hasattr(m, 'weight'):
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        bias_for_classification = self.__bias_init_with_prob(0.01)
        nn.init.normal_(self._classification.weight, mean=0, std=0.01)
        if hasattr(self._classification, 'bias') and self._classification.bias is not None:
            nn.init.constant_(self._classification.bias, bias_for_classification)

        nn.init.normal_(self._regression.weight, mean=0, std=0.01)
        if hasattr(self._regression, 'bias') and self._regression.bias is not None:
            nn.init.constant_(self._regression.bias, 0)

        nn.init.normal_(self._centerness.weight, mean=0, std=0.01)
        if hasattr(self._centerness, 'bias') and self._centerness.bias is not None:
            nn.init.constant_(self._centerness.bias, 0)

    def forward(self, inputs):
        assert isinstance(inputs, (list, tuple))
        assert len(inputs) == self._num_heads
        classification_outputs = []
        regression_outputs = []
        centerness_outputs = []

        for i, input_x in enumerate(inputs):
            classification_input = input_x
            regression_input = input_x
            for layer in self._classification_path:
                classification_input = layer(classification_input)
            classification_output = self._classification(classification_input)
            centerness_output = self._centerness(classification_input)

            for layer in self._regression_path:
                regression_input = layer(regression_input)
            regression_output = self._regression(regression_input)
            regression_output = self._scales[i](regression_output)
            regression_output = regression_output.float().exp()

            classification_outputs.append(classification_output)
            centerness_outputs.append(centerness_output)
            regression_outputs.append(regression_output)

        return classification_outputs, regression_outputs, centerness_outputs
