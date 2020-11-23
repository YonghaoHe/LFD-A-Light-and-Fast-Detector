# -*- coding: utf-8 -*-
# author: Yonghao He
# description: 
import torch
import torch.nn as nn

__all__ = ['LFDHead']


def get_operator_from_cfg(operator_cfg):
    operator_cfg_copy = operator_cfg.copy()
    construct_str = 'nn.'
    construct_str += operator_cfg_copy.pop('type') + '('
    for k, v in operator_cfg_copy.items():
        construct_str += k + '=' + str(v) + ','
    construct_str += ')'

    return eval(construct_str)


class LFDHead(nn.Module):

    def __init__(self,
                 num_classes,
                 num_input_channels,
                 num_head_channels=128,
                 num_conv_layers=2,
                 activation_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='BatchNorm2d')
                 ):
        super(LFDHead, self).__init__()

        self._num_classes = num_classes
        self._num_input_channels = num_input_channels
        self._num_head_channels = num_head_channels
        self._num_conv_layers = num_conv_layers
        self._activation_cfg = activation_cfg
        self._norm_cfg = norm_cfg

        self._classification_path = list()
        self._regression_path = list()

        for i in range(self._num_conv_layers):
            in_channels = self._num_input_channels if i == 0 else self._num_head_channels

            self._classification_path.append(
                nn.Conv2d(in_channels=in_channels, out_channels=self._num_head_channels, kernel_size=1, stride=1, padding=0, bias=True if self._norm_cfg is None else False)
            )
            if self._norm_cfg is not None:
                temp_norm_cfg = self._norm_cfg.copy()
                if temp_norm_cfg['type'] == 'BatchNorm2d':
                    temp_norm_cfg['num_features'] = self._num_head_channels
                else:
                    temp_norm_cfg['num_channels'] = self._num_head_channels
                self._classification_path.append(get_operator_from_cfg(temp_norm_cfg))
            self._classification_path.append(get_operator_from_cfg(self._activation_cfg))

            self._regression_path.append(
                nn.Conv2d(in_channels=in_channels, out_channels=self._num_head_channels, kernel_size=1, stride=1, padding=0, bias=True if self._norm_cfg is None else False)
            )
            if self._norm_cfg is not None:
                temp_norm_cfg = self._norm_cfg.copy()
                if temp_norm_cfg['type'] == 'BatchNorm2d':
                    temp_norm_cfg['num_features'] = self._num_head_channels
                else:
                    temp_norm_cfg['num_channels'] = self._num_head_channels
                self._regression_path.append(get_operator_from_cfg(temp_norm_cfg))
            self._regression_path.append(get_operator_from_cfg(self._activation_cfg))

        self._classification_path = nn.Sequential(*self._classification_path)
        self._regression_path = nn.Sequential(*self._regression_path)

        self._classification = nn.Conv2d(in_channels=self._num_head_channels, out_channels=self._num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self._regression = nn.Conv2d(in_channels=self._num_head_channels, out_channels=4, kernel_size=1, stride=1, padding=0, bias=True)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, inputs):

        classification_outputs = []
        regression_outputs = []

        for i, temp_input in enumerate(inputs):
            classification_input = temp_input
            regression_input = temp_input

            classification_output = self._classification_path(classification_input)
            regression_output = self._regression_path(regression_input)

            classification_output = self._classification(classification_output)
            regression_output = self._regression(regression_output)

            classification_outputs.append(classification_output)
            regression_outputs.append(regression_output)

        return classification_outputs, regression_outputs
