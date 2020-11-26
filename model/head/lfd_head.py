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
                 num_heads,
                 num_head_channels=128,
                 num_conv_layers=2,
                 activation_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='BatchNorm2d'),
                 classification_loss_type='FocalLoss',
                 share_head_flag=False,
                 ):
        super(LFDHead, self).__init__()
        assert classification_loss_type in ['BCEWithLogitsLoss', 'FocalLoss', 'CrossEntropyLoss']

        self._num_classes = num_classes
        self._num_input_channels = num_input_channels
        self._num_head_channels = num_head_channels
        self._num_conv_layers = num_conv_layers
        self._activation_cfg = activation_cfg
        self._norm_cfg = norm_cfg
        self._share_head_flag = share_head_flag
        self._num_heads = num_heads
        self._classification_loss_type = classification_loss_type

        for i in range(self._num_heads):
            if i == 0:
                classification_path, regression_path = self._build_head()
                setattr(self, 'head%d_classification_path' % i, classification_path)
                setattr(self, 'head%d_regression_path' % i, regression_path)
            else:
                if self._share_head_flag:
                    setattr(self, 'head%d_classification_path' % i, getattr(self, 'head%d_classification_path' % 0))
                    setattr(self, 'head%d_regression_path' % i, getattr(self, 'head%d_regression_path' % 0))
                else:
                    classification_path, regression_path = self._build_head()
                    setattr(self, 'head%d_classification_path' % i, classification_path)
                    setattr(self, 'head%d_regression_path' % i, regression_path)

        self._init_weights()

    def _build_head(self):
        classification_path = list()
        regression_path = list()

        for i in range(self._num_conv_layers):
            in_channels = self._num_input_channels if i == 0 else self._num_head_channels

            classification_path.append(
                nn.Conv2d(in_channels=in_channels, out_channels=self._num_head_channels, kernel_size=1, stride=1, padding=0, bias=True if self._norm_cfg is None else False)
            )
            if self._norm_cfg is not None:
                temp_norm_cfg = self._norm_cfg.copy()
                if temp_norm_cfg['type'] == 'BatchNorm2d':
                    temp_norm_cfg['num_features'] = self._num_head_channels
                else:
                    temp_norm_cfg['num_channels'] = self._num_head_channels
                classification_path.append(get_operator_from_cfg(temp_norm_cfg))
            classification_path.append(get_operator_from_cfg(self._activation_cfg))

            regression_path.append(
                nn.Conv2d(in_channels=in_channels, out_channels=self._num_head_channels, kernel_size=1, stride=1, padding=0, bias=True if self._norm_cfg is None else False)
            )
            if self._norm_cfg is not None:
                temp_norm_cfg = self._norm_cfg.copy()
                if temp_norm_cfg['type'] == 'BatchNorm2d':
                    temp_norm_cfg['num_features'] = self._num_head_channels
                else:
                    temp_norm_cfg['num_channels'] = self._num_head_channels
                regression_path.append(get_operator_from_cfg(temp_norm_cfg))
            regression_path.append(get_operator_from_cfg(self._activation_cfg))

        if self._classification_loss_type == 'CrossEntropyLoss':
            classification_path.append(nn.Conv2d(in_channels=self._num_head_channels, out_channels=self._num_classes + 1, kernel_size=1, stride=1, padding=0, bias=True))
        else:
            classification_path.append(nn.Conv2d(in_channels=self._num_head_channels, out_channels=self._num_classes, kernel_size=1, stride=1, padding=0, bias=True))

        regression_path.append(nn.Conv2d(in_channels=self._num_head_channels, out_channels=4, kernel_size=1, stride=1, padding=0, bias=True))

        classification_path = nn.Sequential(*classification_path)
        regression_path = nn.Sequential(*regression_path)
        return classification_path, regression_path

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
        assert len(inputs) == self._num_heads

        classification_outputs = []
        regression_outputs = []

        for i, temp_input in enumerate(inputs):
            classification_input = temp_input
            regression_input = temp_input

            classification_output = getattr(self, 'head%d_classification_path' % i)(classification_input)
            regression_output = getattr(self, 'head%d_regression_path' % i)(regression_input)

            classification_outputs.append(classification_output)
            regression_outputs.append(regression_output)

        return classification_outputs, regression_outputs
