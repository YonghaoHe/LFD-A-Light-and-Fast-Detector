# -*- coding: utf-8 -*-

import torch.nn as nn
__all__ = ['SimpleNeck']


def get_operator_from_cfg(operator_cfg):
    operator_cfg_copy = operator_cfg.copy()
    construct_str = 'nn.'
    construct_str += operator_cfg_copy.pop('type') + '('
    for k, v in operator_cfg_copy.items():
        construct_str += k + '=' + str(v) + ','
    construct_str += ')'

    return eval(construct_str)


class SimpleNeck(nn.Module):

    def __init__(self,
                 num_neck_channels,
                 num_input_channels_list,
                 num_input_strides_list,
                 norm_cfg=dict(type='BatchNorm2d'),
                 activation_cfg=dict(type='ReLU', inplace=True)):
        super(SimpleNeck, self).__init__()
        assert len(num_input_channels_list) == len(num_input_strides_list)
        self._num_neck_channels = num_neck_channels
        self._num_input_channels_list = num_input_channels_list
        self._num_input_strides_list = num_input_strides_list
        self._norm_cfg = norm_cfg
        self._activation_cfg = activation_cfg
        self._num_inputs = len(num_input_channels_list)

        for i, num_channels in enumerate(self._num_input_channels_list):
            temp_neck_layer_list = list()
            temp_neck_layer_list.append(nn.Conv2d(in_channels=num_channels, out_channels=self._num_neck_channels, kernel_size=1, stride=1, padding=0, bias=True if self._norm_cfg is None else False))
            if self._norm_cfg is not None:
                temp_norm_cfg = self._norm_cfg.copy()
                if temp_norm_cfg['type'] == 'BatchNorm2d':
                    temp_norm_cfg['num_features'] = self._num_neck_channels
                else:
                    temp_norm_cfg['num_channels'] = self._num_neck_channels
                temp_neck_layer_list.append(get_operator_from_cfg(temp_norm_cfg))
            temp_neck_layer_list.append(get_operator_from_cfg(self._activation_cfg))

            setattr(self, 'neck%d' % i, nn.Sequential(*temp_neck_layer_list))

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

    @property
    def num_output_strides_list(self):
        return self._num_input_strides_list

    def forward(self, inputs):
        assert len(inputs) == self._num_inputs

        outputs = list()
        for i in range(self._num_inputs):
            outputs.append(getattr(self, 'neck%d' % i)(inputs[i]))

        return tuple(outputs)
