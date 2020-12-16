# -*- coding: utf-8 -*-

"""
作者: 何泳澔
日期: 2020-05-22
模块文件: fpn.py
模块描述: 这个模块实现FPN的结构。

  000   -> 000 -> feature map
   ^        |
   |        v
 00000  -> 000 -> feature map
   ^        |
   |        v
0000000 -> 000 -> feature map

"""
import torch.nn as nn
__all__ = ['FPN']


class FPN(nn.Module):
    def __init__(self,
                 num_input_channels_list,
                 num_input_strides_list,
                 num_output_channels,
                 num_outputs,
                 extra_on_input=False,
                 extra_type='conv',
                 norm_on_lateral=False,
                 relu_on_lateral=False,
                 relu_before_extra=False,
                 norm_cfg=None,
                 ):
        super(FPN, self).__init__()
        assert num_outputs >= 1
        assert extra_type in ['conv', 'pooling']

        if norm_on_lateral:
            assert norm_cfg is not None
        if norm_cfg is not None:
            assert 'type' in norm_cfg
            assert norm_cfg['type'] in ['BN', 'GN']
            if norm_cfg['type'] == 'GN':
                assert 'num_groups' in norm_cfg

        assert len(num_input_channels_list) == len(num_input_strides_list), 'they must have the same length!'
        self._num_input_channels_list = num_input_channels_list
        self._num_input_strides_list = num_input_strides_list
        self._num_inputs = len(self._num_input_channels_list)
        self._num_output_channels = num_output_channels
        self._num_outputs = num_outputs
        self._extra_on_input = extra_on_input
        self._extra_type = extra_type
        self._norm_on_lateral = norm_on_lateral
        self._relu_on_lateral = relu_on_lateral
        self._relu_before_extra = relu_before_extra
        self._norm_cfg = norm_cfg

        # lateral convs
        for i in range(self._num_inputs):
            lateral = []
            if self._norm_on_lateral:
                lateral.append(nn.Conv2d(self._num_input_channels_list[i], self._num_output_channels, kernel_size=1, stride=1, padding=0, bias=False))
                lateral.append(nn.BatchNorm2d(num_features=self._num_output_channels) if self._norm_cfg['type'] == 'BN' else
                               nn.GroupNorm(num_groups=self._norm_cfg['num_groups'], num_channels=self._num_output_channels))
            else:
                lateral.append(nn.Conv2d(self._num_input_channels_list[i], self._num_output_channels, kernel_size=1, stride=1, padding=0, bias=True))

            if self._relu_on_lateral:
                lateral.append(nn.ReLU(inplace=False))

            setattr(self, 'lateral%d' % i, nn.Sequential(*lateral))

        # output convs
        for i in range(self._num_outputs):
            fpn_out = []

            if i == self._num_inputs:
                if self._extra_on_input:
                    if self._relu_before_extra:
                        fpn_out.append(nn.ReLU(inplace=True))
                    if self._extra_type == 'conv':
                        fpn_out.append(nn.Conv2d(self._num_input_channels_list[-1], self._num_output_channels, kernel_size=3, stride=2, padding=1, bias=True))
                    else:
                        fpn_out.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
                else:
                    if self._relu_before_extra:
                        fpn_out.append(nn.ReLU(inplace=True))
                    if self._extra_type == 'conv':
                        fpn_out.append(nn.Conv2d(self._num_output_channels, self._num_output_channels, kernel_size=3, stride=2, padding=1, bias=True))
                    else:
                        fpn_out.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            elif i > self._num_inputs:
                if self._relu_before_extra:
                    fpn_out.append(nn.ReLU(inplace=True))
                if self._extra_type == 'conv':
                    fpn_out.append(nn.Conv2d(self._num_output_channels, self._num_output_channels, kernel_size=3, stride=2, padding=1, bias=True))
                else:
                    fpn_out.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            else:
                fpn_out.append(nn.Conv2d(self._num_output_channels, self._num_output_channels, kernel_size=3, stride=1, padding=1, bias=True))

            setattr(self, 'fpn_out%d' % i, nn.Sequential(*fpn_out))

        self.__init_weights()

        # compute the output stride for subsequent use by FCOS(point stride)
        if self._num_outputs <= self._num_inputs:
            self._num_output_strides_list = self._num_input_strides_list[:self._num_outputs]
        else:
            self._num_output_strides_list = self._num_input_strides_list
            for i in range(self._num_outputs - self._num_inputs):
                self._num_output_strides_list.append(self._num_input_strides_list[-1] * 2**(i+1))

    @property
    def num_output_strides_list(self):
        return self._num_output_strides_list

    def __init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        assert len(inputs) == len(self._num_input_channels_list)

        lateral_outputs = []
        for i in range(self._num_inputs):
            lateral_outputs.append(getattr(self, 'lateral%d' % i)(inputs[i]))

        # top-down
        for i in range(self._num_inputs - 1, 0, -1):
            target_shape = lateral_outputs[i - 1].shape[2:]
            lateral_outputs[i - 1] += nn.Upsample(target_shape, mode='nearest')(lateral_outputs[i])

        # fpn output
        fpn_outputs = []
        for i in range(self._num_outputs):
            if i == self._num_inputs:
                if self._extra_on_input:
                    fpn_outputs.append(getattr(self, 'fpn_out%d' % i)(inputs[-1]))
                else:
                    fpn_outputs.append(getattr(self, 'fpn_out%d' % i)(fpn_outputs[-1]))
            elif i > self._num_inputs:
                fpn_outputs.append(getattr(self, 'fpn_out%d' % i)(fpn_outputs[-1]))
            else:
                fpn_outputs.append(getattr(self, 'fpn_out%d' % i)(lateral_outputs[i]))

        return tuple(fpn_outputs)
