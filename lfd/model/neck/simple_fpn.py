# -*- coding: utf-8 -*-

"""

"""
import torch.nn as nn

__all__ = ['SimpleFPN']


def get_operator_from_cfg(operator_cfg):
    operator_cfg_copy = operator_cfg.copy()
    construct_str = 'nn.'
    construct_str += operator_cfg_copy.pop('type') + '('
    for k, v in operator_cfg_copy.items():
        construct_str += k + '=' + str(v) + ','
    construct_str += ')'

    return eval(construct_str)


class SimpleFPN(nn.Module):
    def __init__(self,
                 num_input_channels_list,
                 num_input_strides_list,
                 num_output_channels,
                 num_outputs,
                 extra_on_input=False,
                 extra_type='conv',
                 norm_on_lateral=False,
                 relu_on_lateral=False,
                 relu_before_extra=True,
                 norm_cfg=None,
                 neighbouring_mode=False
                 ):
        super(SimpleFPN, self).__init__()
        assert num_outputs >= 1, 'the number of outputs must >= 1'
        assert extra_type in ['conv', 'pooling']

        if norm_on_lateral:
            assert norm_cfg is not None
        if norm_cfg is not None:
            assert 'type' in norm_cfg
            assert norm_cfg['type'] in ['BatchNorm2d', 'GroupNorm']
            if norm_cfg['type'] == 'GroupNorm':
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
        self._neighbouring_mode = neighbouring_mode

        if self._neighbouring_mode:
            assert self._num_outputs + 1 >= self._num_inputs

        # lateral convs
        for i in range(self._num_inputs):
            lateral = []
            if self._norm_on_lateral:
                lateral.append(nn.Conv2d(self._num_input_channels_list[i], self._num_output_channels, kernel_size=1, stride=1, padding=0, bias=False))
                temp_norm_cfg = self._norm_cfg.copy()
                if temp_norm_cfg['type'] == 'BatchNorm2d':
                    temp_norm_cfg['num_features'] = self._num_output_channels
                else:
                    temp_norm_cfg['num_channels'] = self._num_output_channels
                lateral.append(get_operator_from_cfg(temp_norm_cfg))
            else:
                lateral.append(nn.Conv2d(self._num_input_channels_list[i], self._num_output_channels, kernel_size=1, stride=1, padding=0, bias=True))

            if self._relu_on_lateral:
                lateral.append(nn.ReLU(inplace=False))

            setattr(self, 'lateral%d' % i, nn.Sequential(*lateral))

        # output convs or null
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
                pass  # do nothing

            setattr(self, 'fpn_out%d' % i, nn.Sequential(*fpn_out))

        self.__init_weights()

        # compute the output stride for subsequent use
        if self._num_outputs <= self._num_inputs:
            self._num_output_strides_list = self._num_input_strides_list[:self._num_outputs]
        else:
            self._num_output_strides_list = self._num_input_strides_list
            for i in range(self._num_outputs - self._num_inputs):
                self._num_output_strides_list.append(self._num_input_strides_list[-1] * 2 ** (i + 1))

    @property
    def num_output_strides_list(self):
        return self._num_output_strides_list

    def __init_weights(self):
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
        assert len(inputs) == self._num_inputs

        lateral_outputs = []
        for i in range(self._num_inputs):
            lateral_outputs.append(getattr(self, 'lateral%d' % i)(inputs[i]))

        if self._neighbouring_mode:
            # bottom-up
            for i in range(self._num_inputs - 1):
                target_shape = lateral_outputs[i].shape[2:]
                lateral_outputs[i] += nn.Upsample(target_shape, mode='nearest')(lateral_outputs[i + 1])
        else:
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
