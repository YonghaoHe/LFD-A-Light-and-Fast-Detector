# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn

__all__ = ['FastBlock', 'FasterBlock', 'FastestBlock', 'LFDResNet']


def get_operator_from_cfg(operator_cfg):
    operator_cfg_copy = operator_cfg.copy()
    construct_str = 'nn.'
    construct_str += operator_cfg_copy.pop('type') + '('
    for k, v in operator_cfg_copy.items():
        construct_str += k + '=' + str(v) + ','
    construct_str += ')'

    return eval(construct_str)


class FastBlock(nn.Module):
    def __init__(self,
                 num_input_channels,
                 num_block_channels,
                 stride=1,
                 downsample=None,
                 activation_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=None):
        super(FastBlock, self).__init__()
        if downsample is not None:
            assert stride == 2
        if norm_cfg is not None:
            assert norm_cfg['type'] in ['BatchNorm2d', 'GroupNorm']

        self._num_input_channel = num_input_channels
        self._num_block_channel = num_block_channels
        self._stride = stride
        self._activation_cfg = activation_cfg
        self._norm_cfg = norm_cfg
        self._downsample = downsample

        self._conv1 = nn.Conv2d(in_channels=self._num_input_channel, out_channels=self._num_block_channel, kernel_size=3, stride=self._stride, padding=1, bias=True if self._norm_cfg is None else False)
        if self._norm_cfg is not None:
            temp_norm_cfg = self._norm_cfg.copy()
            if temp_norm_cfg['type'] == 'BatchNorm2d':
                temp_norm_cfg['num_features'] = self._num_block_channel
            else:
                temp_norm_cfg['num_channels'] = self._num_block_channel
            self._norm1 = get_operator_from_cfg(temp_norm_cfg)

        self._activation = get_operator_from_cfg(self._activation_cfg)

        self._conv2 = nn.Conv2d(in_channels=self._num_block_channel, out_channels=self._num_block_channel, kernel_size=1, stride=1, padding=0, bias=True if self._norm_cfg is None else False)
        if self._norm_cfg is not None:
            temp_norm_cfg = self._norm_cfg.copy()
            if temp_norm_cfg['type'] == 'BatchNorm2d':
                temp_norm_cfg['num_features'] = self._num_block_channel
            else:
                temp_norm_cfg['num_channels'] = self._num_block_channel
            self._norm2 = get_operator_from_cfg(temp_norm_cfg)

        self._conv3 = nn.Conv2d(in_channels=self._num_block_channel, out_channels=self._num_block_channel, kernel_size=3, stride=1, padding=1, bias=True if self._norm_cfg is None else False)
        if self._norm_cfg is not None:
            temp_norm_cfg = self._norm_cfg.copy()
            if temp_norm_cfg['type'] == 'BatchNorm2d':
                temp_norm_cfg['num_features'] = self._num_block_channel
            else:
                temp_norm_cfg['num_channels'] = self._num_block_channel
            self._norm3 = get_operator_from_cfg(temp_norm_cfg)

    def forward(self, x):
        identity = x

        out = self._conv1(x)
        if self._norm_cfg is not None:
            out = self._norm1(out)
        out = self._activation(out)

        out = self._conv2(out)
        if self._norm_cfg is not None:
            out = self._norm2(out)
        out = self._activation(out)

        out = self._conv3(out)
        if self._norm_cfg is not None:
            out = self._norm3(out)

        if self._downsample is not None:
            identity = self._downsample(x)
        out += identity
        out = self._activation(out)

        return out


class FasterBlock(nn.Module):
    def __init__(self,
                 num_input_channels,
                 num_block_channels,
                 stride=1,
                 downsample=None,
                 activation_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=None):
        super(FasterBlock, self).__init__()
        if downsample is not None:
            assert stride == 2
        if norm_cfg is not None:
            assert norm_cfg['type'] in ['BatchNorm2d', 'GroupNorm']

        self._num_input_channel = num_input_channels
        self._num_block_channel = num_block_channels
        self._stride = stride
        self._activation_cfg = activation_cfg
        self._norm_cfg = norm_cfg
        self._downsample = downsample

        self._conv1 = nn.Conv2d(in_channels=self._num_input_channel, out_channels=self._num_block_channel, kernel_size=3, stride=self._stride, padding=1, bias=True if self._norm_cfg is None else False)
        if self._norm_cfg is not None:
            temp_norm_cfg = self._norm_cfg.copy()
            if temp_norm_cfg['type'] == 'BatchNorm2d':
                temp_norm_cfg['num_features'] = self._num_block_channel
            else:
                temp_norm_cfg['num_channels'] = self._num_block_channel
            self._norm1 = get_operator_from_cfg(temp_norm_cfg)

        self._activation = get_operator_from_cfg(self._activation_cfg)

        self._conv2 = nn.Conv2d(in_channels=self._num_block_channel, out_channels=self._num_block_channel, kernel_size=3, stride=1, padding=1, bias=True if self._norm_cfg is None else False)
        if self._norm_cfg is not None:
            temp_norm_cfg = self._norm_cfg.copy()
            if temp_norm_cfg['type'] == 'BatchNorm2d':
                temp_norm_cfg['num_features'] = self._num_block_channel
            else:
                temp_norm_cfg['num_channels'] = self._num_block_channel
            self._norm2 = get_operator_from_cfg(temp_norm_cfg)

    def forward(self, x):
        identity = x

        out = self._conv1(x)
        if self._norm_cfg is not None:
            out = self._norm1(out)
        out = self._activation(out)

        out = self._conv2(out)
        if self._norm_cfg is not None:
            out = self._norm2(out)

        if self._downsample is not None:
            identity = self._downsample(x)
        out += identity
        out = self._activation(out)

        return out


class FastestBlock(nn.Module):
    def __init__(self,
                 num_input_channels,
                 num_block_channels,
                 stride=1,
                 downsample=None,
                 activation_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=None):
        super(FastestBlock, self).__init__()
        if downsample is not None:
            assert stride == 2
        if norm_cfg is not None:
            assert norm_cfg['type'] in ['BatchNorm2d', 'GroupNorm']

        self._num_input_channel = num_input_channels
        self._num_block_channel = num_block_channels
        self._stride = stride
        self._activation_cfg = activation_cfg
        self._norm_cfg = norm_cfg
        self._downsample = downsample

        self._conv1 = nn.Conv2d(in_channels=self._num_input_channel, out_channels=self._num_block_channel // 2, kernel_size=3, stride=self._stride, padding=1, bias=True if self._norm_cfg is None else False)
        if self._norm_cfg is not None:
            temp_norm_cfg = self._norm_cfg.copy()
            if temp_norm_cfg['type'] == 'BatchNorm2d':
                temp_norm_cfg['num_features'] = self._num_block_channel // 2
            else:
                temp_norm_cfg['num_channels'] = self._num_block_channel // 2
            self._norm1 = get_operator_from_cfg(temp_norm_cfg)

        self._activation = get_operator_from_cfg(self._activation_cfg)

        self._conv2 = nn.Conv2d(in_channels=self._num_block_channel // 2, out_channels=self._num_block_channel, kernel_size=3, stride=1, padding=1, bias=True if self._norm_cfg is None else False)
        if self._norm_cfg is not None:
            temp_norm_cfg = self._norm_cfg.copy()
            if temp_norm_cfg['type'] == 'BatchNorm2d':
                temp_norm_cfg['num_features'] = self._num_block_channel
            else:
                temp_norm_cfg['num_channels'] = self._num_block_channel
            self._norm2 = get_operator_from_cfg(temp_norm_cfg)

    def forward(self, x):
        identity = x

        out = self._conv1(x)
        if self._norm_cfg is not None:
            out = self._norm1(out)
        out = self._activation(out)

        out = self._conv2(out)
        if self._norm_cfg is not None:
            out = self._norm2(out)

        if self._downsample is not None:
            identity = self._downsample(x)
        out += identity
        out = self._activation(out)

        return out


class LFDResNet(nn.Module):

    # default body architectures are set, or you can specify your own architectures
    # by default, all default bodies contain 5 stages
    mode_to_body_architectures = {
        'fast': [4, 2, 2, 1, 1],
        'faster': [2, 1, 1, 1, 1],
        'fastest': [2, 1, 1, 1, 1]
    }
    mode_to_body_channels = {
        'fast': [64, 64, 128, 256, 512],
        'faster': [64, 64, 128, 128, 256],
        'fastest': [32, 32, 64, 64, 128]
    }

    def __init__(self,
                 block_mode='fast',  # affect block type
                 stem_mode='fast',  # affect stem type
                 body_mode='fast',  # affect body architecture
                 input_channels=3,
                 stem_channels=64,
                 body_architecture=None,
                 body_channels=None,
                 out_indices=((0, 3), (1, 1), (2, 1), (3, 0), (4, 0)),
                 frozen_stages=-1,
                 activation_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='BatchNorm2d'),
                 init_with_weight_file=None,
                 norm_eval=False
                 ):
        super(LFDResNet, self).__init__()
        assert block_mode in ['fast', 'faster', 'fastest']
        assert stem_mode in ['fast', 'faster', 'fastest']
        assert body_mode in ['fast', 'faster', 'fastest', None]
        # when body mode is None, body_architecture and body_channels must be provided
        if body_mode is None:
            assert body_architecture is not None
            assert body_channels is not None

        # get body architecture
        if body_mode is not None:
            self._body_architecture = self.mode_to_body_architectures[body_mode]
            self._body_channels = self.mode_to_body_channels[body_mode] if body_channels is None else body_channels
        else:
            self._body_architecture = body_architecture
            self._body_channels = body_channels
        assert len(self._body_architecture) == len(self._body_channels)

        self._block_mode = block_mode
        self._stem_mode = stem_mode

        self._input_channels = input_channels
        self._stem_channels = stem_channels

        out_indices = sorted(out_indices, key=lambda x: (x[0], x[1]))
        self._out_indices = out_indices
        for index in self._out_indices:
            assert 0 <= index[0] < len(self._body_architecture)
            assert 0 <= index[1] < self._body_architecture[index[0]]
        max_stage_index = max([index[0] for index in self._out_indices])
        # adjust body according to the max stage index
        self._body_architecture = self._body_architecture[:max_stage_index + 1]
        self._body_channels = self._body_channels[:max_stage_index + 1]

        assert frozen_stages <= max_stage_index + 1
        self._frozen_stages = frozen_stages
        self._activation_cfg = activation_cfg
        self._norm_cfg = norm_cfg
        self._init_with_weight_file = init_with_weight_file
        self._norm_eval = norm_eval

        self._make_stem()
        self._make_stages()

        self._init_weights()
        if self._init_with_weight_file is not None:
            assert isinstance(self._init_with_weight_file, str), 'weight file must be the string path of the file!'
            self._init_with_pretrained_weights()

        # obtain out channels based on out indices; obtain strides for each output map
        # both of them will be used by subsequent modules
        self._num_output_channels_list = []
        self._num_output_strides_list = []
        stem_stride = 2 if self._stem_mode == 'fast' else 4
        for i, (stage_index, _) in enumerate(self._out_indices):
            self._num_output_channels_list.append(self._body_channels[stage_index])
            self._num_output_strides_list.append(stem_stride * (2 ** (stage_index + 1)))

    @property
    def num_output_channels_list(self):
        return self._num_output_channels_list

    @property
    def num_output_strides_list(self):
        return self._num_output_strides_list

    def _init_with_pretrained_weights(self):

        assert os.path.isfile(self._init_with_weight_file), 'pretrained weight file [{}] does not exist!'.format(self.init_with_weight_file)

        weights = torch.load(self._init_with_weight_file)

        # rename keys of 'state_dict' (pth from pre-train may contain 'backbone')
        new_state_dict = dict()
        for k in weights['state_dict']:
            v = weights['state_dict'][k]
            k_splits = k.split('.')
            if 'backbone' in k_splits[0]:
                del k_splits[0]

            new_k = '.'.join(k_splits)
            new_state_dict[new_k] = v

        missing_keys, unexpected_keys = self.load_state_dict(new_state_dict, strict=False)
        if missing_keys:
            print('[WARNING: ResNet pretrained weights load] missing keys:')
            for i, key in enumerate(missing_keys):
                print(key + '\t', end='') if i < len(missing_keys) - 1 else print(key)

        if unexpected_keys:
            print('[WARNING: ResNet pretrained weights load] unexpected keys:')
            for i, key in enumerate(unexpected_keys):
                print(key + '\t', end='') if i < len(unexpected_keys) - 1 else print(key)

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

    def _make_stem(self):
        stem_layer_list = list()
        if self._stem_mode == 'fast':
            stem_layer_list.append(nn.Conv2d(in_channels=self._input_channels, out_channels=self._stem_channels, kernel_size=3, stride=2, padding=1, bias=True if self._norm_cfg is None else False))
            if self._norm_cfg is not None:
                temp_norm_cfg = self._norm_cfg.copy()
                if temp_norm_cfg['type'] == 'BatchNorm2d':
                    temp_norm_cfg['num_features'] = self._stem_channels
                else:
                    temp_norm_cfg['num_channels'] = self._stem_channels
                stem_layer_list.append(get_operator_from_cfg(temp_norm_cfg))
            stem_layer_list.append(get_operator_from_cfg(self._activation_cfg))
            stem_layer_list.append(nn.Conv2d(in_channels=self._stem_channels, out_channels=self._stem_channels, kernel_size=1, stride=1, padding=0, bias=True if self._norm_cfg is None else False))
            if self._norm_cfg is not None:
                temp_norm_cfg = self._norm_cfg.copy()
                if temp_norm_cfg['type'] == 'BatchNorm2d':
                    temp_norm_cfg['num_features'] = self._stem_channels
                else:
                    temp_norm_cfg['num_channels'] = self._stem_channels
                stem_layer_list.append(get_operator_from_cfg(temp_norm_cfg))
            stem_layer_list.append(get_operator_from_cfg(self._activation_cfg))

        elif self._stem_mode == 'faster':
            stem_layer_list.append(nn.Conv2d(in_channels=self._input_channels, out_channels=self._stem_channels, kernel_size=3, stride=2, padding=1, bias=True if self._norm_cfg is None else False))
            if self._norm_cfg is not None:
                temp_norm_cfg = self._norm_cfg.copy()
                if temp_norm_cfg['type'] == 'BatchNorm2d':
                    temp_norm_cfg['num_features'] = self._stem_channels
                else:
                    temp_norm_cfg['num_channels'] = self._stem_channels
                stem_layer_list.append(get_operator_from_cfg(temp_norm_cfg))
            stem_layer_list.append(get_operator_from_cfg(self._activation_cfg))
            stem_layer_list.append(nn.Conv2d(in_channels=self._stem_channels, out_channels=self._stem_channels, kernel_size=1, stride=1, padding=0, bias=True if self._norm_cfg is None else False))
            if self._norm_cfg is not None:
                temp_norm_cfg = self._norm_cfg.copy()
                if temp_norm_cfg['type'] == 'BatchNorm2d':
                    temp_norm_cfg['num_features'] = self._stem_channels
                else:
                    temp_norm_cfg['num_channels'] = self._stem_channels
                stem_layer_list.append(get_operator_from_cfg(temp_norm_cfg))
            stem_layer_list.append(get_operator_from_cfg(self._activation_cfg))

            stem_layer_list.append(nn.Conv2d(in_channels=self._stem_channels, out_channels=self._stem_channels, kernel_size=3, stride=2, padding=1, bias=True if self._norm_cfg is None else False))
            if self._norm_cfg is not None:
                temp_norm_cfg = self._norm_cfg.copy()
                if temp_norm_cfg['type'] == 'BatchNorm2d':
                    temp_norm_cfg['num_features'] = self._stem_channels
                else:
                    temp_norm_cfg['num_channels'] = self._stem_channels
                stem_layer_list.append(get_operator_from_cfg(temp_norm_cfg))
            stem_layer_list.append(get_operator_from_cfg(self._activation_cfg))
            stem_layer_list.append(nn.Conv2d(in_channels=self._stem_channels, out_channels=self._stem_channels, kernel_size=1, stride=1, padding=0, bias=True if self._norm_cfg is None else False))
            if self._norm_cfg is not None:
                temp_norm_cfg = self._norm_cfg.copy()
                if temp_norm_cfg['type'] == 'BatchNorm2d':
                    temp_norm_cfg['num_features'] = self._stem_channels
                else:
                    temp_norm_cfg['num_channels'] = self._stem_channels
                stem_layer_list.append(get_operator_from_cfg(temp_norm_cfg))
            stem_layer_list.append(get_operator_from_cfg(self._activation_cfg))

        elif self._stem_mode == 'fastest':
            stem_layer_list.append(nn.Conv2d(in_channels=self._input_channels, out_channels=self._stem_channels // 2, kernel_size=3, stride=2, padding=1, bias=True if self._norm_cfg is None else False))
            if self._norm_cfg is not None:
                temp_norm_cfg = self._norm_cfg.copy()
                if temp_norm_cfg['type'] == 'BatchNorm2d':
                    temp_norm_cfg['num_features'] = self._stem_channels // 2
                else:
                    temp_norm_cfg['num_channels'] = self._stem_channels // 2
                stem_layer_list.append(get_operator_from_cfg(temp_norm_cfg))
            stem_layer_list.append(get_operator_from_cfg(self._activation_cfg))

            stem_layer_list.append(nn.Conv2d(in_channels=self._stem_channels // 2, out_channels=self._stem_channels, kernel_size=3, stride=2, padding=1, bias=True if self._norm_cfg is None else False))
            if self._norm_cfg is not None:
                temp_norm_cfg = self._norm_cfg.copy()
                if temp_norm_cfg['type'] == 'BatchNorm2d':
                    temp_norm_cfg['num_features'] = self._stem_channels
                else:
                    temp_norm_cfg['num_channels'] = self._stem_channels
                stem_layer_list.append(get_operator_from_cfg(temp_norm_cfg))
            stem_layer_list.append(get_operator_from_cfg(self._activation_cfg))

        else:
            raise ValueError('Unsupported stem_mode!')

        self._stem = nn.Sequential(*stem_layer_list)

    def _make_stages(self):

        if self._block_mode == 'fast':
            self._block = FastBlock
        elif self._block_mode == 'faster':
            self._block = FasterBlock
        elif self._block_mode == 'fastest':
            self._block = FastestBlock
        else:
            raise ValueError('Unsupported block mode!')

        for i, num_blocks in enumerate(self._body_architecture):
            num_stage_channels = self._body_channels[i]
            stage_list = nn.ModuleList()
            in_channels = self._stem_channels if i == 0 else self._body_channels[i-1]
            for j in range(num_blocks):

                if j == 0:
                    downsample_list = list()
                    downsample_list.append(nn.Conv2d(in_channels=in_channels, out_channels=num_stage_channels, kernel_size=1, stride=2, padding=0, bias=True if self._norm_cfg is None else False))
                    if self._norm_cfg is not None:
                        temp_norm_cfg = self._norm_cfg.copy()
                        if temp_norm_cfg['type'] == 'BatchNorm2d':
                            temp_norm_cfg['num_features'] = num_stage_channels
                        else:
                            temp_norm_cfg['num_channels'] = num_stage_channels
                        downsample_list.append(get_operator_from_cfg(temp_norm_cfg))
                    downsample = nn.Sequential(*downsample_list)
                    stage_list.append(self._block(num_input_channels=in_channels, num_block_channels=num_stage_channels, stride=2, downsample=downsample, activation_cfg=self._activation_cfg, norm_cfg=self._norm_cfg))
                else:
                    stage_list.append(self._block(num_input_channels=num_stage_channels, num_block_channels=num_stage_channels, stride=1, downsample=None, activation_cfg=self._activation_cfg, norm_cfg=self._norm_cfg))

            setattr(self, 'stage%d' % i, stage_list)

    def _freeze_stages(self):
        if self._frozen_stages > 0:
            self._stem.eval()
            for param in self._stem.parameters():
                param.requires_grad = False

        for i in range(0, self._frozen_stages):
            for j in range(self._body_architecture[i]):
                m = getattr(self, 'stage%d' % i)[j]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def forward(self, x):

        x = self._stem(x)

        outs = []

        for i, num_blocks in enumerate(self._body_architecture):
            for j in range(num_blocks):
                block = getattr(self, 'stage%d' % i)[j]
                x = block(x)
                if (i, j) in self._out_indices:
                    outs.append(x)

        return tuple(outs)

    def train(self, mode=True):
        super(LFDResNet, self).train(mode)
        self._freeze_stages()
        if mode and self._norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
