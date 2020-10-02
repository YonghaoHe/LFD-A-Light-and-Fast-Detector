# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

__all__ = ['ResNet']


def get_norm_layer(norm_cfg, num_channels, suffix=1):
    norm_name = norm_cfg['type'].lower() + str(suffix)
    norm = nn.BatchNorm2d(num_features=num_channels) if norm_cfg['type'] == 'BN' else nn.GroupNorm(num_groups=norm_cfg['num_groups'], num_channels=num_channels)

    return norm_name, norm


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 num_input_channels,
                 num_block_channels,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 dcn=None,
                 plugins=None,
                 style='pytorch',
                 with_cp=False,
                 norm_cfg=dict(type='BN')  # GN can be set as dict(type='GN', num_groups=32)
                 ):
        super(BasicBlock, self).__init__()
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'
        assert norm_cfg['type'] in ['BN', 'GN']
        if norm_cfg['type'] == 'GN':
            assert 'num_groups' in norm_cfg

        self.norm1_name, norm1 = get_norm_layer(norm_cfg, num_channels=num_block_channels, suffix=1)
        self.add_module(self.norm1_name, norm1)

        self.norm2_name, norm2 = get_norm_layer(norm_cfg, num_channels=num_block_channels, suffix=2)
        self.add_module(self.norm2_name, norm2)

        self.conv1 = nn.Conv2d(num_input_channels, num_block_channels, 3, stride=stride, padding=dilation, dilation=dilation, bias=False)

        self.conv2 = nn.Conv2d(num_block_channels, num_block_channels, 3, padding=1, bias=False)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 num_input_channels,
                 num_block_channels,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),  # GN can be set as dict(type='GN', num_groups=32)
                 dcn=None, ):
        """Bottleneck block for ResNet.
        If style is "pytorch", the stride-two layer is the 3x3 conv layer,
        if it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottleneck, self).__init__()
        assert style in ['pytorch', 'caffe']
        assert dcn is None or isinstance(dcn, dict)
        assert norm_cfg['type'] in ['BN', 'GN']
        if norm_cfg['type'] == 'GN':
            assert 'num_groups' in norm_cfg

        self.num_input_channels = num_input_channels
        self.num_block_channels = num_block_channels
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.dcn = dcn
        self.with_dcn = dcn is not None

        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.norm1_name, norm1 = get_norm_layer(norm_cfg, num_channels=num_block_channels, suffix=1)
        self.add_module(self.norm1_name, norm1)
        self.norm2_name, norm2 = get_norm_layer(norm_cfg, num_channels=num_block_channels, suffix=2)
        self.add_module(self.norm2_name, norm2)
        self.norm3_name, norm3 = get_norm_layer(norm_cfg, num_channels=num_block_channels * self.expansion, suffix=3)
        self.add_module(self.norm3_name, norm3)

        self.conv1 = nn.Conv2d(num_input_channels, num_block_channels, kernel_size=1, stride=self.conv1_stride, bias=False)

        # 第二个卷积是否采用DCN，目前此处暂不支持DCN
        fallback_on_stride = False
        if self.with_dcn:
            fallback_on_stride = dcn.pop('fallback_on_stride', False)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = nn.Conv2d(num_block_channels,
                                   num_block_channels,
                                   kernel_size=3,
                                   stride=self.conv2_stride,
                                   padding=dilation,
                                   dilation=dilation,
                                   bias=False)
        else:
            assert False, 'DCN is not supported currently.'
            # assert self.conv_cfg is None, 'conv_cfg must be None for DCN'
            # self.conv2 = build_conv_layer(
            #     dcn,
            #     num_block_channels,
            #     num_block_channels,
            #     kernel_size=3,
            #     stride=self.conv2_stride,
            #     padding=dilation,
            #     dilation=dilation,
            #     bias=False)

        self.conv3 = nn.Conv2d(num_block_channels, num_block_channels * self.expansion, kernel_size=1, bias=False)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        return getattr(self, self.norm3_name)

    def forward(self, x):

        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """ResNet backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        in_channels (int): Number of input image channels. Normally 3.
        num_stages (int): Resnet stages, normally 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        norm_cfg (dict): Dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        plugins (list[dict]): List of plugins for stages, each dict contains:

            - cfg (dict, required): Cfg dict to build plugin.
            - position (str, required): Position inside block to insert
              plugin, options are 'after_conv1', 'after_conv2', 'after_conv3'.
            - stages (tuple[bool], optional): Stages to apply plugin, length
              should be same as 'num_stages'.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity.

    Example:
        >>> from mmdet.models import ResNet
        >>> import torch
        >>> self = ResNet(depth=18)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 64, 8, 8)
        (1, 128, 4, 4)
        (1, 256, 2, 2)
        (1, 512, 1, 1)
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 in_channels=3,
                 base_channels=64,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 # out_indices的每个元素依次为stage的索引和block的索引，stage的索引为1-based，block的索引为0-based。 这样便于更加灵活拿取backbone中的任意block的输出
                 out_indices=((1, 1), (2, 1), (3, 1), (4, 1)),
                 style='pytorch',
                 deep_stem=False,
                 avg_down=False,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,  # 注意这个参数
                 dcn=None,
                 stage_with_dcn=(False, False, False, False),
                 with_cp=False,
                 zero_init_residual=True,
                 init_with_weight_file=None):
        super(ResNet, self).__init__()
        assert depth in self.arch_settings, 'depth: %s is not supported!' % str(depth)
        assert norm_cfg['type'] in ['BN', 'GN']
        if norm_cfg['type'] == 'GN':
            assert 'num_groups' in norm_cfg

        self.depth = depth
        self.base_channels = base_channels
        self.num_stages = max([stage_index for (stage_index, block_index) in out_indices])  # get num_stages from out_indices
        assert 1 <= self.num_stages <= 4
        self.strides = strides[:self.num_stages]
        self.dilations = dilations[:self.num_stages]

        self.zero_init_residual = zero_init_residual
        self.init_with_weight_file = init_with_weight_file
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:self.num_stages]

        self.out_indices = list(out_indices)
        for (stage_index, block_index) in self.out_indices:
            assert 1 <= stage_index <= self.num_stages
            assert 0 <= block_index < self.stage_blocks[stage_index - 1]
        # sort the out indices in ascend
        self.out_indices.sort(key=lambda item: (item[0], item[1]))

        self.style = style
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.dcn = dcn
        self.stage_with_dcn = stage_with_dcn
        if dcn is not None:
            assert len(stage_with_dcn) == self.num_stages

        self.inplanes = base_channels

        # construct stem layers
        self._make_stem_layer(in_channels, base_channels)

        # construct stages
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = self.strides[i]
            dilation = self.dilations[i]
            dcn = self.dcn if self.stage_with_dcn[i] else None
            planes = base_channels * 2 ** i

            # CAUTION:  stage index is 1-based, inner block index is 0-based
            self._make_stage(i + 1, num_blocks, self.inplanes, planes, stride, dilation, dcn)
            self.inplanes = planes * self.block.expansion

        self._init_weights()

        if self.init_with_weight_file is not None:
            assert isinstance(self.init_with_weight_file, str), 'weight file must be the string path of the file!'
            self._init_with_pretrained_weights()

        # obtain out channels based on out indices; obtain strides for each output map
        # both of them will be used by subsequent modules
        self._num_output_channels_list = []
        self._num_output_strides_list = []
        for i, (stage_index, _) in enumerate(self.out_indices):
            self._num_output_channels_list.append(self.block.expansion * self.base_channels * 2 ** (stage_index - 1))
            self._num_output_strides_list.append(2 ** (stage_index + 1))

    @property
    def num_output_channels_list(self):
        return self._num_output_channels_list

    @property
    def num_output_strides_list(self):
        return self._num_output_strides_list

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self, in_channels, base_channels):
        if self.deep_stem:  # three 3x3 conv layers

            self.stem = nn.Sequential(

                nn.Conv2d(in_channels, base_channels // 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(num_features=base_channels // 2) if self.norm_cfg['type'] == 'BN' else nn.GroupNorm(num_groups=self.norm_cfg['num_groups'], num_channels=base_channels // 2),
                nn.ReLU(inplace=True),

                nn.Conv2d(base_channels // 2, base_channels // 2, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(num_features=base_channels // 2) if self.norm_cfg['type'] == 'BN' else nn.GroupNorm(num_groups=self.norm_cfg['num_groups'], num_channels=base_channels // 2),
                nn.ReLU(inplace=True),

                nn.Conv2d(base_channels // 2, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(num_features=base_channels) if self.norm_cfg['type'] == 'BN' else nn.GroupNorm(num_groups=self.norm_cfg['num_groups'], num_channels=base_channels),
                nn.ReLU(inplace=True))
        else:

            self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False)
            self.norm1_name, norm1 = get_norm_layer(self.norm_cfg, num_channels=base_channels, suffix=1)
            self.add_module(self.norm1_name, norm1)
            self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _make_stage(self, stage_index, num_blocks, inplanes, planes, stride, dilation, dcn):

        downsample = None
        if stride != 1 or inplanes != planes * self.block.expansion:
            downsample = []
            conv_stride = stride
            if self.avg_down and stride != 1:
                conv_stride = 1
                downsample.append(
                    nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False)
                )

            downsample.append(
                nn.Conv2d(inplanes, planes * self.block.expansion, kernel_size=1, stride=conv_stride, bias=False)
            )
            downsample.append(
                nn.BatchNorm2d(num_features=planes * self.block.expansion) if self.norm_cfg['type'] == 'BN' else nn.GroupNorm(num_groups=self.norm_cfg['num_groups'], num_channels=planes * self.block.expansion)
            )
            downsample = nn.Sequential(*downsample)

        module_list = nn.ModuleList()

        module_list.append(self.block(inplanes, planes, stride=stride, dilation=dilation, style=self.style, downsample=downsample, dcn=dcn, norm_cfg=self.norm_cfg))

        inplanes = planes * self.block.expansion
        for i in range(1, num_blocks):
            module_list.append(self.block(inplanes, planes, stride=1, dilation=dilation, style=self.style, dcn=dcn, norm_cfg=self.norm_cfg))

        setattr(self, 'layer%d' % stage_index, module_list)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            if self.deep_stem:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            else:
                self.norm1.eval()
                for m in [self.conv1, self.norm1]:
                    for param in m.parameters():
                        param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            for j in range(self.stage_blocks[i - 1]):
                m = getattr(self, 'layer%d' % i)[j]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def _init_with_pretrained_weights(self):

        assert os.path.isfile(self.init_with_weight_file), 'pretrained weight file [{}] does not exist!'.format(self.init_with_weight_file)

        weights = torch.load(self.init_with_weight_file)

        missing_keys, unexpected_keys = self.load_state_dict(weights['state_dict'], strict=False)
        if missing_keys:
            print('[WARNING: ResNet pretrained weights load] missing keys:')
            for i, key in enumerate(missing_keys):
                print(key + '\t', end='')
                if (i + 1) % 10 == 0:
                    print()

        if unexpected_keys:
            print('[WARNING: ResNet pretrained weights load] unexpected keys:')
            for i, key in enumerate(unexpected_keys):
                print(key + '\t', end='')
                if (i + 1) % 10 == 0:
                    print()
        print()

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

        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.norm3.weight, 0)
                    nn.init.constant_(m.norm3.bias, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.norm2.weight, 0)
                    nn.init.constant_(m.norm2.bias, 0)

    def forward(self, x):
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        outs = []

        for i in range(1, self.num_stages + 1):
            for j in range(self.stage_blocks[i - 1]):
                block = getattr(self, 'layer%d' % i)[j]
                x = block(x)
                if (i, j) in self.out_indices:
                    outs.append(x)

        return tuple(outs)

    def train(self, mode=True):
        super(ResNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
