import torch
import torch.nn as nn
from torch.nn import functional as F
from ssd.modeling import registry
from ssd.utils.model_zoo import load_state_dict_from_url
from timm.models.layers import SelectAdaptivePool2d
from timm.models.layers.activations import hard_sigmoid
from timm.models import resume_checkpoint
from .builder_util import *
from .build_childnet import *

arch_list = [[0], [3], [3, 3], [3, 3], [3, 3, 3], [3, 3], [0]]
image_size = 160
stem = ['ds_r1_k3_s1_e1_c16_se0.25', 'cn_r1_k1_s1_c320_se0.25']
choice_block_pool = ['ir_r1_k3_s2_e4_c24_se0.25',
                     'ir_r1_k5_s2_e4_c40_se0.25',
                     'ir_r1_k3_s2_e6_c80_se0.25',
                     'ir_r1_k3_s1_e6_c96_se0.25',
                     'ir_r1_k5_s2_e6_c192_se0.25']
arch_def = [[stem[0]]] + [[choice_block_pool[idx]
                           for repeat_times in range(len(arch_list[idx + 1]))]
                          for idx in range(len(choice_block_pool))] + [[stem[1]]]


def decode_arch_def(
        arch_def,
        depth_multiplier=1.0,
        depth_trunc='ceil',
        experts_multiplier=1):
    arch_args = []
    for stack_idx, block_strings in enumerate(arch_def):
        assert isinstance(block_strings, list)
        stack_args = []
        repeats = []
        for block_str in block_strings:
            assert isinstance(block_str, str)
            ba, rep = decode_block_str(block_str)
            if ba.get('num_experts', 0) > 0 and experts_multiplier > 1:
                ba['num_experts'] *= experts_multiplier
            stack_args.append(ba)
            repeats.append(rep)
        arch_args.append(
            scale_stage_depth(
                stack_args,
                repeats,
                depth_multiplier,
                depth_trunc))
    return arch_args


class ChildNet(nn.Module):

    def __init__(
            self,
            block_args,
            num_classes=21,
            in_chans=3,
            stem_size=16,
            num_features=1280,
            head_bias=True,
            channel_multiplier=1.0,
            pad_type='',
            act_layer=nn.ReLU,
            drop_rate=0.,
            drop_path_rate=0.,
            se_kwargs=None,
            norm_layer=nn.BatchNorm2d,
            norm_kwargs=None,
            global_pool='avg',
            logger=None,
            verbose=False):
        super(ChildNet, self).__init__()

        self.num_classes = num_classes
        self.num_features = num_features
        self.drop_rate = drop_rate
        self._in_chs = in_chans
        self.logger = logger

        # Stem
        stem_size = round_channels(stem_size, channel_multiplier)
        self.conv_stem = create_conv2d(
            self._in_chs, stem_size, 3, stride=2, padding=pad_type)
        self.bn1 = norm_layer(stem_size, **norm_kwargs)
        self.act1 = act_layer(inplace=True)
        self._in_chs = stem_size

        # Middle stages (IR/ER/DS Blocks)
        builder = ChildNetBuilder(
            channel_multiplier, 8, None, 32, pad_type, act_layer, se_kwargs,
            norm_layer, norm_kwargs, drop_path_rate, verbose=verbose)
        self.blocks = nn.Sequential(*builder(self._in_chs, block_args))
        # self.blocks = builder(self._in_chs, block_args)
        self._in_chs = builder.in_chs

        # Head + Pooling
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.conv_head = create_conv2d(
            self._in_chs,
            self.num_features,
            1,
            padding=pad_type,
            bias=head_bias)
        self.act2 = act_layer(inplace=True)

        # Classifier

        self.classifi = nn.Linear(
            self.num_features *
            self.global_pool.feat_mult(),
            self.num_classes)


        efficientnet_init_weights(self)


    '''
    def get_classifi(self):
        return self.classifi

    def reset_classifi(self, num_classes, global_pool='avg'):
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.num_classes = num_classes
        self.classifi = nn.Linear(
            self.num_features * self.global_pool.feat_mult(),
            num_classes) if self.num_classes else None
    


    def forward_features(self, x):
        # architecture = [[0], [], [], [], [], [0]]
        x = self.conv_stem(x)
        print(x.shape)
        x = self.bn1(x)
        x = self.act1(x)
        #x = self.blocks(x)
        for i in self.blocks:
            x = i(x)
            print(x.shape)
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        return x
    '''

    def forward(self, x):
        features = []
        #x = self.forward_features(x)
        x = self.conv_stem(x)
        #print(x.shape)
        x = self.bn1(x)
        x = self.act1(x)
        #print(len(self.blocks))
        x = self.blocks[0](x)
        x = self.blocks[1](x)
        #features.append(x)  # 40
        x = self.blocks[2](x)
        features.append(x)  # 20
        x = self.blocks[3](x)
        x = self.blocks[4](x)
        features.append(x)  # 10
        x = self.blocks[5](x)
        x = self.blocks[6](x)
        features.append(x)  # 5
        # features 特征图尺寸为 20,10,5
        return tuple(features)


def gen_childnet(arch_list, arch_def, **kwargs):
    # arch_list = [[0], [], [], [], [], [0]]
    choices = {'kernel_size': [3, 5, 7], 'exp_ratio': [4, 6]}
    choices_list = [[x, y] for x in choices['kernel_size']
                    for y in choices['exp_ratio']]

    num_features = 1280

    # act_layer = HardSwish
    act_layer = Swish

    new_arch = []
    # change to child arch_def
    for i, (layer_choice, layer_arch) in enumerate(zip(arch_list, arch_def)):
        if len(layer_arch) == 1:
            new_arch.append(layer_arch)
            continue
        else:
            new_layer = []
            for j, (block_choice, block_arch) in enumerate(
                    zip(layer_choice, layer_arch)):
                kernel_size, exp_ratio = choices_list[block_choice]
                elements = block_arch.split('_')
                block_arch = block_arch.replace(
                    elements[2], 'k{}'.format(str(kernel_size)))
                block_arch = block_arch.replace(
                    elements[4], 'e{}'.format(str(exp_ratio)))
                new_layer.append(block_arch)
            new_arch.append(new_layer)

    model_kwargs = dict(
        block_args=decode_arch_def(new_arch),
        num_features=num_features,
        stem_size=16,
        norm_kwargs=resolve_bn_args(kwargs),
        act_layer=act_layer,
        se_kwargs=dict(
            act_layer=nn.ReLU,
            gate_fn=hard_sigmoid,
            reduce_mid=True,
            divisor=8),
        **kwargs,
    )
    model = ChildNet(**model_kwargs)
    return model


class ShuffleNetV2(nn.Module):
    def __init__(self, stages_repeats, stages_out_channels, num_classes=1000, inverted_residual=InvertedResidual):
        super(ShuffleNetV2, self).__init__()

        if len(stages_repeats) != 3:
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels

        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(
                stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [inverted_residual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(inverted_residual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Linear(output_channels, num_classes)
        self.extras = nn.ModuleList([
            InvertedResidual(1024, 512, 2),
            InvertedResidual(512, 256, 2),
            InvertedResidual(256, 256, 2),
            InvertedResidual(256, 64, 2)
        ])

        self.reset_parameters()


    def reset_parameters(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    '''
    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = x.mean([2, 3])  # globalpool
        x = self.fc(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)
    '''
    #重写forward方法，适应SSD-master的要求
    def forward(self, x):
        features = []

        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        features.append(x)

        x = self.stage4(x)
        x = self.conv5(x)
        features.append(x)

        for i in range(len(self.extras)):
            x = self.extras[i](x)
            features.append(x)

        return tuple(features)


def _shufflenetv2(arch, pretrained, *args, **kwargs):
    model = ShuffleNetV2(*args, **kwargs)

    if pretrained:
        model_url = model_urls[arch]
        if model_url is None:
            raise NotImplementedError('pretrained {} is not supported as of now'.format(arch))
        else:
            state_dict = load_state_dict_from_url(model_url)
            model.load_state_dict(state_dict, strict=False)

    return model


def shufflenet_v2_x0_5(pretrained=False, progress=True, **kwargs):
    """
    Constructs a ShuffleNetV2 with 0.5x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _shufflenetv2('shufflenetv2_x0.5', pretrained, progress,
                         [4, 8, 4], [24, 48, 96, 192, 1024], **kwargs)


def shufflenet_v2_x1_0(pretrained=True, **kwargs):
    """
    Constructs a ShuffleNetV2 with 1.0x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _shufflenetv2('shufflenetv2_x1.0', pretrained,
                         [4, 8, 4], [24, 116, 232, 464, 1024], **kwargs)


def shufflenet_v2_x1_5(pretrained=False, progress=True, **kwargs):
    """
    Constructs a ShuffleNetV2 with 1.5x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _shufflenetv2('shufflenetv2_x1.5', pretrained, progress,
                         [4, 8, 4], [24, 176, 352, 704, 1024], **kwargs)


def shufflenet_v2_x2_0(pretrained=False, progress=True, **kwargs):
    """
    Constructs a ShuffleNetV2 with 2.0x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _shufflenetv2('shufflenetv2_x2.0', pretrained, progress,
                         [4, 8, 4], [24, 244, 488, 976, 2048], **kwargs)


@registry.BACKBONES.register('shufflenet_v2')
def shufflenet_v2(cfg, pretrained=True):
    model = gen_childnet(
        arch_list,
        arch_def,
        num_classes=21,
        drop_rate=0.0,
        global_pool='avg')
    #_, __ = resume_checkpoint(model, '114.pth.tar')
    state_dict = torch.load("114.pth.tar")["state_dict"]
    model.load_state_dict(state_dict, strict=False)
    return model
