from .builder_util import *
from .build_childnet import *
import torch.nn as nn
from ssd.modeling import registry
from timm.models.layers import SelectAdaptivePool2d
from timm.models.layers.activations import hard_sigmoid
#from timm.models import resume_checkpoint


# 参数定义部分 其中arch_list受到加载模型的影响
# 114.pth.tar对应的参数
arch_list114 = [[0], [3], [3, 3], [3, 3], [3, 3, 3], [3, 3], [0]]
image_size114 = 160

# 604.pth.tar对应的参数
arch_list604 = [[0], [3, 3, 2, 3, 3], [3, 2, 3, 2, 3], [3, 2, 3, 2, 3],[3, 3, 2, 2, 3, 3], [3, 3, 2, 3, 3, 3], [0]]
image_size604 = 224

arch_list = arch_list604
image_size = image_size604

stem = ['ds_r1_k3_s1_e1_c16_se0.25', 'cn_r1_k1_s1_c320_se0.25']
choice_block_pool = ['ir_r1_k3_s2_e4_c24_se0.25',
                         'ir_r1_k5_s2_e4_c40_se0.25',
                         'ir_r1_k3_s2_e6_c80_se0.25',
                         'ir_r1_k3_s1_e6_c96_se0.25',
                         'ir_r1_k5_s2_e6_c192_se0.25']
arch_def = [[stem[0]]] + [[choice_block_pool[idx]
                               for repeat_times in range(len(arch_list[idx + 1]))]
                              for idx in range(len(choice_block_pool))] + [[stem[1]]]


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class ChildNet(nn.Module):

    def __init__(
            self,
            block_args,
            num_classes=1000,
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

        # 额外层
        self.extras = nn.ModuleList([InvertedResidual(320, 640, 2, 0.2)])
        self.reset_parameters()


        efficientnet_init_weights(self)

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


    def forward(self, x):
        features = []
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        #features.append(x) # 16*160*160

        x = self.blocks[0](x) #16*160*160
        x = self.blocks[1](x) #24*80*80
        x = self.blocks[2](x) #40*40*40
        x = self.blocks[3](x) #80*20*20
        x = self.blocks[4](x) #96*20*20
        features.append(x)
        x = self.blocks[5](x) #192*10*10
        x = self.blocks[6](x) #320*10*10
        features.append(x)

        for i in self.extras:
            x = i(x) # 640*5*5
            features.append(x)

        '''
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        features.append(x) # 1280*1*1
        '''

        # for i in features:
        #     print(i.shape)
        # raise AssertionError

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


@registry.BACKBONES.register('childnet')
def childnet(cfg,pretrained = False):
    model = gen_childnet(
        arch_list,
        arch_def,
        num_classes=1000,
        drop_rate=0.0,
        global_pool='avg')
    #_, __ = resume_checkpoint(model, '604.pth.tar')
    state_dict = torch.load("604.pth.tar")["state_dict"]
    model.load_state_dict(state_dict, strict=False)
    return model