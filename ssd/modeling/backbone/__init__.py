from ssd.modeling import registry
from .vgg import VGG
from .mobilenet import MobileNetV2
from .efficient_net import EfficientNet
from .shufflenetv2 import ShuffleNetV2
from .childnet import ChildNet

__all__ = ['build_backbone', 'VGG', 'MobileNetV2', 'EfficientNet', 'ShuffleNetV2', 'ChildNet']


def build_backbone(cfg):
    return registry.BACKBONES[cfg.MODEL.BACKBONE.NAME](cfg, cfg.MODEL.BACKBONE.PRETRAINED)
