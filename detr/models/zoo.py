from typing import Literal

import torch

from .backbone import Backbone, ResNetBackbone
from .convert_state_dict import convert_original_state_dict
from .detr import DETR
from .position_embedding import PositionEmbeddingSine
from .transformer import TransformerDecoder, TransformerEncoder


def _make_detr(
    backbone_name: Literal["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"], dilation=False, num_classes=91
):
    embedding_dim = 256
    position_embedding = PositionEmbeddingSine(embedding_dim)
    resnet_backbone = ResNetBackbone(backbone_name, train_backbone=True, dilation=dilation)
    transformer_encoder = TransformerEncoder(num_layers=6, embedding_dim=embedding_dim, num_heads=8, mlp_dim=2048)
    backbone = Backbone(resnet_backbone, position_embedding, transformer_encoder)
    norm = torch.nn.LayerNorm(embedding_dim)
    decoder = TransformerDecoder(num_layers=6, norm=norm, embedding_dim=embedding_dim, num_heads=8, mlp_dim=2048)
    detr = DETR(backbone, decoder, num_classes=num_classes, num_queries=100)
    return detr


def detr_resnet50(pretrained=False, num_classes=91):
    """
    DETR R50 with 6 encoder and 6 decoder layers.

    Achieves 42/62.4 AP/AP50 on COCO val5k.
    """
    model = _make_detr("resnet50", dilation=False, num_classes=num_classes)
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth", map_location="cpu", check_hash=True
        )
        model_state_dict = checkpoint["model"]
        model_state_dict = convert_original_state_dict(model_state_dict)
        model.load_state_dict(model_state_dict)

    return model


def detr_resnet50_dc5(pretrained=False, num_classes=91):
    """
    DETR-DC5 R50 with 6 encoder and 6 decoder layers.

    The last block of ResNet-50 has dilation to increase
    output resolution.
    Achieves 43.3/63.1 AP/AP50 on COCO val5k.
    """
    model = _make_detr("resnet50", dilation=True, num_classes=num_classes)
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/detr/detr-r50-dc5-f0fb7ef5.pth", map_location="cpu", check_hash=True
        )
        model_state_dict = checkpoint["model"]
        model_state_dict = convert_original_state_dict(model_state_dict)
        model.load_state_dict(model_state_dict)
    return model


def detr_resnet101(pretrained=False, num_classes=91):
    """
    DETR-DC5 R101 with 6 encoder and 6 decoder layers.

    Achieves 43.5/63.8 AP/AP50 on COCO val5k.
    """
    model = _make_detr("resnet101", dilation=False, num_classes=num_classes)
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/detr/detr-r101-2c7b67e5.pth", map_location="cpu", check_hash=True
        )
        model_state_dict = checkpoint["model"]
        model_state_dict = convert_original_state_dict(model_state_dict)
        model.load_state_dict(model_state_dict)
    return model


def detr_resnet101_dc5(pretrained=False, num_classes=91):
    """
    DETR-DC5 R101 with 6 encoder and 6 decoder layers.

    The last block of ResNet-101 has dilation to increase
    output resolution.
    Achieves 44.9/64.7 AP/AP50 on COCO val5k.
    """
    model = _make_detr("resnet101", dilation=True, num_classes=num_classes)
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/detr/detr-r101-dc5-a2e86def.pth", map_location="cpu", check_hash=True
        )
        model_state_dict = checkpoint["model"]
        model_state_dict = convert_original_state_dict(model_state_dict)
        model.load_state_dict(model_state_dict)
    return model
