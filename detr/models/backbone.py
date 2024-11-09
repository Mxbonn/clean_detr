# adapted from https://github.com/facebookresearch/detr/blob/main/models/backbone.py
import functools
from typing import Literal

import torchvision
from jaxtyping import Float
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.resnet import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
)
from torchvision.ops.misc import FrozenBatchNorm2d

from .transformer import TransformerEncoder


class BackboneBase(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        train_backbone: bool,
        num_channels: int,
    ):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or "layer2" not in name and "layer3" not in name and "layer4" not in name:
                parameter.requires_grad_(False)
        return_layers = {"layer4": "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, x: Float[Tensor, " b c h w"]) -> Float[Tensor, " b c_out h_out w_out"]:
        out = self.body(x)["0"]
        return out


RESNET_WEIGHTS = {
    "resnet18": ResNet18_Weights.DEFAULT,
    "resnet34": ResNet34_Weights.DEFAULT,
    "resnet50": ResNet50_Weights.DEFAULT,
    "resnet101": ResNet101_Weights.DEFAULT,
    "resnet152": ResNet152_Weights.DEFAULT,
}


class ResNetBackbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(
        self,
        model: Literal["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
        train_backbone: bool,
        dilation: bool,
    ):
        norm_layer = functools.partial(FrozenBatchNorm2d, eps=1e-5)
        backbone = getattr(torchvision.models, model)(
            weights=RESNET_WEIGHTS[model],
            replace_stride_with_dilation=[False, False, dilation],
            norm_layer=norm_layer,
        )
        num_channels = 512 if model in ("resnet18", "resnet34") else 2048
        super().__init__(backbone, train_backbone, num_channels)


class Backbone(nn.Module):
    def __init__(self, backbone: BackboneBase, position_embedding: nn.Module, transformer_encoder: TransformerEncoder):
        super().__init__()
        self.backbone = backbone
        self.position_embedding = position_embedding
        self.transformer_encoder = transformer_encoder
        self.input_proj = nn.Conv2d(backbone.num_channels, transformer_encoder.embedding_dim, kernel_size=1)

    def forward(self, x: Float[Tensor, " b c h w"]) -> tuple[Float[Tensor, " b hw d"], Float[Tensor, " b hw d"]]:
        x = self.backbone(x)
        x = self.input_proj(x)
        pos = self.position_embedding(x).to(x.dtype)  # TODO move position embedding to detr module?

        # flatten N x C x H x W to N x HW x C
        x_tokens = x.flatten(2).permute(0, 2, 1)
        pos_tokens = pos.flatten(2).permute(0, 2, 1)
        img_tokens = self.transformer_encoder(x_tokens, pos_tokens)
        return img_tokens, pos_tokens
