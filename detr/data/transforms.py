from collections.abc import Sequence
from typing import TypeVar

import torch
from jaxtyping import Float, Shaped
from torch import Tensor, nn
from torchvision.transforms.v2 import InterpolationMode
from torchvision.transforms.v2 import RandomResizedCrop as _RandomResizedCrop
from torchvision.transforms.v2.functional import resize, resized_crop

from detr.data.bboxes import BoundingBoxes

B = TypeVar("B", bound=BoundingBoxes)


def resize_bboxes(
    bboxes: Shaped[B, " *b"], current_size: tuple[int, int], target_size: tuple[int, int]
) -> Shaped[B, " *b"]:
    current_size = torch.tensor(current_size, device=bboxes.device)  # pyright: ignore
    new_size = torch.tensor(target_size, device=bboxes.device)  # pyright: ignore
    new_bboxes = bboxes.copy()  # pyright: ignore
    scale_factors = new_size / current_size  # shape: (n, 2)
    # Convert from [h, w] to [w, h, w, h] for each bbox
    scale_factors = torch.cat([scale_factors.flip(-1), scale_factors.flip(-1)], dim=-1)
    new_bboxes.bboxes = bboxes.bboxes * scale_factors
    return new_bboxes


def crop_bboxes(bboxes: B, top: int, left: int, height: int, width: int) -> B:
    bboxes = bboxes.copy()  # pyright: ignore
    bboxes.bboxes = bboxes.bboxes - torch.as_tensor(
        [left, top, left, top], device=bboxes.bboxes.device, dtype=bboxes.bboxes.dtype
    )
    bboxes.bboxes[..., 0::2].clamp_(min=0, max=width)
    bboxes.bboxes[..., 1::2].clamp_(min=0, max=height)
    return bboxes


def sanitize_bboxes(bboxes: B, min_size: int = 1, min_area: int = 1) -> B:
    ws, hs = bboxes.bboxes[..., 2] - bboxes.bboxes[..., 0], bboxes.bboxes[..., 3] - bboxes.bboxes[..., 1]
    valid = (ws >= min_size) & (hs >= min_size) & (bboxes.bboxes >= 0).all(dim=-1) & (ws * hs >= min_area)
    # apply the mask to the bboxes but also to other fields such as labels
    # cant apply to everything because the common dimension is only on image level
    # e.g. canvas_size is not repeated per bounding box but per image

    sanitized_bboxes = bboxes[valid]  # pyright: ignore
    return sanitized_bboxes


class ImageWithBoundingBoxesTransform(nn.Module):
    def forward(
        self, image: Float[Tensor, " 3 h w"], targets: Shaped[B, ""]
    ) -> tuple[Float[Tensor, " 3 h w"], Shaped[B, ""]]:
        raise NotImplementedError()


class RandomHorizontalFlip(ImageWithBoundingBoxesTransform):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(
        self, image: Float[Tensor, " 3 h w"], targets: Shaped[B, ""]
    ) -> tuple[Float[Tensor, " 3 h w"], Shaped[B, ""]]:
        if torch.rand(1) >= self.p:
            return image, targets
        w = image.shape[-1]
        image = image.flip(-1)
        flipped_centers: B = targets.copy()  # pyright: ignore[reportAttributeAccessIssue]
        flipped_centers.bboxes = flipped_centers.bboxes[:, [2, 1, 0, 3]] * torch.as_tensor(
            [-1, 1, -1, 1]
        ) + torch.as_tensor([w, 0, w, 0])
        return image, flipped_centers


class Resize(ImageWithBoundingBoxesTransform):
    def __init__(
        self,
        size: int | Sequence[int] | None = None,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        max_size: int | None = None,
        antialias: bool = True,
    ) -> None:
        super().__init__()

        if isinstance(size, int):
            size = [size]
        elif isinstance(size, Sequence) and len(size) in {1, 2}:
            size = list(size)
        elif size is None:
            if not isinstance(max_size, int):
                raise ValueError(f"max_size must be an integer when size is None, but got {max_size} instead.")
        else:
            raise ValueError(
                f"size can be an integer, a sequence of one or two integers, or None, but got {size} instead."
            )
        self.size = size

        self.interpolation = interpolation
        self.max_size = max_size
        self.antialias = antialias

    def forward(
        self, image: Float[Tensor, " 3 h w"], targets: Shaped[B, ""]
    ) -> tuple[Float[Tensor, " 3 h w"], Shaped[B, ""]]:
        current_size = (image.shape[-2], image.shape[-1])
        new_image = resize(image, self.size, self.interpolation, self.max_size, self.antialias)
        target_size = (new_image.shape[-2], new_image.shape[-1])
        new_targets = resize_bboxes(targets, current_size, target_size)
        return new_image, new_targets


class RandomResizedCrop(ImageWithBoundingBoxesTransform):
    def __init__(
        self,
        size: int | tuple[int, int],
        scale: tuple[float, float] = (0.08, 1.0),
        ratio: tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        antialias: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation
        self.antialias = antialias

    def forward(
        self, image: Float[Tensor, " 3 h w"], targets: Shaped[B, ""]
    ) -> tuple[Float[Tensor, " 3 h w"], Shaped[B, ""]]:
        current_size = (image.shape[-2], image.shape[-1])
        top, left, height, width = _RandomResizedCrop.get_params(image, scale=self.scale, ratio=self.ratio)  # pyright: ignore
        new_image = resized_crop(image, top, left, height, width, list(self.size), self.interpolation, self.antialias)
        new_targets = crop_bboxes(targets, top, left, height, width)
        new_targets = resize_bboxes(new_targets, current_size, self.size)
        return new_image, new_targets


class SanitizeBoundingBoxes(ImageWithBoundingBoxesTransform):
    def __init__(self, min_size: int = 1, min_area: int = 1):
        super().__init__()
        if min_size < 1:
            raise ValueError(f"min_size must be at least 1, but got {min_size} instead.")
        if min_area < 1:
            raise ValueError(f"min_area must be at least 1, but got {min_area} instead.")

        self.min_size = min_size
        self.min_area = min_area

    def forward(
        self, image: Float[Tensor, " 3 h w"], targets: Shaped[B, ""]
    ) -> tuple[Float[Tensor, " 3 h w"], Shaped[B, ""]]:
        return image, sanitize_bboxes(targets, self.min_size, self.min_area)


class NormalizeBoundingBoxes(ImageWithBoundingBoxesTransform):
    def forward(
        self, image: Float[Tensor, " 3 h w"], targets: Shaped[B, ""]
    ) -> tuple[Float[Tensor, " 3 h w"], Shaped[B, ""]]:
        current_size = (image.shape[-2], image.shape[-1])
        new_targets = resize_bboxes(targets, current_size, (1, 1))
        return image, new_targets
