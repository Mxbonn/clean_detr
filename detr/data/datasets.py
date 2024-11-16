from typing import Literal

import torch
import torchvision
from jaxtyping import Float, Shaped
from torch import Tensor
from torchvision.ops.boxes import box_convert
from torchvision.transforms.v2 import Compose, Normalize, RandomChoice
from torchvision.transforms.v2.functional import pil_to_tensor, to_dtype

from .bboxes import GTBoundingBoxes
from .transforms import (
    NormalizeBoundingBoxes,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    SanitizeBoundingBoxes,
    resize_bboxes,
)


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, root: str, ann_file: str, transforms=None) -> None:
        super().__init__(root, ann_file)
        self._transforms = transforms

    def __getitem__(self, index: int):
        img, target = super().__getitem__(index)
        img = to_dtype(pil_to_tensor(img), scale=True)
        labels = torch.tensor([ann["category_id"] for ann in target], dtype=torch.int)
        bboxes = torch.tensor([ann["bbox"] for ann in target], dtype=torch.float32).reshape(-1, 4)
        bboxes = box_convert(bboxes, "xywh", "xyxy")
        bboxes = GTBoundingBoxes(bboxes=bboxes, class_labels=labels, batch_size=[len(bboxes)])  # pyright: ignore
        if self._transforms is not None:
            img, bboxes = self._transforms(img, bboxes)
        return img, bboxes


def collate_fn(
    batch: list[tuple[Float[Tensor, " 3 h w"], Shaped[GTBoundingBoxes, " n"]]],
) -> tuple[Float[Tensor, " b 3 h w"], list[Shaped[GTBoundingBoxes, " n"]]]:
    images, bboxes = zip(*batch)
    # Get max dimensions
    max_height = max(img.shape[-2] for img in images)
    max_width = max(img.shape[-1] for img in images)

    padded_images = []
    new_bboxes = []
    for img, _bboxes in zip(images, bboxes):
        h, w = img.shape[-2:]
        padded_img = torch.nn.functional.pad(img, (0, max_width - w, 0, max_height - h))
        _new_bboxes = resize_bboxes(_bboxes, (max_height, max_width), (h, w))
        padded_images.append(padded_img)
        new_bboxes.append(_new_bboxes)

    # Stack into batches
    image_batch = torch.stack(padded_images, dim=0)

    return image_batch, new_bboxes


def make_coco_transforms(stage: Literal["train", "val"]):
    final_transforms = [
        SanitizeBoundingBoxes(),
        NormalizeBoundingBoxes(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]

    if stage == "train":
        scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
        transforms = Compose(
            [
                RandomHorizontalFlip(),
                RandomChoice(
                    [
                        RandomChoice([Resize(size, max_size=1333) for size in scales]),
                        RandomChoice([RandomResizedCrop(size, scale=(0.6, 1)) for size in scales]),
                    ]
                ),
                *final_transforms,
            ]
        )
    elif stage == "val":
        transforms = Compose([Resize(800, max_size=1333), *final_transforms])
    else:
        raise ValueError(f"Unknown stage: {stage}")
    return transforms
