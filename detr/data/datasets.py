from typing import Literal

import torch
import torchvision
from torchvision.ops.boxes import box_convert
from torchvision.transforms.v2 import Compose, Normalize, RandomChoice
from torchvision.transforms.v2.functional import pil_to_tensor, to_dtype

from .bboxes import GTBoundingBoxes
from .transforms import RandomHorizontalFlip, RandomResizedCrop, Resize, SanitizeBoundingBoxes


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, root: str, annFile: str, transforms=None) -> None:
        super().__init__(root, annFile)
        self._transforms = transforms

    def __getitem__(self, index: int):
        img, target = super().__getitem__(index)
        img = to_dtype(pil_to_tensor(img), scale=True)
        labels = torch.tensor([ann["category_id"] for ann in target], dtype=torch.int)
        bboxes = torch.tensor([ann["bbox"] for ann in target], dtype=torch.float32).reshape(-1, 4)
        bboxes = box_convert(bboxes, "xywh", "xyxy")
        size = torch.tensor(img.shape[-2:])

        bboxes = GTBoundingBoxes(bboxes=bboxes, canvas_size=size, class_labels=labels)  # pyright: ignore
        if self._transforms is not None:
            img, bboxes = self._transforms(img, bboxes)
        return img, bboxes


def make_coco_transforms(stage: Literal["train", "val"]):
    final_transforms = [SanitizeBoundingBoxes(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]

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
