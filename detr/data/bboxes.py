from jaxtyping import Float, Int
from tensordict import tensorclass
from torch import Tensor
from torchvision.ops.boxes import box_convert


@tensorclass
class BoundingBoxes:
    bboxes: Float[  #
        Tensor, " *b n 4"  # assuming the format is XYXY
    ]  # Not use torchvision.tv_tensors.BoundingBoxes as it cannot have batch dimension and does not generalize to e.g. keypoints
    canvas_size: Int[Tensor, " *b 2"]  # (height, width) of the corresponding image, handy for resizing

    @property
    def bboxes_cxcywh(self) -> Float[Tensor, " *b n 4"]:
        return box_convert(self.bboxes, "xyxy", "cxcywh")


@tensorclass
class GTBoundingBoxes(BoundingBoxes):
    class_labels: Int[Tensor, " *b n"]


@tensorclass
class PredictedBoundingBoxes(BoundingBoxes):
    class_outputs: Float[Tensor, " *b n c"]

    @property
    def class_probs(self) -> Float[Tensor, " *b n c"]:
        raise NotImplementedError


@tensorclass
class DetrOutputs(PredictedBoundingBoxes):
    @property
    def class_probs(self) -> Float[Tensor, " *b n c"]:
        return self.class_outputs.softmax(dim=-1)
