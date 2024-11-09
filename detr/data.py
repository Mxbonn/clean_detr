from jaxtyping import Float
from tensordict import tensorclass
from torch import Tensor


@tensorclass
class PredictedBoundingBox:
    bboxes: Float[  #
        Tensor, " *b n 4"  # assuming the format is XYXY
    ]  # Not use torchvision.tv_tensors.BoundingBoxes as it cannot have batch dimension and does not generalize to e.g. keypoints
    class_outputs: Float[Tensor, " *b n c"]

    @property
    def class_probs(self) -> Float[Tensor, " *b n c"]:
        raise NotImplementedError


@tensorclass
class DetrOutputs(PredictedBoundingBox):
    @property
    def class_probs(self) -> Float[Tensor, " *b n c"]:
        return self.class_outputs.softmax(dim=-1)
