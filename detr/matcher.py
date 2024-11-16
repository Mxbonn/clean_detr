# adapted from https://github.com/facebookresearch/detr/blob/main/models/matcher.py#L12
import torch
import torch.nn as nn
from jaxtyping import Int, Shaped
from scipy.optimize import linear_sum_assignment
from torch import Tensor
from torchvision.ops.boxes import generalized_box_iou

from detr.data.bboxes import DetrOutputs, GTBoundingBoxes


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs: Shaped[DetrOutputs, " *b num_queries"], targets: list[GTBoundingBoxes]):
        """Performs the matching
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs.batch_size  # pyright: ignore

        # flatten the outputs
        outputs = outputs.flatten(0, 1)  # pyright: ignore
        # targets cannot be stacked, because they may have different number of targets
        # so we cat them instead
        cat_targets: GTBoundingBoxes = torch.cat(targets)  # pyright: ignore
        # We flatten the outputs to compute the cost matrices in a batch
        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -outputs.class_probs.flatten(0, 1)[..., cat_targets.class_labels]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(outputs.bboxes_cxcywh, cat_targets.bboxes_cxcywh, p=1)

        # Compute the giou cost between boxes
        cost_giou = -generalized_box_iou(outputs.bboxes, cat_targets.bboxes)

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(_targets.bboxes) for _targets in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def get_src_indices(indices: list[tuple[Tensor, Tensor]]) -> tuple[Int[Tensor, " n"], Int[Tensor, " n"]]:
    # permute predictions following indices
    dim_0_indices = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    dim_1_indices = torch.cat([src for (src, _) in indices])
    return dim_0_indices, dim_1_indices


def get_target_indices(indices: list[tuple[Tensor, Tensor]]) -> tuple[Int[Tensor, " n"], Int[Tensor, " n"]]:
    # permute targets following indices
    dim_0_indices = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
    dim_1_indices = torch.cat([tgt for (_, tgt) in indices])
    return dim_0_indices, dim_1_indices
