import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Shaped
from torch import Tensor
from torchvision.ops.boxes import box_convert, generalized_box_iou

from detr.data.bboxes import DetrOutputs, GTBoundingBoxes
from detr.data.transforms import resize_bboxes

from .matcher import HungarianMatcher, get_src_indices, get_target_indices


class DetrLoss(nn.Module):
    def __init__(
        self,
        matcher: HungarianMatcher,
        num_classes: int,
        weight_class: float,
        weight_bbox: float,
        weight_giou: float,
        no_object_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.matcher = matcher
        self.num_classes = num_classes
        self.weight_class = weight_class
        self.weight_bbox = weight_bbox
        self.weight_giou = weight_giou
        self.no_object_weight = no_object_weight

    def forward(
        self, outputs: Shaped[DetrOutputs, " *b num_queries"], targets: list[Shaped[GTBoundingBoxes, ""]]
    ) -> tuple[Float[Tensor, ""], dict]:
        outputs = resize_bboxes(outputs, (1, 1))
        targets = [resize_bboxes(target, (1, 1)) for target in targets]

        matches = self.matcher(outputs, targets)

        # Compute the average number of target centers, for normalization purposes
        num_bboxes = sum(len(target.class_labels) for target in targets)  # type: ignore
        num_bboxes = torch.as_tensor([num_bboxes], dtype=torch.float, device=targets[0].device)  # pyright: ignore[reportAttributeAccessIssue]

        src_query_indices = get_src_indices(matches)
        target_query_indices = get_target_indices(matches)

        # Compute the L1 loss between centers
        matched_output_bboxes = outputs.bboxes[src_query_indices]
        matched_target_bboxes = torch.stack(
            [
                targets[batch_idx.item()].bboxes[target_idx.item()]  # type: ignore
                for batch_idx, target_idx in zip(*target_query_indices)
            ]
        )

        loss_bboxes = F.l1_loss(
            box_convert(matched_output_bboxes, "xyxy", "cxcywh"),
            box_convert(matched_target_bboxes, "xyxy", "cxcywh"),
            reduction="none",
        )
        loss_bboxes = loss_bboxes.sum() / num_bboxes

        loss_giou = 1 - torch.diag(generalized_box_iou(matched_output_bboxes, matched_target_bboxes))
        loss_giou = loss_giou.sum() / num_bboxes

        # Compute the classification loss
        src_logits = outputs.class_logits
        matched_target_labels = torch.stack(
            [
                targets[batch_idx.item()].class_labels[target_idx.item()]  # type: ignore
                for batch_idx, target_idx in zip(*target_query_indices)
            ]
        )
        target_labels = torch.full(src_logits.shape[:2], self.num_classes, device=src_logits.device, dtype=torch.int64)
        target_labels[src_query_indices] = matched_target_labels.to(torch.int64)
        weights = torch.ones(self.num_classes + 1, device=src_logits.device, dtype=torch.float)
        weights[-1] = self.no_object_weight
        classification_loss = F.cross_entropy(src_logits.transpose(1, 2), target_labels, weight=weights)
        loss = self.weight_bbox * loss_bboxes + self.weight_class * classification_loss + self.weight_giou * loss_giou
        loss_dict = {
            "bboxes_l1": loss_bboxes.item(),
            "bbox_giou": loss_giou.item(),
            "classification_ce": classification_loss.item(),
        }
        return loss, loss_dict
