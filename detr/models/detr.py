from jaxtyping import Float, Shaped
from torch import Tensor, nn
from torchvision.ops.boxes import box_convert

from detr.data.bboxes import DetrOutputs

from .misc import MLP
from .transformer import TransformerDecoder


class DETR(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        transformer_decoder: TransformerDecoder,
        num_classes: int,
        num_queries: int,
        aux_loss: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.transformer_decoder = transformer_decoder
        self.num_queries = num_queries
        self.aux_loss = aux_loss
        embedding_dim = transformer_decoder.embedding_dim
        self.class_embed = MLP(embedding_dim, embedding_dim, num_classes + 1, 1)
        self.bbox_embed = MLP(embedding_dim, embedding_dim, 4, 3, sigmoid_output=True)
        self.query_embed = nn.Embedding(num_queries, embedding_dim)

    def forward(self, input: Float[Tensor, " b 3 h w"]) -> Shaped[DetrOutputs, " b num_queries"]:
        b, _, h, w = input.shape
        img_tokens, pos_tokens = self.backbone(input)

        query = self.query_embed.weight[None, ...].expand(b, -1, -1)

        processed_queries = self.transformer_decoder(img_tokens, pos_tokens, query)

        processed_queries = processed_queries[-1]
        outputs_class = self.class_embed(processed_queries)
        outputs_bbox = self.bbox_embed(processed_queries)  # cxcywh format and normalized by image size
        outputs_bbox = box_convert(outputs_bbox, "cxcywh", "xyxy")

        predicted_bboxes = DetrOutputs(
            bboxes=outputs_bbox,  # pyright: ignore[reportCallIssue]
            class_outputs=outputs_class,  # pyright: ignore[reportCallIssue]
            batch_size=outputs_bbox.shape[:-1],  # pyright: ignore[reportCallIssue]
        )
        return predicted_bboxes
