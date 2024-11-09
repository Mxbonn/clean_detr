from jaxtyping import Float, Shaped
from tensordict import tensorclass
from torch import Tensor, nn

from .misc import MLP
from .transformer import TransformerDecoder


@tensorclass
class PredictedBoundingBox:
    bbox: Tensor
    class_logits: Tensor


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

    def forward(self, input: Float[Tensor, " b 3 h w"]) -> Shaped[PredictedBoundingBox, " b num_queries"]:
        b = input.shape[0]
        img_tokens, pos_tokens = self.backbone(input)

        query = self.query_embed.weight[None, ...].expand(b, -1, -1)

        processed_queries = self.transformer_decoder(img_tokens, pos_tokens, query)

        processed_queries = processed_queries[-1]
        outputs_class = self.class_embed(processed_queries)
        outputs_coord = self.bbox_embed(processed_queries)
        predicted_bboxes = PredictedBoundingBox(
            bbox=outputs_coord,  # pyright: ignore
            class_logits=outputs_class,  # pyright: ignore
            batch_size=outputs_coord.shape[:-1],  # pyright: ignore
        )
        return predicted_bboxes
