import torch
from jaxtyping import Float
from torch import Tensor, nn


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: type[nn.Module] = nn.ReLU,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(embedding_dim, num_heads, mlp_dim, activation, dropout) for _ in range(num_layers)]
        )

        # reset parameters of transformer
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self, image_embeddings: Float[Tensor, " b hw c"], pos_embeddings: Float[Tensor, " b hw c"]
    ) -> Float[Tensor, " b hw c"]:
        for layer in self.layers:
            image_embeddings = layer(image_embeddings, pos_embeddings)
        return image_embeddings


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: type[nn.Module] = nn.ReLU,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(embedding_dim, mlp_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(mlp_dim, embedding_dim)

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation()

    def forward(
        self, image_embeddings: Float[Tensor, " b hw c"], pos_embeddings: Float[Tensor, " b hw c"]
    ) -> Float[Tensor, " b hw c"]:
        q = k = image_embeddings + pos_embeddings
        v = image_embeddings
        attn_out = self.self_attn(q, k, v)[0]
        out = v + self.dropout1(attn_out)
        out = self.norm1(out)
        mlp_out = self.linear2(self.dropout(self.activation(self.linear1(out))))
        out = out + self.dropout2(mlp_out)
        out = self.norm2(out)
        return out


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        norm: nn.Module,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: type[nn.Module] = nn.ReLU,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.layers = nn.ModuleList(
            [TransformerDecoderLayer(embedding_dim, num_heads, mlp_dim, activation, dropout) for _ in range(num_layers)]
        )
        self.norm = norm

        # reset parameters of transformer
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        image_embeddings: Float[Tensor, " b hw c"],
        pos_embeddings: Float[Tensor, " b hw c"],
        queries: Float[Tensor, " b n c"],
    ) -> Float[Tensor, " l b n c"]:
        queries_pe = queries
        queries = torch.zeros_like(queries_pe)
        intermediate_tensors = []
        for layer in self.layers:
            queries = layer(image_embeddings, pos_embeddings, queries, queries_pe)
            intermediate_tensors.append(self.norm(queries))
        out = torch.stack(intermediate_tensors)
        return out


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: type[nn.Module] = nn.ReLU,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(embedding_dim, mlp_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(mlp_dim, embedding_dim)

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.norm3 = nn.LayerNorm(embedding_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = activation()

    def forward(
        self,
        image_embeddings: Float[Tensor, " b hw c"],
        pos_embeddings: Float[Tensor, " b hw c"],
        queries: Float[Tensor, " b n c"],
        queries_pe: Float[Tensor, " b n c"],
    ) -> Float[Tensor, " b n c"]:
        q = k = queries + queries_pe
        v = queries
        attn_out = self.self_attn(q, k, v)[0]
        queries = queries + self.dropout1(attn_out)
        queries = self.norm1(queries)
        q = queries + queries_pe
        k = image_embeddings + pos_embeddings
        v = image_embeddings
        attn_out = self.multihead_attn(q, k, v)[0]
        queries = queries + self.dropout2(attn_out)
        queries = self.norm2(queries)
        mlp_out = self.linear2(self.dropout(self.activation(self.linear1(queries))))
        queries = queries + self.dropout3(mlp_out)
        queries = self.norm3(queries)
        return queries
