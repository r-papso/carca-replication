from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import Tensor


class Embedding(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: Tensor, a: Tensor, c: Tensor) -> Tensor:
        pass


class AllEmbedding(Embedding):
    def __init__(self, n_items: int, d: int, g: int, n_ctx: int, n_attrs: int):
        super().__init__()

        self.d = d

        self.items_embed = nn.Embedding(num_embeddings=n_items, embedding_dim=d, padding_idx=0)
        self.feats_embed = nn.Linear(in_features=n_ctx + n_attrs, out_features=g)
        self.joint_embed = nn.Linear(in_features=g + d, out_features=d)

        nn.init.xavier_uniform_(self.items_embed.weight)
        # nn.init.normal_(self.items_embed.weight, std=0.01)
        nn.init.normal_(self.feats_embed.weight, std=0.01)
        nn.init.normal_(self.joint_embed.weight, std=0.01)

        self.items_embed._fill_padding_idx_with_zero()
        nn.init.zeros_(self.feats_embed.bias)
        nn.init.zeros_(self.joint_embed.bias)

    def forward(self, x: Tensor, a: Tensor, c: Tensor) -> Tensor:
        q = self.feats_embed.forward(torch.cat((a, c), dim=-1))
        z = self.items_embed.forward(x)
        z = z * (self.d**0.5)  # Scale embedding output

        e = self.joint_embed.forward(torch.cat((z, q), dim=-1))
        return e


class AttrCtxEmbedding(Embedding):
    def __init__(self, d: int, g: int, n_ctx: int, n_attrs: int):
        super().__init__()

        self.d = d

        self.feats_embed = nn.Linear(in_features=n_ctx + n_attrs, out_features=g)
        self.joint_embed = nn.Linear(in_features=g + d, out_features=d)

        nn.init.normal_(self.feats_embed.weight, std=0.01)
        nn.init.normal_(self.joint_embed.weight, std=0.01)

        nn.init.zeros_(self.feats_embed.bias)
        nn.init.zeros_(self.joint_embed.bias)

    def forward(self, x: Tensor, a: Tensor, c: Tensor) -> Tensor:
        q = self.feats_embed.forward(torch.cat((a, c), dim=-1))
        e = self.joint_embed.forward(q, dim=-1)
        return e


class AttrEmbedding(Embedding):
    def __init__(self, d: int, g: int, n_attrs: int):
        super().__init__()

        self.d = d

        self.feats_embed = nn.Linear(in_features=n_attrs, out_features=g)
        self.joint_embed = nn.Linear(in_features=g + d, out_features=d)

        nn.init.normal_(self.feats_embed.weight, std=0.01)
        nn.init.normal_(self.joint_embed.weight, std=0.01)

        nn.init.zeros_(self.feats_embed.bias)
        nn.init.zeros_(self.joint_embed.bias)

    def forward(self, x: Tensor, a: Tensor, c: Tensor) -> Tensor:
        q = self.feats_embed.forward(a)
        e = self.joint_embed.forward(q)
        return e
