from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from ..utils import get_mask
from .embedding import Embedding
from .encoder import SelfAttentionBlock
from .decoder import CrossAttentionBlock


class CARCA(nn.Module):
    def __init__(self, d: int, H: int, p: float, B: int, res_sa: bool, res_ca: bool, emb: Embedding):
        super().__init__()

        self.embeds = emb
        self.dropout = nn.Dropout(p=p)
        self.sa_blocks = nn.ModuleList([SelfAttentionBlock(d, H, p, res_sa) for _ in range(B)])
        self.norm = nn.LayerNorm(normalized_shape=d)
        self.ca_blocks = CrossAttentionBlock(d, H, p, res_ca)

    def forward(self, profile: Tuple[Tensor, Tensor, Tensor], targets: List[Tuple[Tensor, Tensor, Tensor]]) -> Tensor:
        p_x, p_a, p_c = profile
        p_mask = get_mask(p_x)

        p_e = self.embeds.forward(p_x, p_a, p_c)
        p_e = self.dropout.forward(p_e)

        for block in self.sa_blocks:
            p_e = block.forward(p_e, p_mask)

        p_e = self.norm.forward(p_e)
        y_preds = []

        for o_x, o_a, o_c in targets:
            o_mask = get_mask(o_x)
            o_e = self.embeds.forward(o_x, o_a, o_c)

            y_pred = self.ca_blocks.forward(o_e, o_mask, p_e, p_mask)
            y_preds.append(y_pred)

        return torch.cat(y_preds, dim=-1)


class BinaryCrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred: Tensor, y_true: Tensor, mask: Tensor, eps: float = 1e-8) -> Tensor:
        loss = -(y_true * torch.log(y_pred + eps) + (1.0 - y_true) * torch.log(1.0 - y_pred + eps))
        loss = torch.sum(loss * mask) / torch.sum(mask)
        return loss
