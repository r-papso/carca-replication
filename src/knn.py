from typing import List, Tuple

from torch import Tensor
import torch
from .abstract import Model


class KNN(Model):
    def __init__(self):
        super().__init__()

    def forward(self, profile: Tuple[Tensor, Tensor, Tensor], targets: List[Tuple[Tensor, Tensor, Tensor]]) -> Tensor:
        p_x, p_a, p_c = profile
        last_p = p_a[:, -1:, :]
        y_preds = []

        for o_x, o_a, o_c in targets:
            y_pred = torch.sum(last_p * o_a, dim=-1)  # Dot-product between last profile item and target items
            y_preds.append(y_pred)

        return torch.cat(y_preds, dim=-1)
