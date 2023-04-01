from typing import List, Tuple

from torch import Tensor
from .abstract import Model


class KNN(Model):
    def __init__(self):
        super().__init__()

    def forward(self, profile: Tuple[Tensor, Tensor, Tensor], targets: List[Tuple[Tensor, Tensor, Tensor]]) -> Tensor:
        p_x, p_a, p_c = profile
        last = p_a[-1]
        y_preds = []

        for o_x, o_a, o_c in targets:
            pass
