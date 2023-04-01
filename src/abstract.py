from abc import ABC, abstractmethod
from typing import List, Tuple

import torch.nn as nn
from torch import Tensor


class Model(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, profile: Tuple[Tensor, Tensor, Tensor], targets: List[Tuple[Tensor, Tensor, Tensor]]) -> Tensor:
        pass


class Embedding(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: Tensor, a: Tensor, c: Tensor) -> Tensor:
        pass
