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
    def forward(self, x: Tensor, a: Tensor, c: Tensor, mask: Tensor, target: bool) -> Tensor:
        pass


class Encoding(nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        pass


class Encoder(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        pass


class Decoder(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, o: Tensor, o_mask: Tensor, p: Tensor, p_mask: Tensor) -> Tensor:
        pass
