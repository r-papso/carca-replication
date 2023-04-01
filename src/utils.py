from typing import Tuple

import torch


def get_mask(input: torch.Tensor) -> torch.Tensor:
    return torch.where(input == 0.0, 0.0, 1.0)


def to(*tensors: torch.Tensor, device: str) -> Tuple[torch.Tensor, ...]:
    return tuple([t.to(device) for t in tensors])
