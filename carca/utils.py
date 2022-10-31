from typing import Tuple
import torch


def to(*tensors: torch.Tensor, device: str) -> Tuple[torch.Tensor, ...]:
    return tuple([t.to(device) for t in tensors])
