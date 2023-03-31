import torch
import torch.nn as nn
from torch import Tensor

from .attention import MultiHeadAttention


# Target-level cross-attention block
class CrossAttentionBlock(nn.Module):
    def __init__(self, d: int, H: int, p: float, residual: bool):
        super().__init__()

        self.residual = residual

        # Attention
        self.attn = MultiHeadAttention(embed_dim=d, num_heads=H, dropout=p)

        # FFN
        self.ffn = nn.Linear(in_features=d, out_features=1)
        self.sig = nn.Sigmoid()

        nn.init.normal_(self.ffn.weight, std=0.01)
        nn.init.zeros_(self.ffn.bias)

    def forward(self, e: Tensor, e_mask: Tensor, f: Tensor, f_mask: Tensor) -> Tensor:
        s = self.attn.forward(e, f, f, q_mask=e_mask, k_mask=f_mask)

        if self.residual:
            s = torch.mul(s, e)  # Multiplicative residual connection

        y = self.ffn.forward(s)
        y = y.squeeze()  # Squeeze output ([batch_size, seq_size, 1] -> [batch_size, seq_size])
        y = self.sig.forward(y)

        return y
