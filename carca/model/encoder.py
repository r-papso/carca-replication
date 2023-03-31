import torch
import torch.nn as nn
from torch import Tensor

from .attention import MultiHeadAttention


# Profile-level self-attention block
class SelfAttentionBlock(nn.Module):
    def __init__(self, d: int, H: int, p: float, residual: bool):
        super().__init__()

        self.residual = residual

        # Attention
        self.norm1 = nn.LayerNorm(normalized_shape=d)
        self.attn = MultiHeadAttention(embed_dim=d, num_heads=H, dropout=p)

        # FFN
        self.norm2 = nn.LayerNorm(normalized_shape=d)
        self.ffn_1 = nn.Conv1d(in_channels=d, out_channels=d, kernel_size=1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        self.dropout1 = nn.Dropout(p=p)

        self.ffn_2 = nn.Conv1d(in_channels=d, out_channels=d, kernel_size=1)
        self.dropout2 = nn.Dropout(p=p)

        nn.init.xavier_uniform_(self.ffn_1.weight)
        nn.init.xavier_uniform_(self.ffn_2.weight)

        nn.init.zeros_(self.ffn_1.bias)  # type: ignore
        nn.init.zeros_(self.ffn_2.bias)  # type: ignore

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        q = self.norm1.forward(x)
        s = self.attn.forward(q, x, x, q_mask=mask, k_mask=mask)

        if self.residual:
            s = torch.mul(s, q)  # Multiplicative residual connection

        s = self.norm2.forward(s)
        f = s.transpose(1, 2).contiguous()  # Change dim order to get channel dim to middle

        f = self.ffn_1.forward(f)
        f = self.lrelu.forward(f)
        f = self.dropout1.forward(f)

        f = self.ffn_2.forward(f)
        f = self.dropout2.forward(f)
        f = f.transpose(1, 2).contiguous()  # Change dim order back

        if self.residual:
            f = torch.add(f, s)  # Additive residual connection

        return f
