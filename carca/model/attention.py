import torch
import torch.nn as nn
from torch import Tensor


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()

        assert embed_dim % num_heads == 0.0, "Embedding dim must be divisible by number of heads"

        self.d = embed_dim
        self.H = num_heads

        self.WQ = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.WK = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.WV = nn.Linear(in_features=embed_dim, out_features=embed_dim)

        self.qrelu = nn.LeakyReLU(negative_slope=0.2)
        self.krelu = nn.LeakyReLU(negative_slope=0.2)
        self.vrelu = nn.LeakyReLU(negative_slope=0.2)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=dropout)

        nn.init.xavier_uniform_(self.WQ.weight)
        nn.init.xavier_uniform_(self.WK.weight)
        nn.init.xavier_uniform_(self.WV.weight)

        nn.init.zeros_(self.WQ.bias)
        nn.init.zeros_(self.WK.bias)
        nn.init.zeros_(self.WV.bias)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, q_mask: Tensor, k_mask: Tensor) -> Tensor:
        query = self.WQ.forward(query)
        key = self.WK.forward(key)
        value = self.WV.forward(value)

        query = self.qrelu.forward(query)
        key = self.krelu.forward(key)
        value = self.vrelu.forward(value)

        query = torch.cat(torch.split(query, self.d // self.H, dim=2), dim=0)
        key = torch.cat(torch.split(key, self.d // self.H, dim=2), dim=0)
        value = torch.cat(torch.split(value, self.d // self.H, dim=2), dim=0)

        mat1, mat2 = q_mask.unsqueeze(1).transpose(1, 2), k_mask.unsqueeze(1)
        attn_mask = torch.bmm(mat1, mat2).bool()
        attn_mask = torch.tile(attn_mask, (self.H, 1, 1))
        add_mask = torch.where(attn_mask, 0.0, -(2**32) + 1.0)

        out = torch.baddbmm(add_mask, query, key.transpose(1, 2))
        out = out / (self.d / self.H) ** 0.5
        out = self.softmax.forward(out)

        weight_mask = torch.tile(q_mask, (self.H, 1)).unsqueeze(2)
        out = out * weight_mask
        out = self.dropout.forward(out)

        out = torch.bmm(out, value)
        out = torch.cat(torch.split(out, out.shape[0] // self.H, dim=0), dim=2)

        return out
