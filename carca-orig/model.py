from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .utils import get_mask


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

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, q_mask: Tensor, k_mask: Tensor
    ) -> Tensor:
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


class Embeddings(nn.Module):
    def __init__(self, n_items: int, d: int, g: int, n_ctx: int, n_attrs: int):
        super().__init__()

        self.d = d

        self.items_embed = nn.Embedding(num_embeddings=n_items, embedding_dim=d, padding_idx=0)
        self.feats_embed = nn.Linear(in_features=n_ctx + n_attrs, out_features=g)
        self.joint_embed = nn.Linear(in_features=g + d, out_features=d)

        nn.init.xavier_uniform_(self.items_embed.weight)
        # nn.init.normal_(self.items_embed.weight, std=0.01)
        nn.init.normal_(self.feats_embed.weight, std=0.01)
        nn.init.normal_(self.joint_embed.weight, std=0.01)

        self.items_embed._fill_padding_idx_with_zero()
        nn.init.zeros_(self.feats_embed.bias)
        nn.init.zeros_(self.joint_embed.bias)

    def forward(self, x: Tensor, q: Tensor, mask: Tensor, scale: bool = True) -> Tensor:
        q = self.feats_embed.forward(q)
        z = self.items_embed.forward(x)

        if scale:
            z = z * (self.d**0.5)  # Scale embedding output

        e = self.joint_embed.forward(torch.cat((z, q), dim=-1))
        e = e * mask.unsqueeze(2)

        return e


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

        f = f * mask.unsqueeze(2)
        return f


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

        s = s * e_mask.unsqueeze(2)

        y = self.ffn.forward(s)
        y = y.squeeze()  # Squeeze output ([batch_size, seq_size, 1] -> [batch_size, seq_size])
        y = y * e_mask
        y = self.sig.forward(y)

        return y


class CARCA(nn.Module):
    def __init__(
        self,
        n_items: int,
        d: int,
        g: int,
        n_ctx: int,
        n_attrs: int,
        H: int,
        p: float,
        B: int,
        res_sa: bool,
        res_ca: bool,
    ):
        super().__init__()

        self.embeds = Embeddings(n_items, d, g, n_ctx, n_attrs)
        self.dropout = nn.Dropout(p=p)
        self.sa_blocks = nn.ModuleList([SelfAttentionBlock(d, H, p, res_sa) for _ in range(B)])
        self.norm = nn.LayerNorm(normalized_shape=d)
        self.ca_blocks = CrossAttentionBlock(d, H, p, res_ca)

    def forward(
        self, profile: Tuple[Tensor, Tensor], targets: List[Tuple[Tensor, Tensor]]
    ) -> Tensor:
        p_x, p_q = profile
        p_mask = get_mask(p_x)

        p_e = self.embeds.forward(p_x, p_q, p_mask)
        p_e = self.dropout.forward(p_e)

        for block in self.sa_blocks:
            p_e = block.forward(p_e, p_mask)

        p_e = self.norm.forward(p_e)
        y_preds = []

        for o_x, o_q in targets:
            o_mask = get_mask(o_x)
            o_e = self.embeds.forward(o_x, o_q, o_mask, scale=False)

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
