from typing import List, Tuple, Union

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

        self.WQ = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=False)
        self.WK = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=False)
        self.WV = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=False)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, attn_mask: Union[Tensor, None] = None
    ) -> Tensor:
        query = self.WQ.forward(query)
        key = self.WK.forward(key)
        value = self.WV.forward(value)

        query = torch.concat(torch.split(query, self.d // self.H, dim=2), dim=0)
        key = torch.concat(torch.split(key, self.d // self.H, dim=2), dim=0)
        value = torch.concat(torch.split(value, self.d // self.H, dim=2), dim=0)

        if attn_mask is not None:
            add_mask = torch.where(attn_mask, 0.0, -1e6)
            out = torch.baddbmm(add_mask, query, key.transpose(1, 2))
        else:
            out = torch.bmm(query, key.transpose(1, 2))

        out /= (self.d / self.H) ** 0.5
        out = self.softmax.forward(out)
        out = out * attn_mask
        out = self.dropout.forward(out)

        out = torch.bmm(out, value)
        out = torch.concat(torch.split(out, out.shape[0] // self.H, dim=0), dim=2)

        return out


class Embeddings(nn.Module):
    def __init__(self, n_items: int, d: int, g: int, n_ctx: int, n_attrs: int):
        super().__init__()

        self.d = d

        self.items_embed = nn.Embedding(num_embeddings=n_items, embedding_dim=d, padding_idx=0)
        self.feats_embed = nn.Linear(in_features=n_ctx + n_attrs, out_features=g)
        self.joint_embed = nn.Linear(in_features=g + d, out_features=d)

    def forward(self, x: Tensor, q: Tensor, mask: Tensor) -> Tensor:
        z = self.items_embed.forward(x)
        z = z * (self.d**0.5)  # Scale embedding output
        q = self.feats_embed.forward(q)

        e = self.joint_embed.forward(torch.cat((z, q), dim=-1))
        e = e * mask.unsqueeze(2)

        return e


# Profile-level self-attention block
class SelfAttentionBlock(nn.Module):
    def __init__(self, d: int, H: int, p: float, residual: bool):
        super().__init__()

        self.H = H
        self.residual = residual

        # Attention
        self.norm1 = nn.LayerNorm(normalized_shape=d)
        # self.attn = nn.MultiheadAttention(embed_dim=d, num_heads=H, dropout=p, batch_first=True)
        self.attn = MultiHeadAttention(embed_dim=d, num_heads=H, dropout=p)

        # FFN
        self.norm2 = nn.LayerNorm(normalized_shape=d)
        self.ffn_1 = nn.Conv1d(in_channels=d, out_channels=d, kernel_size=1)
        self.activation = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(p=p)

        self.ffn_2 = nn.Conv1d(in_channels=d, out_channels=d, kernel_size=1)
        self.dropout3 = nn.Dropout(p=p)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        q = self.norm1.forward(x)

        # s, _ = self.attn.forward(q, x, x, key_padding_mask=mask == 0, need_weights=False)
        # s = s * mask.unsqueeze(2)
        mat1, mat2 = mask.unsqueeze(1).transpose(1, 2), mask.unsqueeze(1)
        attn_mask = torch.bmm(mat1, mat2).bool()
        attn_mask = torch.tile(attn_mask, (self.H, 1, 1))

        s = self.attn.forward(q, x, x, attn_mask=attn_mask)

        if self.residual:
            s = torch.mul(s, q)  # Multiplicative residual connection

        s = self.norm2.forward(s)
        f = s.transpose(1, 2)  # Change dim order to get channel dim to middle (for Conv1d)

        f = self.ffn_1.forward(f)
        f = self.activation.forward(f)
        f = self.dropout2.forward(f)
        # f = f * mask.unsqueeze(1)

        f = self.ffn_2.forward(f)
        f = self.dropout3.forward(f)
        # f = f * mask.unsqueeze(1)
        f = f.transpose(1, 2)

        if self.residual:
            f = torch.add(f, s)  # Multiplicative residual connection

        return f


# Target-level cross-attention block
class CrossAttentionBlock(nn.Module):
    def __init__(self, d: int, H: int, p: float, residual: bool):
        super().__init__()

        self.H = H
        self.residual = residual

        # Attention
        # self.attn = nn.MultiheadAttention(embed_dim=d, num_heads=H, dropout=p, batch_first=True)
        self.attn = MultiHeadAttention(embed_dim=d, num_heads=H, dropout=p)

        # FFN
        self.ffn = nn.Conv1d(in_channels=d, out_channels=1, kernel_size=1)
        self.sig = nn.Sigmoid()

    def forward(self, e: Tensor, e_mask: Tensor, f: Tensor, f_mask: Tensor) -> Tensor:
        mat1, mat2 = e_mask.unsqueeze(1).transpose(1, 2), f_mask.unsqueeze(1)
        attn_mask = torch.bmm(mat1, mat2).bool()
        attn_mask = torch.tile(attn_mask, (self.H, 1, 1))

        s = self.attn.forward(e, f, f, attn_mask=attn_mask)
        # s, _ = self.attn.forward(e, f, f, key_padding_mask=f_mask == 0, need_weights=False)
        # s = s * e_mask.unsqueeze(2)

        if self.residual:
            s = torch.mul(s, e)  # Multiplicative residual connection

        s = s.transpose(1, 2)  # Change dim order to get channel dim to middle (for Conv1d)

        y = self.ffn.forward(s)
        y = self.sig.forward(y)
        # y = y * e_mask.unsqueeze(1)
        y = y.squeeze()  # Squeeze output ([batch_size, 1, seq_size] -> [batch_size, seq_size])

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

        for target in targets:
            o_x, o_q = target
            o_mask = get_mask(o_x)
            o_e = self.embeds.forward(o_x, o_q, o_mask)

            y_pred = self.ca_blocks.forward(o_e, o_mask, p_e, p_mask)
            y_preds.append(y_pred)

        return torch.concat(y_preds, dim=-1)

    # def forward(
    #     self, p_x: Tensor, p_q: Tensor, p_mask: Tensor, o_x: Tensor, o_q: Tensor, o_mask: Tensor
    # ) -> Tensor:
    #     p_e = self.embeds.forward(p_x, p_q, p_mask)
    #     p_e = self.dropout.forward(p_e)
    #     o_e = self.embeds.forward(o_x, o_q, o_mask)


#
#     for block in self.sa_blocks:
#         p_e = block.forward(p_e, p_mask)
#
#     p_e = self.norm.forward(p_e)
#
#     y_pred = self.ca_blocks.forward(o_e, o_mask, p_e, p_mask)
#     return y_pred


class BinaryCrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred: Tensor, y_true: Tensor, mask: Tensor, eps: float = 1e-8) -> Tensor:
        loss = -(y_true * torch.log(y_pred + eps) + (1.0 - y_true) * torch.log(1.0 - y_pred + eps))
        loss = torch.sum(loss * mask) / torch.sum(mask)
        return loss
