import math
from typing import Iterable, List, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .abstract import Embedding, Model, Encoder, Decoder, Encoding
from .utils import get_mask


# ---------- Positional Encoding ---------- #


class LearnableEncoding(Encoding):
    def __init__(self, d: int, max_len: int):
        super().__init__()

        self.max_len = max_len
        self.encoding = nn.Embedding(max_len, d)

        nn.init.xavier_uniform_(self.encoding.weight)
        self.encoding._fill_padding_idx_with_zero()

    def forward(self, x: Tensor) -> Tensor:
        positions = torch.arange(self.max_len, dtype=torch.long, device=x.device)
        positions = positions.unsqueeze(0).expand([x.shape[0], self.max_len])
        embedding = self.encoding.forward(positions)

        x = x + embedding[:, : x.size(1), :]
        return x


class IdentityEncoding(Encoding):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x


# Code from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(Encoding):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:, : x.size(1), :]
        return x


# ---------- Embedding ---------- #


class AllEmbedding(Embedding):
    def __init__(self, n_items: int, d: int, g: int, n_ctx: int, n_attrs: int, enc: Encoding):
        super().__init__()

        self.d = d
        self.enc = enc

        self.items_embed = nn.Embedding(num_embeddings=n_items, embedding_dim=d, padding_idx=0)
        self.feats_embed = nn.Linear(in_features=n_ctx + n_attrs, out_features=g)
        self.joint_embed = nn.Linear(in_features=g + d, out_features=d)

        nn.init.xavier_uniform_(self.items_embed.weight)
        nn.init.xavier_uniform_(self.feats_embed.weight)
        nn.init.xavier_uniform_(self.joint_embed.weight)

        self.items_embed._fill_padding_idx_with_zero()
        nn.init.zeros_(self.feats_embed.bias)
        nn.init.zeros_(self.joint_embed.bias)

    def forward(self, x: Tensor, a: Tensor, c: Tensor, mask: Tensor, target: bool) -> Tensor:
        q = self.feats_embed.forward(torch.cat((a, c), dim=-1))
        z = self.items_embed.forward(x)
        z = z * (self.d**0.5)  # Scale embedding output
        e = self.joint_embed.forward(torch.cat((z, q), dim=-1))

        if not target:
            e = self.enc.forward(e)  # Positional encoding

        e = e * mask.unsqueeze(2)
        return e


class AttrCtxEmbedding(Embedding):
    def __init__(self, d: int, g: int, n_ctx: int, n_attrs: int, enc: Encoding):
        super().__init__()

        self.d = d
        self.enc = enc

        self.feats_embed = nn.Linear(in_features=n_ctx + n_attrs, out_features=g)
        self.joint_embed = nn.Linear(in_features=g, out_features=d)

        nn.init.xavier_uniform_(self.feats_embed.weight)
        nn.init.xavier_uniform_(self.joint_embed.weight)

        nn.init.zeros_(self.feats_embed.bias)
        nn.init.zeros_(self.joint_embed.bias)

    def forward(self, x: Tensor, a: Tensor, c: Tensor, mask: Tensor, target: bool) -> Tensor:
        q = self.feats_embed.forward(torch.cat((a, c), dim=-1))
        e = self.joint_embed.forward(q)

        if not target:
            e = self.enc.forward(e)  # Positional encoding

        e = e * mask.unsqueeze(2)
        return e


class AttrEmbedding(Embedding):
    def __init__(self, d: int, g: int, n_attrs: int, enc: Encoding):
        super().__init__()

        self.d = d
        self.enc = enc

        self.feats_embed = nn.Linear(in_features=n_attrs, out_features=g)
        self.joint_embed = nn.Linear(in_features=g, out_features=d)

        nn.init.xavier_uniform_(self.feats_embed.weight)
        nn.init.xavier_uniform_(self.joint_embed.weight)

        nn.init.zeros_(self.feats_embed.bias)
        nn.init.zeros_(self.joint_embed.bias)

    def forward(self, x: Tensor, a: Tensor, c: Tensor, mask: Tensor, target: bool) -> Tensor:
        q = self.feats_embed.forward(a)
        e = self.joint_embed.forward(q)

        if not target:
            e = self.enc.forward(e)  # Positional encoding

        e = e * mask.unsqueeze(2)
        return e


class IdEmbedding(Embedding):
    def __init__(self, n_items: int, d: int, enc: Encoding):
        super().__init__()

        self.d = d
        self.enc = enc
        self.items_embed = nn.Embedding(num_embeddings=n_items, embedding_dim=d, padding_idx=0)

        nn.init.xavier_uniform_(self.items_embed.weight)
        self.items_embed._fill_padding_idx_with_zero()

    def forward(self, x: Tensor, a: Tensor, c: Tensor, mask: Tensor, target: bool) -> Tensor:
        e = self.items_embed.forward(x)
        e = e * (self.d**0.5)  # Scale embedding output

        if not target:
            e = self.enc.forward(e)  # Positional encoding

        e = e * mask.unsqueeze(2)
        return e


class MLPIdEmbedding(Embedding):
    def __init__(self, n_items: int, d: int, g: int, enc: Encoding):
        super().__init__()

        self.d = d
        self.enc = enc
        self.items_embed = nn.Embedding(num_embeddings=n_items, embedding_dim=g, padding_idx=0)
        self.feats_embed = nn.Linear(in_features=g, out_features=d)

        nn.init.xavier_uniform_(self.items_embed.weight)
        nn.init.xavier_uniform_(self.feats_embed.weight)

        nn.init.zeros_(self.feats_embed.bias)
        self.items_embed._fill_padding_idx_with_zero()

    def forward(self, x: Tensor, a: Tensor, c: Tensor, mask: Tensor, target: bool) -> Tensor:
        e = self.items_embed.forward(x)
        e = e * (self.d**0.5)  # Scale embedding output
        e = self.feats_embed.forward(e)

        if not target:
            e = self.enc.forward(e)  # Positional encoding

        e = e * mask.unsqueeze(2)
        return e


# ---------- Attention ---------- #


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()

        assert embed_dim % num_heads == 0.0, "Embedding dim must be divisible by number of heads"

        self.d = embed_dim
        self.H = num_heads

        self.WQ = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.WK = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.WV = nn.Linear(in_features=embed_dim, out_features=embed_dim)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=dropout)

        nn.init.xavier_uniform_(self.WQ.weight)
        nn.init.xavier_uniform_(self.WK.weight)
        nn.init.xavier_uniform_(self.WV.weight)

        nn.init.zeros_(self.WQ.bias)
        nn.init.zeros_(self.WK.bias)
        nn.init.zeros_(self.WV.bias)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        q_mask: Tensor,
        k_mask: Tensor,
        causal: int = None,
        return_w: bool = False,
    ) -> Tensor:
        query = self.WQ.forward(query)
        key = self.WK.forward(key)
        value = self.WV.forward(value)

        query = torch.cat(torch.split(query, self.d // self.H, dim=2), dim=0)
        key = torch.cat(torch.split(key, self.d // self.H, dim=2), dim=0)
        value = torch.cat(torch.split(value, self.d // self.H, dim=2), dim=0)

        mat1, mat2 = q_mask.unsqueeze(1).transpose(1, 2), k_mask.unsqueeze(1)

        attn_mask = torch.bmm(mat1, mat2).bool()
        attn_mask = torch.tile(attn_mask, (self.H, 1, 1))
        attn_mask = torch.tril(attn_mask, diagonal=causal) if causal is not None else attn_mask  # Causality constraint
        add_mask = torch.where(attn_mask, 0.0, -(2**32) + 1.0)

        weights = torch.baddbmm(add_mask, query, key.transpose(1, 2))
        weights = weights / ((self.d / self.H) ** 0.5)
        weights = self.softmax.forward(weights)
        weights = weights * attn_mask

        out = self.dropout.forward(weights)
        out = torch.bmm(out, value)
        out = torch.cat(torch.split(out, out.shape[0] // self.H, dim=0), dim=2)

        if return_w:
            return weights, out
        else:
            return out


# ---------- Encoder / Decoder ---------- #


# Profile-level self-attention block
class SelfAttentionBlock(Encoder):
    def __init__(self, d: int, H: int, p: float, residual: bool):
        super().__init__()

        self.residual = residual

        # Attention
        self.norm1 = nn.LayerNorm(normalized_shape=d)
        self.attn = MultiHeadAttention(embed_dim=d, num_heads=H, dropout=p)

        # FFN
        self.norm2 = nn.LayerNorm(normalized_shape=d)
        self.ffn_1 = nn.Conv1d(in_channels=d, out_channels=d, kernel_size=1)
        self.lrelu = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(p=p)

        self.ffn_2 = nn.Conv1d(in_channels=d, out_channels=d, kernel_size=1)
        self.dropout2 = nn.Dropout(p=p)

        nn.init.xavier_uniform_(self.ffn_1.weight)
        nn.init.xavier_uniform_(self.ffn_2.weight)

        nn.init.zeros_(self.ffn_1.bias)  # type: ignore
        nn.init.zeros_(self.ffn_2.bias)  # type: ignore

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        q = self.norm1.forward(x)
        s = self.attn.forward(q, x, x, q_mask=mask, k_mask=mask, causal=0)

        if self.residual:
            s = torch.add(s, q)  # Residual connection

        s = self.norm2.forward(s)
        f = s.transpose(1, 2).contiguous()  # Change dim order to get channel dim to middle

        f = self.ffn_1.forward(f)
        f = self.lrelu.forward(f)
        f = self.dropout1.forward(f)

        f = self.ffn_2.forward(f)
        f = self.dropout2.forward(f)
        f = f.transpose(1, 2).contiguous()  # Change dim order back

        if self.residual:
            f = torch.add(f, s)  # Residual connection

        return f


# Target-level cross-attention block
class CrossAttentionBlock(Decoder):
    def __init__(self, d: int, H: int, p: float, residual: bool):
        super().__init__()

        self.residual = residual

        # Attention
        self.attn = MultiHeadAttention(embed_dim=d, num_heads=H, dropout=p)

        # FFN
        self.ffn = nn.Linear(in_features=d, out_features=1)
        self.sig = nn.Sigmoid()

        nn.init.xavier_uniform_(self.ffn.weight)
        nn.init.zeros_(self.ffn.bias)

    def forward(self, o: Tensor, o_mask: Tensor, p: Tensor, p_mask: Tensor) -> Tensor:
        causal = -1 if self.training else None
        s = self.attn.forward(o, p, p, q_mask=o_mask, k_mask=p_mask, causal=causal)

        if self.residual:
            s = torch.add(s, o)  # Residual connection

        y = self.ffn.forward(s)
        y = y.squeeze()  # Squeeze output ([batch_size, seq_size, 1] -> [batch_size, seq_size])
        y = self.sig.forward(y)

        return y


class DotProduct(Decoder):
    def __init__(self) -> None:
        super().__init__()

        self.sig = nn.Sigmoid()

    def forward(self, o: Tensor, o_mask: Tensor, p: Tensor, p_mask: Tensor) -> Tensor:
        if self.training:
            y = torch.sum(p * o, dim=-1)  # Dot-product between profile items and target items
        else:
            y = torch.sum(p[:, -1:, :] * o, dim=-1)  # Dot-product between last profile item and target items

        y = self.sig.forward(y)
        return y


class WeightedDotProduct(Decoder):
    def __init__(self, gamma: float, seq_len: int, normalize: bool, device: str):
        super().__init__()

        self.norm = normalize
        self.W = gamma ** torch.arange(0, seq_len, device=device).unsqueeze(0).repeat(seq_len, 1)
        self.W = self.W.tril().unsqueeze(-1)
        self.sig = nn.Sigmoid()

    def forward(self, o: Tensor, o_mask: Tensor, p: Tensor, p_mask: Tensor) -> Tensor:
        pw = p.unsqueeze(2).repeat(1, 1, p.size(1), 1)
        p = torch.sum(pw * self.W, dim=2)

        if self.norm:
            p = torch.nn.functional.normalize(p, dim=2)
            o = torch.nn.functional.normalize(o, dim=2)

        if self.training:
            y = torch.sum(p * o, dim=-1)  # Dot-product between profile items and target items
        else:
            y = torch.sum(p[:, -1:, :] * o, dim=-1)  # Dot-product between last profile item and target items

        if self.norm:
            y = (y + 1.0) / 2.0
        else:
            y = self.sig.forward(y)

        return y


# ---------- CARCA ---------- #


class CARCA(Model):
    def __init__(self, d: int, p: float, emb: Embedding, enc: Iterable[Encoder], dec: Decoder):
        super().__init__()

        self.embeds = emb
        self.dropout = nn.Dropout(p=p)
        self.encoder = enc
        self.norm = nn.LayerNorm(normalized_shape=d)
        self.decoder = dec

    def forward(self, profile: Tuple[Tensor, Tensor, Tensor], targets: List[Tuple[Tensor, Tensor, Tensor]]) -> Tensor:
        p_x, p_a, p_c = profile
        p_mask = get_mask(p_x)

        p_e = self.embeds.forward(p_x, p_a, p_c, p_mask, False)
        p_e = self.dropout.forward(p_e)

        for block in self.encoder:
            p_e = block.forward(p_e, p_mask)

        p_e = self.norm.forward(p_e)
        y_preds = []

        for o_x, o_a, o_c in targets:
            o_mask = get_mask(o_x)
            o_e = self.embeds.forward(o_x, o_a, o_c, o_mask, True)

            y_pred = self.decoder.forward(o_e, o_mask, p_e, p_mask)
            y_preds.append(y_pred)

        return torch.cat(y_preds, dim=-1)


# ---------- Binary cross-entropy loss with masking ---------- #


class BinaryCrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred: Tensor, y_true: Tensor, mask: Tensor, eps: float = 1e-8) -> Tensor:
        loss = -(y_true * torch.log(y_pred + eps) + (1.0 - y_true) * torch.log(1.0 - y_pred + eps))
        loss = torch.sum(loss * mask) / torch.sum(mask)
        return loss
