import torch
import torch.nn as nn
from torch import Tensor


class Embeddings(nn.Module):
    def __init__(self, n_items: int, d: int, g: int, n_ctx: int, n_attrs: int):
        super().__init__()

        self.items_embed = nn.Embedding(num_embeddings=n_items, embedding_dim=d, padding_idx=0)
        self.feats_embed = nn.Linear(in_features=n_ctx + n_attrs, out_features=g)
        self.joint_embed = nn.Linear(in_features=g + d, out_features=d)

    def forward(self, x: Tensor, q: Tensor, mask: Tensor) -> Tensor:
        z = self.items_embed.forward(x)
        q = self.feats_embed.forward(q)

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
        self.attention = nn.MultiheadAttention(embed_dim=d, num_heads=H, batch_first=True)
        self.dropout1 = nn.Dropout(p=p)

        # FFN
        self.norm2 = nn.LayerNorm(normalized_shape=d)
        self.ffn_1 = nn.Conv1d(in_channels=d, out_channels=d, kernel_size=1)
        self.activation = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(p=p)

        self.ffn_2 = nn.Conv1d(in_channels=d, out_channels=d, kernel_size=1)
        self.dropout3 = nn.Dropout(p=p)
        self.norm3 = nn.LayerNorm(normalized_shape=d)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        key_mask = mask == 0
        exp_mask = mask.unsqueeze(2)

        q = self.norm1.forward(x)

        s, _ = self.attention.forward(q, x, x, key_padding_mask=key_mask, need_weights=False)
        s = self.dropout1.forward(s)
        s = s * exp_mask

        if self.residual:
            s = torch.mul(s, x)  # Multiplicative residual connection

        s = self.norm2.forward(s)
        f = s.permute(0, 2, 1)  # Change dim order to get channel dim to middle (for Conv1d)

        f = self.ffn_1.forward(f)
        f = self.activation.forward(f)
        f = self.dropout2.forward(f)

        f = f.permute(0, 2, 1)
        f = f * exp_mask
        f = f.permute(0, 2, 1)

        f = self.ffn_2.forward(f)
        f = self.dropout3.forward(f)
        f = f.permute(0, 2, 1)
        f = f * exp_mask

        if self.residual:
            f = torch.mul(f, s)  # Multiplicative residual connection

        f = self.norm3.forward(f)
        return f


# Target-level cross-attention block
class CrossAttentionBlock(nn.Module):
    def __init__(self, d: int, H: int, p: float, residual: bool):
        super().__init__()

        self.residual = residual

        # Attention
        self.attention = nn.MultiheadAttention(embed_dim=d, num_heads=H, batch_first=True)
        self.dropout = nn.Dropout(p=p)

        # FFN
        self.ffn = nn.Conv1d(in_channels=d, out_channels=1, kernel_size=1)
        self.sig = nn.Sigmoid()

    def forward(self, e: Tensor, e_mask: Tensor, f: Tensor, f_mask: Tensor) -> Tensor:
        key_mask = f_mask == 0
        exp_mask = e_mask.unsqueeze(2)

        s, _ = self.attention.forward(e, f, f, key_padding_mask=key_mask, need_weights=False)
        s = self.dropout.forward(s)
        s = s * exp_mask

        if self.residual:
            s = torch.mul(s, e)  # Multiplicative residual connection

        s = s.permute(0, 2, 1)  # Change dim order to get channel dim to middle (for Conv1d)

        y = self.ffn.forward(s)
        y = self.sig.forward(y)
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
        self.sa_blocks = nn.ModuleList([SelfAttentionBlock(d, H, p, res_sa) for _ in range(B)])
        # self.sa_blocks = nn.Sequential(*[SelfAttentionBlock(d, H, p, res_sa) for _ in range(B)])
        self.ca_blocks = CrossAttentionBlock(d, H, p, res_ca)

    def forward(
        self, p_x: Tensor, p_q: Tensor, p_mask: Tensor, o_x: Tensor, o_q: Tensor, o_mask: Tensor
    ) -> Tensor:
        p_e = self.embeds.forward(p_x, p_q, p_mask)
        o_e = self.embeds.forward(o_x, o_q, o_mask)

        for block in self.sa_blocks:
            p_e = block.forward(p_e, p_mask)

        y_pred = self.ca_blocks.forward(o_e, o_mask, p_e, p_mask)
        return y_pred


class BinaryCrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred: Tensor, y_true: Tensor, mask: Tensor, eps: float = 1e-8) -> Tensor:
        loss = -(y_true * torch.log(y_pred + eps) + (1.0 - y_true) * torch.log(1.0 - y_pred + eps))
        loss = torch.sum(loss * mask) / torch.sum(y_true)
        return loss
