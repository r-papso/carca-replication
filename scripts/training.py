import argparse
import json
import os
import random
import sys

import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset

sys.path.append("..")

from src.abstract import Decoder, Embedding, Encoding
from src.carca import (
    CARCA,
    AllEmbedding,
    AttrCtxEmbedding,
    AttrEmbedding,
    CrossAttentionBlock,
    DotProduct,
    IdEmbedding,
    IdentityEncoding,
    LearnableEncoding,
    PositionalEncoding,
    SelfAttentionBlock,
)
from src.data import CARCADataset, load_attrs, load_ctx, load_profiles
from src.train import train

parser = argparse.ArgumentParser()

parser.add_argument("--out_dir", type=str, default="output")

parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--seq_len", type=int, default=50)
parser.add_argument("--n_blocks", type=int, default=2)
parser.add_argument("--n_heads", type=int, default=1)
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--l2_reg", type=float, default=0.00001)
parser.add_argument("--d_dim", type=int, default=50)
parser.add_argument("--g_dim", type=int, default=250)
parser.add_argument("--residual_sa", type=bool, default=True)
parser.add_argument("--residual_ca", type=bool, default=True)
parser.add_argument("--epochs", type=int, default=500)
parser.add_argument("--early_stop", type=int, default=30)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--beta1", type=float, default=0.9)
parser.add_argument("--beta2", type=float, default=0.98)
parser.add_argument("--device", type=str, default="cuda")

parser.add_argument("--encoding", type=str, default="identity")
parser.add_argument("--embedding", type=str, default="id")
parser.add_argument("--decoder", type=str, default="dot")


def get_encoding(args) -> Encoding:
    if args.encoding.lower() == "identity":
        return IdentityEncoding()
    elif args.encoding.lower() == "learnable":
        return LearnableEncoding(args.d_dim, args.seq_len)
    elif args.encoding.lower() == "positional":
        return PositionalEncoding(args.d_dim, args.seq_len)
    else:
        raise ValueError(f"Unknown encoding type: {args.encoding}")


def get_embedding(args, n_items: int, n_ctx: int, n_attrs: int, enc: Encoding) -> Embedding:
    if args.embedding.lower() == "id":
        return IdEmbedding(n_items, args.d_dim, enc)
    elif args.embedding.lower() == "attr":
        return AttrEmbedding(args.d_dim, args.g_dim, n_attrs, enc)
    elif args.embedding.lower() == "attrctx":
        return AttrCtxEmbedding(args.d_dim, args.g_dim, n_ctx, n_attrs, enc)
    elif args.embedding.lower() == "all":
        return AllEmbedding(n_items, args.d_dim, args.g_dim, n_ctx, n_attrs, enc)
    else:
        raise ValueError(f"Unknown embedding type: {args.embedding}")


def get_decoder(args) -> Decoder:
    if args.decoder.lower() == "ca":
        return CrossAttentionBlock(args.d_dim, args.n_heads, args.dropout, args.residual_ca)
    elif args.decoder.lower() == "dot":
        return DotProduct()
    else:
        raise ValueError(f"Unknown decoder type: {args.decoder}")


if __name__ == "__main__":
    args = parser.parse_args()

    os.makedirs(f"../results/{args.out_dir}", exist_ok=True)
    with open(f"../results/{args.out_dir}/args.json", "w") as file:
        file.write(json.dumps(vars(args)))

    attrs = load_attrs("video_games_onehot_5core.dat")
    ctx = load_ctx("video_games_ctx_5core.dat")
    user_ids, item_ids, profiles = load_profiles("video_games_sorted_5core.txt")

    n_items = attrs.shape[0]
    n_ctx = next(iter(ctx.values())).shape[0]
    n_attrs = attrs.shape[1]

    train_data = CARCADataset(
        user_ids=user_ids,
        item_ids=item_ids,
        profiles=profiles,
        attrs=attrs,
        ctx=ctx,
        profile_seq_len=args.seq_len,
        target_seq_len=100,
        mode="train",
    )
    val_data = CARCADataset(
        user_ids=user_ids,
        item_ids=item_ids,
        profiles=profiles,
        attrs=attrs,
        ctx=ctx,
        profile_seq_len=args.seq_len,
        target_seq_len=100,
        mode="val",
    )
    test_data = CARCADataset(
        user_ids=user_ids,
        item_ids=item_ids,
        profiles=profiles,
        attrs=attrs,
        ctx=ctx,
        profile_seq_len=args.seq_len,
        target_seq_len=100,
        mode="test",
    )

    val_idx = random.sample(range(len(val_data)), 10_000) if len(val_data) > 10_000 else range(len(val_data))
    val_sub = Subset(val_data, val_idx)
    test_idx = random.sample(range(len(test_data)), 10_000) if len(test_data) > 10_000 else range(len(test_data))
    test_sub = Subset(test_data, test_idx)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_sub, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_sub, batch_size=args.batch_size, shuffle=False, num_workers=0)

    encoding = get_encoding(args)
    embedding = get_embedding(args, n_items, n_ctx, n_attrs, encoding)
    encoder = nn.ModuleList(
        [SelfAttentionBlock(args.d_dim, args.n_heads, args.dropout, args.residual_sa) for _ in range(args.n_blocks)]
    )
    decoder = get_decoder(args)

    model = CARCA(d=args.d_dim, p=args.dropout, emb=embedding, enc=encoder, dec=decoder)
    model = model.to(args.device)
    optim = Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_reg, betas=(args.beta1, args.beta2))

    model = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=args.device,
        optim=optim,
        epochs=args.epochs,
        early_stop=args.early_stop,
        datadir=f"../results/{args.out_dir}"
        # scheduler=scheduler
    )
