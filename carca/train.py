from datetime import datetime
from typing import Tuple

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .model import BinaryCrossEntropy
from .utils import to


def compute_HR(y_pred: torch.Tensor, y_true: torch.Tensor, k: int) -> float:
    y_pred_sort, idxs = torch.sort(y_pred, descending=True)
    y_true_sort = torch.gather(y_true, dim=1, index=idxs)

    top_k = y_true_sort[:, :k]

    return torch.sum(top_k).item()


def compute_NDCG(y_pred: torch.Tensor, y_true: torch.Tensor, k: int) -> float:
    y_pred_sort, idxs = torch.sort(y_pred, descending=True)
    y_true_sort = torch.gather(y_true, dim=1, index=idxs)

    top_k = y_true_sort[:, :k]
    ranks = torch.nonzero(top_k)[:, 1]
    scores = 1.0 / torch.log2(ranks + 2)

    return torch.sum(scores).item()


def evaluate(model: nn.Module, loader: DataLoader, device: str, k: int) -> Tuple[float, float]:
    model = model.eval().to(device)
    HR, NDCG, total = 0, 0, 0

    with torch.no_grad():
        for batch in loader:
            p_x, p_q, o_x, o_q, y_true, mask = to(*batch, device=device)

            y_pred = model(p_x, p_q, o_x, o_q)
            HR += compute_HR(y_pred, y_true, k)
            NDCG += compute_NDCG(y_pred, y_true, k)
            total += y_true.shape[0]

    return HR / total, NDCG / total


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    optim: Optimizer,
    epochs: int,
    top_k: int = 10,
    verbose: int = 1,
) -> nn.Module:
    loss_fn = BinaryCrossEntropy()
    model = model.train().to(device)

    for epoch in range(1, epochs + 1):
        sum_loss = 0

        for i, batch in enumerate(train_loader, start=1):
            p_x, p_q, o_x, o_q, y_true, mask = to(*batch, device=device)

            optim.zero_grad()
            y_pred = model(p_x, p_q, o_x, o_q)
            loss = loss_fn(y_pred, y_true, mask)
            loss.backward()
            optim.step()

            sum_loss += loss.item()

            if verbose == 2:
                time = datetime.now().strftime("%H:%M:%S")
                print(f"{time} - Batch {i:03d}: Avg Loss = {(sum_loss / i):.4f}")

        if verbose in [1, 2]:
            time = datetime.now().strftime("%H:%M:%S")
            print(f"{time} - Epoch {(epoch):03d}: Avg Loss = {(sum_loss / len(train_loader)):.4f}")

        if val_loader is not None and verbose in [1, 2]:
            # Evaluate model
            HR, NDCG = evaluate(model, val_loader, device, top_k)
            time = datetime.now().strftime("%H:%M:%S")
            print(f"{time} - Epoch {(epoch):03d}: HR = {HR:.4f}, NDCG = {NDCG:.4f}")
            model = model.train().to(device)

    return model
