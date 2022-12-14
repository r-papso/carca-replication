import os
from datetime import datetime
from typing import Tuple, Union

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler

from .model import CARCA, BinaryCrossEntropy
from .utils import get_mask, to


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


def evaluate(model: CARCA, loader: DataLoader, device: str, k: int) -> Tuple[float, float, float]:
    model = model.eval().to(device)
    loss_fn = BinaryCrossEntropy()
    HR, NDCG, total, sum_loss = 0, 0, 0, 0

    with torch.no_grad():
        for batch in loader:
            p_x, p_q, o_x, o_q, y_true = to(*batch, device=device)

            y_pred = model.forward(profile=(p_x, p_q), targets=[(o_x, o_q)])
            loss_mask = get_mask(o_x)
            loss = loss_fn.forward(y_pred, y_true, loss_mask)
            sum_loss += loss.item()

            HR += compute_HR(y_pred, y_true, k)
            NDCG += compute_NDCG(y_pred, y_true, k)
            total += y_true.shape[0]

    return HR / total, NDCG / total, sum_loss / len(loader)


def train(
    model: CARCA,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    optim: Optimizer,
    epochs: int,
    top_k: int = 10,
    verbose: int = 1,
    early_stop: int = 10,
    checkpoint: str = "model",
    scheduler: Union[_LRScheduler, None] = None,
) -> CARCA:
    os.makedirs(checkpoint, exist_ok=True)

    loss_fn = BinaryCrossEntropy()
    model = model.train().to(device)
    best_hr, no_improve = 0, 0

    for epoch in range(1, epochs + 1):
        sum_loss = 0

        for i, batch in enumerate(train_loader, start=1):
            p_x, p_q, o_x, o_q, y_true = to(*batch, device=device)
            pos_x, neg_x = torch.split(o_x, o_x.shape[1] // 2, dim=1)
            pos_q, neg_q = torch.split(o_q, o_q.shape[1] // 2, dim=1)

            optim.zero_grad()
            y_pred = model.forward(profile=(p_x, p_q), targets=[(pos_x, pos_q), (neg_x, neg_q)])
            loss_mask = get_mask(o_x)
            loss = loss_fn.forward(y_pred, y_true, loss_mask)

            loss.backward()
            optim.step()
            sum_loss += loss.item()

            if verbose == 2:
                time = datetime.now().strftime("%H:%M:%S")
                print(f"{time} - Batch {i:03d}: Loss = {(sum_loss / i):.4f}")

        if verbose in [1, 2]:
            time = datetime.now().strftime("%H:%M:%S")
            print(f"{time} - Epoch {(epoch):03d}: Loss = {(sum_loss / len(train_loader)):.4f}")

        if scheduler is not None:
            scheduler.step()

        # Evaluate model
        HR, NDCG, loss = evaluate(model, val_loader, device, top_k)

        if HR > best_hr:
            fs = [f for f in os.listdir(checkpoint) if os.path.isfile(os.path.join(checkpoint, f))]
            _ = [os.remove(os.path.join(checkpoint, f)) for f in fs]

            best_hr = HR
            no_improve = 0
            torch.save(model, os.path.join(checkpoint, f"{epoch:03d}_{HR:.4f}.pth"))
        else:
            no_improve += 1

        if verbose in [1, 2]:
            time = datetime.now().strftime("%H:%M:%S")
            print(f"{time} - Epoch {epoch:03d}: Loss = {loss:.4f} HR = {HR:.4f}, NDCG = {NDCG:.4f}")
            model = model.train().to(device)

        if no_improve >= early_stop:
            print(f"No improvement in {no_improve} epochs, early stopping...")
            break

    return model
