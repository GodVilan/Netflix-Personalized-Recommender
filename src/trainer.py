"""
trainer.py  —  DARE-Rec rewrite (2026-04-10)

Training loop exclusively for LightGCN (the only neural model in DARE-Rec).
TemporalEASE and ImplicitALS are non-neural; they have their own fit() methods.

Key decisions:
  BPR loss (not InfoNCE): LightGCN's original paper uses BPR.
  1 negative per positive: standard for BPR on ML-1M.
  Adam optimizer with lr=1e-3, weight_decay=1e-4.
  Early stopping on val NDCG@10 with patience=10.
"""

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Callable


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[Device] Using: {device}", end=" ")
    if device.type == "cuda":
        print(f"({torch.cuda.get_device_name(0)})")
    elif device.type == "mps":
        print("(Apple Silicon GPU)")
    else:
        print("(CPU)")
    return device


class BPRDataset(Dataset):
    """
    BPR triplet dataset: (user, pos_item, neg_item).
    One negative sampled per positive per epoch (re-sampled each epoch).
    """
    def __init__(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        n_items: int,
        seen_items: Optional[dict] = None,
    ):
        self.users     = user_ids
        self.items     = item_ids
        self.n_items   = n_items
        self.seen      = seen_items or {}
        self.rng       = np.random.default_rng(42)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = int(self.users[idx])
        pos  = int(self.items[idx])
        # Sample 1 valid negative
        seen = self.seen.get(user, set())
        neg  = int(self.rng.integers(0, self.n_items))
        while neg in seen:
            neg = int(self.rng.integers(0, self.n_items))
        return (
            torch.tensor(user, dtype=torch.long),
            torch.tensor(pos,  dtype=torch.long),
            torch.tensor(neg,  dtype=torch.long),
        )


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 5e-5):
        self.patience    = patience
        self.min_delta   = min_delta
        self.best_score  = -np.inf
        self.counter     = 0
        self.should_stop = False
        self.best_state  = None

    def step(self, score: float, model: nn.Module):
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter    = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

    def restore_best(self, model: nn.Module):
        if self.best_state:
            model.load_state_dict(self.best_state)


def train_lightgcn(
    model,                    # LightGCN instance
    train_dataset: BPRDataset,
    val_fn: Callable,
    model_name: str = "LightGCN",
    epochs: int = 50,
    batch_size: int = 4096,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 10,
    device: Optional[torch.device] = None,
) -> nn.Module:
    if device is None:
        device = get_device()

    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=(device.type == "cuda"), drop_last=False,
    )

    stopper = EarlyStopping(patience=patience)

    print(f"\n{'='*60}")
    print(f"Training {model_name} on {device}")
    print(f"LR={lr} | Batch={batch_size} | Patience={patience} | Epochs={epochs}")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for users, pos_items, neg_items in loader:
            users     = users.to(device)
            pos_items = pos_items.to(device)
            neg_items = neg_items.to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = model(users, pos_items, neg_items)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        val_ndcg = val_fn(model)
        elapsed  = time.time() - t0

        print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Val NDCG@10: {val_ndcg:.4f} | {elapsed:.1f}s")

        stopper.step(val_ndcg, model)
        if stopper.should_stop:
            print(f"Early stopping at epoch {epoch}. Best NDCG@10: {stopper.best_score:.4f}")
            break

    stopper.restore_best(model)
    return model
