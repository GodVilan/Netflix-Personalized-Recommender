"""
trainer.py
Training loop for PyTorch-based recommendation models (NCF & TwoTower).
Features:
  - Negative sampling (uniform + popularity-biased)
  - Early stopping based on validation NDCG@10
  - Learning rate scheduling (cosine annealing with warm restarts)
  - Mixed precision training (torch.amp)
  - MLflow experiment tracking
"""

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from typing import Optional
import mlflow
import mlflow.pytorch


class InteractionDataset(Dataset):
    """
    Dataset for NCF / Two-Tower training.
    For each positive (user, item) pair, samples `n_neg` negatives uniformly.
    """

    def __init__(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        n_items: int,
        n_neg: int = 4,
        seen_items: Optional[dict] = None,
        genre_features: Optional[np.ndarray] = None,
    ):
        self.users = user_ids
        self.items = item_ids
        self.n_items = n_items
        self.n_neg = n_neg
        self.seen = seen_items or {}
        self.genre_features = genre_features  # (n_items, n_genres)
        self.rng = np.random.default_rng(42)

    def __len__(self):
        return len(self.users)

    def _sample_negatives(self, user: int) -> np.ndarray:
        seen = self.seen.get(user, set())
        negs = []
        while len(negs) < self.n_neg:
            candidates = self.rng.integers(0, self.n_items, size=self.n_neg * 3)
            for c in candidates:
                if c not in seen:
                    negs.append(c)
                if len(negs) == self.n_neg:
                    break
        return np.array(negs[:self.n_neg])

    def __getitem__(self, idx):
        user = int(self.users[idx])
        pos_item = int(self.items[idx])
        neg_items = self._sample_negatives(user)

        sample = {
            "user": torch.tensor(user, dtype=torch.long),
            "pos_item": torch.tensor(pos_item, dtype=torch.long),
            "neg_items": torch.tensor(neg_items, dtype=torch.long),
        }
        if self.genre_features is not None:
            sample["pos_genre"] = torch.tensor(self.genre_features[pos_item], dtype=torch.float32)
            sample["neg_genres"] = torch.tensor(self.genre_features[neg_items], dtype=torch.float32)
        return sample


class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = -np.inf
        self.counter = 0
        self.should_stop = False
        self.best_state = None

    def step(self, score: float, model: nn.Module):
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

    def restore_best(self, model: nn.Module):
        if self.best_state:
            model.load_state_dict(self.best_state)


def train_neural_model(
    model: nn.Module,
    train_dataset: InteractionDataset,
    val_fn,                        # callable(model) -> float (e.g., NDCG@10)
    model_name: str = "model",
    epochs: int = 20,
    batch_size: int = 2048,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    use_mlflow: bool = True,
) -> nn.Module:
    device = torch.device(device)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda")
    )

    stopper = EarlyStopping(patience=patience)

    if use_mlflow:
        mlflow.start_run(run_name=model_name)
        mlflow.log_params({
            "epochs": epochs, "batch_size": batch_size,
            "lr": lr, "weight_decay": weight_decay,
            "n_neg": train_dataset.n_neg,
        })

    print(f"\n{'='*60}")
    print(f"Training {model_name} on {device}")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                loss = model(
                    batch["user"],
                    batch["pos_item"],
                    batch["neg_items"],
                    batch.get("pos_genre"),
                    batch.get("neg_genres"),
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(loader)
        val_ndcg = val_fn(model)
        elapsed = time.time() - t0

        print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Val NDCG@10: {val_ndcg:.4f} | {elapsed:.1f}s")

        if use_mlflow:
            mlflow.log_metrics({"train_loss": avg_loss, "val_ndcg10": val_ndcg}, step=epoch)

        stopper.step(val_ndcg, model)
        if stopper.should_stop:
            print(f"Early stopping at epoch {epoch}. Best NDCG@10: {stopper.best_score:.4f}")
            break

    stopper.restore_best(model)

    if use_mlflow:
        mlflow.log_metric("best_val_ndcg10", stopper.best_score)
        mlflow.pytorch.log_model(model, "model")
        mlflow.end_run()

    return model
