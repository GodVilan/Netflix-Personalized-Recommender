"""
trainer.py  —  FINAL (Fix-11, 2026-04-10)
Training loop for NCF and TwoTower.

Changes from Fix-11:
  patience  7  → 10   (models need more epochs to converge)
  min_delta 1e-4 → 5e-5 (finer early-stopping resolution)
  num_workers 2 → 4   (reduces data-loading bottleneck on A100)

All previous fixes retained:
  Fix-7: get_param_groups() for NCF (heavy embed L2)
  Fix-8/9: AMP, WarmupCosine, gradient accumulation
"""

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Callable
import mlflow
import mlflow.pytorch


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[Device] Using: {device} ", end="")
    if device.type == "cuda":
        print(f"({torch.cuda.get_device_name(0)})")
    elif device.type == "mps":
        print("(Apple Silicon GPU)")
    else:
        print("(CPU — no GPU found)")
    return device


class InteractionDataset(Dataset):
    """
    Positive interactions + sampled negatives for NCF.
    TwoTower ignores neg_items (uses in-batch negatives instead).
    """
    def __init__(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        n_items: int,
        n_neg: int = 64,
        seen_items: Optional[dict] = None,
        genre_features: Optional[np.ndarray] = None,
    ):
        self.users = user_ids
        self.items = item_ids
        self.n_items = n_items
        self.n_neg = n_neg
        self.seen = seen_items or {}
        self.genre_features = genre_features
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
        user     = int(self.users[idx])
        pos_item = int(self.items[idx])
        neg_items = self._sample_negatives(user)
        sample = {
            "user":      torch.tensor(user,      dtype=torch.long),
            "pos_item":  torch.tensor(pos_item,  dtype=torch.long),
            "neg_items": torch.tensor(neg_items, dtype=torch.long),
        }
        if self.genre_features is not None:
            sample["pos_genre"]  = torch.tensor(self.genre_features[pos_item],  dtype=torch.float32)
            sample["neg_genres"] = torch.tensor(self.genre_features[neg_items], dtype=torch.float32)
        return sample


class WarmupCosineScheduler:
    """Linear warmup + cosine decay."""
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, min_lr: float = 1e-6):
        self.optimizer     = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.min_lr        = min_lr
        self.base_lrs      = [pg["lr"] for pg in optimizer.param_groups]

    def step(self, epoch: int):
        if epoch < self.warmup_epochs:
            scale = (epoch + 1) / max(self.warmup_epochs, 1)
        else:
            progress = (epoch - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)
            scale    = 0.5 * (1.0 + np.cos(np.pi * progress))
            scale    = self.min_lr / self.base_lrs[0] + scale * (1.0 - self.min_lr / self.base_lrs[0])
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = base_lr * scale


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 5e-5):  # Fix-11
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


def train_neural_model(
    model: nn.Module,
    train_dataset: InteractionDataset,
    val_fn: Callable,
    model_name: str = "model",
    epochs: int = 30,
    batch_size: int = 2048,
    lr: float = 5e-4,
    weight_decay: float = 1e-5,
    warmup_epochs: int = 2,
    patience: int = 10,       # Fix-11: was 7
    accum_steps: int = 1,
    device: Optional[torch.device] = None,
    use_mlflow: bool = True,
) -> nn.Module:
    if device is None:
        device = get_device()
    else:
        device = torch.device(device)

    model = model.to(device)
    use_amp_cuda = (device.type == "cuda")
    use_amp_mps  = (device.type == "mps")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp_cuda)

    if hasattr(model, "get_param_groups"):
        param_groups = model.get_param_groups(lr=lr, embed_wd=1e-2, mlp_wd=weight_decay)
    else:
        param_groups = [{"params": model.parameters(), "lr": lr, "weight_decay": weight_decay}]

    optimizer = torch.optim.AdamW(param_groups)
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=warmup_epochs, total_epochs=epochs)

    loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=use_amp_cuda, drop_last=True,  # Fix-11: num_workers 2→4
    )

    stopper = EarlyStopping(patience=patience)
    eff_batch = batch_size * accum_steps

    if use_mlflow:
        mlflow.start_run(run_name=model_name)
        mlflow.log_params({
            "epochs": epochs, "batch_size": batch_size, "effective_batch": eff_batch,
            "lr": lr, "weight_decay": weight_decay, "n_neg": train_dataset.n_neg,
            "warmup_epochs": warmup_epochs, "device": device.type,
            "accum_steps": accum_steps,
        })

    print(f"\n{'='*60}")
    print(f"Training {model_name} on {device}")
    amp_desc = "CUDA GradScaler" if use_amp_cuda else "MPS float16" if use_amp_mps else "disabled"
    print(f"AMP: {amp_desc}")
    print(f"LR: {lr} | Warmup: {warmup_epochs} epochs | Patience: {patience}")
    print(f"Batch: {batch_size} × accum {accum_steps} = effective {eff_batch}")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        scheduler.step(epoch - 1)
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            if use_amp_cuda:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    loss = model(
                        batch["user"], batch["pos_item"], batch["neg_items"],
                        batch.get("pos_genre"), batch.get("neg_genres"),
                    )
                loss = loss / accum_steps
                scaler.scale(loss).backward()
            elif use_amp_mps:
                with torch.autocast(device_type="mps", dtype=torch.float16):
                    loss = model(
                        batch["user"], batch["pos_item"], batch["neg_items"],
                        batch.get("pos_genre"), batch.get("neg_genres"),
                    )
                (loss / accum_steps).backward()
            else:
                loss = model(
                    batch["user"], batch["pos_item"], batch["neg_items"],
                    batch.get("pos_genre"), batch.get("neg_genres"),
                )
                (loss / accum_steps).backward()

            epoch_loss += loss.item() * accum_steps

            if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
                if use_amp_cuda:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        current_lr = optimizer.param_groups[0]["lr"]
        avg_loss   = epoch_loss / len(loader)
        val_ndcg   = val_fn(model)
        elapsed    = time.time() - t0

        print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Val NDCG@10: {val_ndcg:.4f} | LR: {current_lr:.2e} | {elapsed:.1f}s")

        if use_mlflow:
            mlflow.log_metrics({"train_loss": avg_loss, "val_ndcg10": val_ndcg, "lr": current_lr}, step=epoch)

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
