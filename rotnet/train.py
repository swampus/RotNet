import os
import random
import time

import numpy as np
import torch
from torch import nn

from .models import AdaptiveRotationNet


def seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def classification_loss(logits, targets):
    """Equal average CE at all stages, using no hard stopping in training."""
    if logits.ndim == 3:
        batch, stages, classes = logits.shape
        return nn.functional.cross_entropy(
            logits.reshape(batch * stages, classes),
            targets[:, None].expand(batch, stages).reshape(-1),
        )
    return nn.functional.cross_entropy(logits, targets)


def train_model(model, loader, epochs: int, learning_rate: float, weight_decay: float,
                device: torch.device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    history = []
    synchronize(device)
    total_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        synchronize(device)
        start = time.perf_counter()
        loss_sum, correct, seen = 0.0, 0, 0
        for images, targets, _ in loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model.forward_all(images) if isinstance(model, AdaptiveRotationNet) else model(images)
            loss = classification_loss(logits, targets)
            loss.backward()
            optimizer.step()
            final_logits = logits[:, -1] if logits.ndim == 3 else logits
            loss_sum += loss.detach().item() * len(targets)
            correct += (final_logits.argmax(dim=-1) == targets).sum().item()
            seen += len(targets)
        synchronize(device)
        row = {"epoch": epoch, "loss": loss_sum / seen, "final_stage_train_accuracy": correct / seen,
               "wall_seconds": time.perf_counter() - start}
        history.append(row)
        print(f"  epoch {epoch}/{epochs}: loss={row['loss']:.4f}, "
              f"train_acc={row['final_stage_train_accuracy']:.4f}, seconds={row['wall_seconds']:.2f}", flush=True)
    synchronize(device)
    return {"training_seconds": time.perf_counter() - total_start, "history": history}
