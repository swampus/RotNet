"""Official torchvision splits; no augmentation or test-driven model selection."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import FashionMNIST, MNIST


def load_data(dataset: str, root: Path, seed: int, train_limit: int | None = None,
              test_limit: int | None = None, download: bool = True):
    dataset_class = {"mnist": MNIST, "fashion-mnist": FashionMNIST}[dataset]
    tensors, splits = {}, {}
    for split, is_train, limit in (("train", True, train_limit), ("test", False, test_limit)):
        raw = dataset_class(str(root), train=is_train, download=download)
        if limit is not None and not 1 <= limit <= len(raw):
            raise ValueError(f"{split}-limit must be between 1 and {len(raw)}")
        ids = torch.arange(len(raw))
        if limit is not None:
            generator = torch.Generator().manual_seed(seed + (0 if is_train else 1))
            ids = torch.randperm(len(raw), generator=generator)[:limit].sort().values
        # Materialize once so PIL conversion is not repeated during every epoch.
        images = raw.data[ids].unsqueeze(1).to(torch.float32).div_(255).sub_(0.5).div_(0.5)
        targets = raw.targets[ids].to(torch.long)
        tensors[split] = TensorDataset(images, targets, ids)
        splits[split] = {"size": len(ids), "original_size": len(raw), "sample_ids": ids.tolist()}
    return tensors, splits


def make_loader(dataset: TensorDataset, batch_size: int, seed: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      generator=torch.Generator().manual_seed(seed), drop_last=False)
