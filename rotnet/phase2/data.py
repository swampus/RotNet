"""Separate validation partition and deterministic 16x16 representation."""

import hashlib

import torch
from torch.nn import functional as F
from torch.utils.data import TensorDataset


def resize_direct(dataset):
    images, targets, ids = dataset.tensors
    # Fixed bilinear interpolation with antialiasing, applied to normalized
    # float tensors. Same preprocessing across models and datasets.
    resized = F.interpolate(images, size=(16, 16), mode="bilinear", align_corners=False, antialias=True)
    return TensorDataset(resized, targets, ids)


def validation_split(dataset, size=5000, split_seed=202602):
    if not 0 < size < len(dataset):
        raise ValueError("validation size must be smaller than the training dataset and positive")
    order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(split_seed))
    validation_ids, train_ids = order[:size].sort().values, order[size:].sort().values

    def select(indices):
        return TensorDataset(*(t[indices] for t in dataset.tensors))

    return select(train_ids), select(validation_ids)


def split_record(dataset):
    ids = dataset.tensors[2].tolist()
    return {"size": len(dataset), "sample_ids": ids,
            "sample_ids_sha256": hashlib.sha256(dataset.tensors[2].numpy().tobytes()).hexdigest()}
