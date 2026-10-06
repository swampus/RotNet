"""Butterfly pairing and vectorized O(n) pairwise rotations; no dense matrix."""

import math

import torch
from torch import Tensor, nn

from .common import ProjectedClassifier


def butterfly_strides(n: int) -> list[int]:
    if n < 2 or n & (n - 1):
        raise ValueError("hidden dimension must be a power of two >= 2")
    return [1 << i for i in range(n.bit_length() - 1)]


def pairing_indices(n: int, stride: int) -> tuple[Tensor, Tensor]:
    """In each block of 2*stride, pair offset j with offset j+stride.

    Equivalently, pair i with i XOR stride, taking i whose stride bit is 0.
    Every coordinate occurs exactly once. Pair order matches the reshape in
    PairwiseRotation.forward, so each theta has a stable, documented pair.
    """
    if stride not in butterfly_strides(n):
        raise ValueError("stride must be a power of two smaller than n")
    left = torch.tensor(
        [base + j for base in range(0, n, 2 * stride) for j in range(stride)],
        dtype=torch.long,
    )
    return left, left + stride


class PairwiseRotation(nn.Module):
    """Rotate the last axis, supporting arbitrary leading batch dimensions."""

    def __init__(self, n: int, stride: int) -> None:
        super().__init__()
        if stride not in butterfly_strides(n):
            raise ValueError("invalid butterfly stride")
        self.n = n
        self.stride = stride
        self.theta = nn.Parameter(torch.empty(n // 2))
        nn.init.uniform_(self.theta, -math.pi / 4, math.pi / 4)

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.n:
            raise ValueError(f"expected last dimension {self.n}")
        blocks = x.reshape(*x.shape[:-1], self.n // (2 * self.stride), 2, self.stride)
        a, b = blocks[..., 0, :], blocks[..., 1, :]
        shape = (self.n // (2 * self.stride), self.stride)
        cos = self.theta.cos().reshape(shape)
        sin = self.theta.sin().reshape(shape)
        rotated = torch.stack((a * cos - b * sin, a * sin + b * cos), dim=-2)
        return rotated.reshape_as(x)


class RotationStage(nn.Module):
    def __init__(self, n: int, stride: int) -> None:
        super().__init__()
        self.rotation = PairwiseRotation(n, stride)
        self.scale = nn.Parameter(torch.ones(n))
        self.bias = nn.Parameter(torch.zeros(n))
        self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.rotation(x) * self.scale + self.bias)


class RotationCore(nn.Module):
    def __init__(self, n: int = 256) -> None:
        super().__init__()
        self.strides = butterfly_strides(n)
        self.stages = nn.ModuleList([RotationStage(n, stride) for stride in self.strides])

    def forward(self, x: Tensor) -> Tensor:
        for stage in self.stages:
            x = stage(x)
        return x


class RotationNet(ProjectedClassifier):
    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__(hidden_dim)
        self.core = RotationCore(hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.core(self.project(x)))
