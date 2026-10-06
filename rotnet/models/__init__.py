from .adaptive_rotation import AdaptiveOutput, AdaptiveRotationNet
from .dense import DenseNet
from .low_rank import LowRankNet
from .rotation import PairwiseRotation, RotationNet, RotationStage, pairing_indices

MODEL_NAMES = ("dense", "low-rank", "rotation", "adaptive-rotation")


def make_model(name: str, hidden_dim: int = 256, rank: int = 32, threshold: float = 0.90):
    if name == "dense":
        return DenseNet(hidden_dim)
    if name == "low-rank":
        return LowRankNet(hidden_dim, rank)
    if name == "rotation":
        return RotationNet(hidden_dim)
    if name == "adaptive-rotation":
        return AdaptiveRotationNet(hidden_dim, threshold)
    raise ValueError(f"unknown model: {name}")
