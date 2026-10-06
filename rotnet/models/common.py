"""Shared projection and classifier, constructed before each model's core."""

from torch import Tensor, nn


class ProjectedClassifier(nn.Module):
    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.projection = nn.Linear(784, hidden_dim)
        self.activation = nn.GELU()
        self.classifier = nn.Linear(hidden_dim, 10)

    def project(self, x: Tensor) -> Tensor:
        return self.activation(self.projection(x.flatten(start_dim=1)))
