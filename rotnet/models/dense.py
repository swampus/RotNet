from torch import Tensor, nn

from .common import ProjectedClassifier


class DenseNet(ProjectedClassifier):
    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__(hidden_dim)
        self.core = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU())

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.core(self.project(x)))
