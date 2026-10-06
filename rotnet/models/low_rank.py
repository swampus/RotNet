from torch import Tensor, nn

from .common import ProjectedClassifier


class LowRankNet(ProjectedClassifier):
    def __init__(self, hidden_dim: int = 256, rank: int = 32) -> None:
        if not 1 <= rank <= hidden_dim:
            raise ValueError("rank must be between 1 and hidden_dim")
        super().__init__(hidden_dim)
        self.rank = rank
        self.core = nn.Sequential(
            nn.Linear(hidden_dim, rank, bias=False),
            nn.Linear(rank, hidden_dim),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.core(self.project(x)))
