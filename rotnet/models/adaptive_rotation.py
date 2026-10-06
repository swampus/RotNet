from dataclasses import dataclass
import math

import torch
from torch import Tensor

from .rotation import RotationNet


@dataclass
class AdaptiveOutput:
    prediction: Tensor
    exit_stage: Tensor
    confidence: Tensor
    logits: Tensor


def validate_threshold(threshold: float) -> None:
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("confidence threshold must be finite and between 0 and 1")


class AdaptiveRotationNet(RotationNet):
    def __init__(self, hidden_dim: int = 256, threshold: float = 0.90) -> None:
        validate_threshold(threshold)
        super().__init__(hidden_dim)
        self.threshold = threshold

    def forward_all(self, x: Tensor) -> Tensor:
        """[batch, stage, class] logits; all stages remain in the gradient graph."""
        hidden = self.project(x)
        logits = []
        for stage in self.core.stages:
            hidden = stage(hidden)
            logits.append(self.classifier(hidden))
        return torch.stack(logits, dim=1)

    @torch.inference_mode()
    def infer(self, x: Tensor, threshold: float | None = None) -> AdaptiveOutput:
        """Really stop computation per sample by compacting the active batch.

        Call model.eval() first. Exit stages are 1-based; the last stage is
        unconditional. Returned logits/confidence are from the chosen exit,
        not from an unexecuted later stage.
        """
        if self.training:
            raise RuntimeError("call model.eval() before adaptive inference")
        threshold = self.threshold if threshold is None else threshold
        validate_threshold(threshold)
        hidden = self.project(x)
        batch_size = hidden.shape[0]
        active = torch.arange(batch_size, device=hidden.device)
        chosen_logits = hidden.new_empty((batch_size, self.classifier.out_features))
        exits = torch.empty(batch_size, dtype=torch.long, device=hidden.device)
        for stage_index, stage in enumerate(self.core.stages, start=1):
            if active.numel() == 0:
                break
            hidden = stage(hidden)
            logits = self.classifier(hidden)
            confidence = logits.softmax(dim=-1).amax(dim=-1)
            stop = confidence >= threshold
            if stage_index == len(self.core.stages):
                stop = torch.ones_like(stop)
            chosen_logits[active[stop]] = logits[stop]
            exits[active[stop]] = stage_index
            active = active[~stop]
            hidden = hidden[~stop]
        confidence, prediction = chosen_logits.softmax(dim=-1).max(dim=-1)
        return AdaptiveOutput(prediction, exits, confidence, chosen_logits)
