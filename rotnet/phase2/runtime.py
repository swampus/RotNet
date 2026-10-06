"""Inference-only cached trigonometry; never modifies a trained original."""

import copy

import torch
from torch import nn

from rotnet.models import AdaptiveOutput, AdaptiveRotationNet, PairwiseRotation
from rotnet.models.adaptive_rotation import validate_threshold


class CachedRotation(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.n, self.stride = original.n, original.stride
        shape = (self.n // (2 * self.stride), self.stride)
        self.register_buffer("cos", original.theta.detach().cos().reshape(shape).clone())
        self.register_buffer("sin", original.theta.detach().sin().reshape(shape).clone())

    def forward(self, x):
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("cached rotation is inference-only; use eval() and inference_mode/no_grad")
        blocks = x.reshape(*x.shape[:-1], self.n // (2 * self.stride), 2, self.stride)
        a, b = blocks[..., 0, :], blocks[..., 1, :]
        return torch.stack((a * self.cos - b * self.sin, a * self.sin + b * self.cos), dim=-2).reshape_as(x)


class CachedAdaptive(AdaptiveRotationNet):
    @torch.inference_mode()
    def infer(self, x, threshold=None):
        if self.training:
            raise RuntimeError("call eval() before inference")
        threshold = self.threshold if threshold is None else threshold
        validate_threshold(threshold)
        hidden = self.project(x)
        active = torch.arange(len(hidden), device=hidden.device)
        chosen = hidden.new_empty((len(hidden), self.classifier.out_features))
        confidences = hidden.new_empty(len(hidden))
        predictions = torch.empty(len(hidden), dtype=torch.long, device=hidden.device)
        exits = torch.empty_like(predictions)
        for index, stage in enumerate(self.core.stages, start=1):
            if active.numel() == 0:
                break
            hidden = stage(hidden)
            logits = self.classifier(hidden)
            confidence, prediction = logits.softmax(-1).max(-1)
            stop = confidence >= threshold
            if index == len(self.core.stages):
                stop = torch.ones_like(stop)
            indices = active[stop]
            chosen[indices] = logits[stop]
            confidences[indices] = confidence[stop]
            predictions[indices] = prediction[stop]
            exits[indices] = index
            active, hidden = active[~stop], hidden[~stop]
        # Reuse the already computed exit probabilities instead of applying
        # softmax to the selected logits a second time.
        return AdaptiveOutput(predictions, exits, confidences, chosen)


def optimize_inference(model):
    optimized = copy.deepcopy(model)
    for stage in optimized.core.stages:
        if isinstance(stage.rotation, PairwiseRotation):
            stage.rotation = CachedRotation(stage.rotation)
    if isinstance(optimized, AdaptiveRotationNet):
        optimized.__class__ = CachedAdaptive
    optimized.requires_grad_(False)
    return optimized.eval()


@torch.inference_mode()
def verify_equivalence(original, optimized, loader, device, threshold=0.9, atol=1e-6, rtol=1e-5):
    original.eval()
    optimized.eval()
    max_error, checked = 0.0, 0
    adaptive = isinstance(original, AdaptiveRotationNet)
    for images, _, _ in loader:
        images = images.to(device)
        a = original.forward_all(images) if adaptive else original(images)
        b = optimized.forward_all(images) if adaptive else optimized(images)
        torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
        max_error = max(max_error, (a - b).abs().max().item())
        if adaptive:
            a_exit, b_exit = original.infer(images, threshold), optimized.infer(images, threshold)
            torch.testing.assert_close(a_exit.logits, b_exit.logits, atol=atol, rtol=rtol)
            torch.testing.assert_close(a_exit.confidence, b_exit.confidence, atol=atol, rtol=rtol)
            assert torch.equal(a_exit.prediction, b_exit.prediction), "optimized predictions differ"
            assert torch.equal(a_exit.exit_stage, b_exit.exit_stage), "optimized exits differ"
        else:
            assert torch.equal(a.argmax(-1), b.argmax(-1)), "optimized predictions differ"
        checked += len(images)
    return {"status": "passed", "samples_checked": checked, "max_absolute_logit_error": max_error,
            "atol": atol, "rtol": rtol, "exit_stages_and_predictions_match_exactly": True,
            "optimization": "cached sin/cos; adaptive reuses chosen confidence/prediction"}


@torch.inference_mode()
def profile_calls(model, images, adaptive=False):
    model.eval()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
        for _ in range(10):
            if adaptive:
                model.infer(images)
            else:
                model(images).softmax(-1).max(-1)
    rows = sorted(profiler.key_averages(), key=lambda e: e.self_cpu_time_total, reverse=True)
    return [{"operator": e.key, "calls": e.count, "self_cpu_microseconds": e.self_cpu_time_total}
            for e in rows]
