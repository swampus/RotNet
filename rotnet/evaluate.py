"""Predict with true adaptive execution; benchmark model calls separately."""

import statistics
import time

import torch

from .models import AdaptiveRotationNet
from .train import synchronize


@torch.inference_mode()
def evaluate_model(model, loader, device, threshold: float | None = None, full_depth: bool = False):
    model.eval()
    if isinstance(model, AdaptiveRotationNet) and not full_depth and threshold is None:
        threshold = model.threshold
    rows, correct, count = [], 0, 0
    stage_count = len(model.core.stages) if hasattr(model.core, "stages") else None
    synchronize(device)
    start = time.perf_counter()
    for images, targets, sample_ids in loader:
        images = images.to(device)
        if isinstance(model, AdaptiveRotationNet) and not full_depth:
            result = model.infer(images, threshold)
            logits, confidence, prediction, exits = (result.logits.cpu(), result.confidence.cpu(),
                                                     result.prediction.cpu(), result.exit_stage.cpu())
        else:
            logits = model(images).cpu()
            confidence, prediction = logits.softmax(dim=-1).max(dim=-1)
            exits = torch.full_like(targets, stage_count) if stage_count is not None else None
        for i in range(len(targets)):
            is_correct = int(prediction[i]) == int(targets[i])
            row = {"sample_id": int(sample_ids[i]), "target": int(targets[i]),
                   "prediction": int(prediction[i]), "correct": is_correct,
                   "confidence": float(confidence[i]),
                   "exit_stage": int(exits[i]) if exits is not None else None,
                   "threshold": threshold,
                   "policy": "full-depth" if full_depth else "adaptive" if threshold is not None else "fixed"}
            row.update({f"logit_{j}": float(logits[i, j]) for j in range(logits.shape[-1])})
            rows.append(row)
            correct += is_correct
            count += 1
    synchronize(device)
    metrics = {"test_accuracy": correct / count, "test_samples": count,
               "evaluation_wall_seconds": time.perf_counter() - start}
    if isinstance(model, AdaptiveRotationNet) and not full_depth:
        metrics.update(adaptive_statistics(rows, stage_count))
        metrics["threshold"] = threshold
    return metrics, rows


def adaptive_statistics(rows: list[dict], stages: int) -> dict:
    used = [r["exit_stage"] for r in rows]
    correct = [r["exit_stage"] for r in rows if r["correct"]]
    incorrect = [r["exit_stage"] for r in rows if not r["correct"]]
    confidence_by_stage = []
    for stage in range(1, stages + 1):
        selected = [r for r in rows if r["exit_stage"] == stage]
        confidence_by_stage.append({
            "stage": stage, "samples": len(selected),
            "mean_exit_confidence": statistics.mean(r["confidence"] for r in selected) if selected else None,
            "exit_accuracy": statistics.mean(r["correct"] for r in selected) if selected else None,
        })
    return {
        "average_stages": statistics.mean(used), "median_stages": statistics.median(used),
        "exit_distribution": {str(s): used.count(s) for s in range(1, stages + 1)},
        "correct_samples": len(correct), "incorrect_samples": len(incorrect),
        "average_stages_correct": statistics.mean(correct) if correct else None,
        "average_stages_incorrect": statistics.mean(incorrect) if incorrect else None,
        "confidence_by_exit_stage": confidence_by_stage,
    }


@torch.inference_mode()
def benchmark_inference(model, dataset, device, batch_size: int, repeats: int = 3,
                        warmup: int = 2, threshold: float | None = None,
                        max_samples: int | None = None, full_depth: bool = False) -> dict:
    """Preload outside timing. Same ordered test prefix for all models/thresholds.

    Report the median of complete passes; include output formation and true
    adaptive compaction but exclude data loading, transfers, metrics and CSV.
    Batch=1 and configured batch size are separate experiments.
    """
    model.eval()
    sample_count = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    images = dataset.tensors[0][:sample_count].to(device)

    def predict(batch):
        if isinstance(model, AdaptiveRotationNet) and not full_depth:
            return model.infer(batch, threshold)
        logits = model(batch)
        confidence, prediction = logits.softmax(dim=-1).max(dim=-1)
        return logits, confidence, prediction

    for _ in range(warmup):
        predict(images[:batch_size])
    synchronize(device)
    pass_seconds = []
    for _ in range(repeats):
        synchronize(device)
        start = time.perf_counter()
        for offset in range(0, sample_count, batch_size):
            predict(images[offset:offset + batch_size])
        synchronize(device)
        pass_seconds.append(time.perf_counter() - start)
    median = statistics.median(pass_seconds)
    timed_mean_stages = None
    if isinstance(model, AdaptiveRotationNet) and not full_depth:
        stage_sum = 0
        for offset in range(0, sample_count, batch_size):
            stage_sum += predict(images[offset:offset + batch_size]).exit_stage.sum().item()
        timed_mean_stages = stage_sum / sample_count
    return {"batch_size": batch_size, "samples": sample_count, "repeats": repeats,
            "warmup_batches": warmup, "pass_seconds": pass_seconds,
            "median_wall_seconds": median, "median_ms_per_sample": 1000 * median / sample_count,
            "samples_per_second": sample_count / median, "threshold": threshold,
            "execution": "final-head full depth" if full_depth else "normal inference",
            "adaptive_mean_stages_on_timed_samples": timed_mean_stages,
            "sample_selection": "ordered test split prefix; IDs recorded in splits.json",
            "includes": "model calls and final softmax/predictions; adaptive decisions, compaction and exit outputs",
            "excludes": "data loading, host/device transfers, prediction export, metric aggregation"}
