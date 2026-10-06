"""Explicit parameter and arithmetic accounting, independent of latency."""

from .models import AdaptiveRotationNet, DenseNet, LowRankNet


def parameter_counts(model) -> dict:
    return {
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "core_trainable_parameters": sum(p.numel() for p in model.core.parameters() if p.requires_grad),
        "core_total_parameters": sum(p.numel() for p in model.core.parameters()),
    }


def operation_counts(model, average_stages: float | None = None) -> dict:
    """Scalar multiply/add count per sample, including affine biases.

    A dot product has k multiplies and k-1 adds; a bias adds one more.
    GELU, sin/cos, softmax, confidence decisions, memory and indexing are
    explicitly excluded, so this is not a complete FLOP or speed estimate.
    """
    n = model.hidden_dim
    projection_ops = 2 * 784 * n
    classifier_ops = 2 * n * 10
    if isinstance(model, DenseNet):
        core_ops, gelu_values, heads = 2 * n * n, n, 1
        extra = {}
    elif isinstance(model, LowRankNet):
        core_ops = model.rank * (2 * n - 1) + 2 * model.rank * n
        gelu_values, heads, extra = n, 1, {}
    else:
        full_stages = len(model.core.stages)
        stages = full_stages if average_stages is None else average_stages
        core_ops, gelu_values = 5 * n * stages, n * stages
        heads = stages if isinstance(model, AdaptiveRotationNet) else 1
        extra = {
            "full_depth_core_multiply_add_ops": 5 * n * full_stages,
            "rotation_only_ops_per_stage": 3 * n,
            "scale_bias_ops_per_stage": 2 * n,
            "theta_sin_cos_evaluations_per_executed_stage_per_batch": n,
            "average_executed_stages": stages,
        }
    return {
        "core_multiply_add_ops_per_sample": core_ops,
        "projection_multiply_add_ops_per_sample": projection_ops,
        "classifier_multiply_add_ops_per_sample": classifier_ops * heads,
        "whole_model_multiply_add_ops_per_sample": projection_ops + core_ops + classifier_ops * heads,
        "core_gelu_coordinates_per_sample": gelu_values,
        "excluded": ["GELU", "sin/cos", "softmax", "threshold checks", "indexing", "memory traffic"],
        **extra,
    }
