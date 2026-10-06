"""Validation selection and stage transitions: test labels never choose policy."""

import math

import torch


@torch.inference_mode()
def collect_all_stages(model, loader, device):
    model.eval()
    logits, targets, ids = [], [], []
    for images, target, sample_ids in loader:
        logits.append(model.forward_all(images.to(device)).cpu())
        targets.append(target)
        ids.append(sample_ids)
    return torch.cat(logits), torch.cat(targets), torch.cat(ids)


def exits_from_logits(logits, threshold):
    if logits.ndim != 3 or logits.shape[0] == 0 or not 0 <= threshold <= 1:
        raise ValueError("expected nonempty [batch,stage,class] logits and threshold in [0,1]")
    confidence = logits.softmax(dim=-1).amax(dim=-1)
    eligible = confidence >= threshold
    eligible[:, -1] = True
    exits = eligible.int().argmax(dim=-1)
    chosen = logits[torch.arange(len(logits)), exits]
    return chosen, exits + 1


def select_policy(validation_logits, validation_targets, thresholds, tolerance_pp=0.1):
    """Minimize validation mean stages subject to a preset accuracy constraint.

    Full-depth/no-gating is an explicit candidate and guaranteed fallback.
    Tie-break by higher validation accuracy then lower threshold. Uses only
    validation logits/labels; there is no test data argument.
    """
    if not math.isfinite(tolerance_pp) or tolerance_pp < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    full = (validation_logits[:, -1].argmax(-1) == validation_targets).double().mean().item()
    candidates = []
    for threshold in sorted(set(thresholds)):
        chosen, exits = exits_from_logits(validation_logits, threshold)
        candidates.append({"policy": "threshold", "threshold": threshold,
                           "validation_accuracy": (chosen.argmax(-1) == validation_targets).double().mean().item(),
                           "validation_mean_stages": exits.double().mean().item()})
    candidates.append({"policy": "full-depth", "threshold": None,
                       "validation_accuracy": full, "validation_mean_stages": validation_logits.shape[1]})
    feasible = [c for c in candidates if c["validation_accuracy"] + 1e-12 >= full - tolerance_pp / 100]
    selected = min(feasible, key=lambda c: (c["validation_mean_stages"], -c["validation_accuracy"],
                                          c["threshold"] if c["threshold"] is not None else float("inf")))
    return {"selected": selected, "candidates": candidates, "full_depth_validation_accuracy": full,
            "tolerance_percentage_points": tolerance_pp,
            "objective": "minimum validation mean stages subject to accuracy >= full depth - tolerance",
            "selection_split": "validation", "threshold_frozen_before_test": True}


def stage_transitions(logits, targets):
    predictions = logits.argmax(-1)
    correct = predictions == targets[:, None]
    stages = logits.shape[1]
    transitions = []
    for index in range(stages - 1):
        before, after = correct[:, index], correct[:, index + 1]
        transitions.append({"from_stage": index + 1, "to_stage": index + 2,
                            "rescued": int((~before & after).sum()),
                            "damaged": int((before & ~after).sum()),
                            "incorrect_to_incorrect": int((~before & ~after).sum()),
                            "correct_to_correct": int((before & after).sum()),
                            "changed_prediction": int((predictions[:, index] != predictions[:, index + 1]).sum())})
    return {"samples": len(targets), "accuracy_by_stage": correct.double().mean(0).tolist(),
            "stage1_correct": int(correct[:, 0].sum()), "stage1_errors": int((~correct[:, 0]).sum()),
            "stage1_errors_correct_at_final": int((~correct[:, 0] & correct[:, -1]).sum()),
            "stage1_errors_ever_corrected": int((~correct[:, 0] & correct[:, 1:].any(-1)).sum()),
            "stage1_correct_wrong_at_final": int((correct[:, 0] & ~correct[:, -1]).sum()),
            "stage1_correct_ever_broken": int((correct[:, 0] & ~correct[:, 1:].all(-1)).sum()),
            "transitions": transitions}
