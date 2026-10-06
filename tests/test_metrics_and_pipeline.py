import json

import pytest
import torch

from rotnet.data import load_data, make_loader
from rotnet.evaluate import adaptive_statistics, benchmark_inference, evaluate_model
from rotnet.metrics import operation_counts, parameter_counts
from rotnet.models import make_model
from rotnet.plots import create_plots
from rotnet.reporting import write_predictions
from rotnet.train import seed_everything, train_model


@pytest.mark.parametrize("name,core,total,ops", [
    ("dense", 65792, 269322, 131072),
    ("low-rank", 16640, 220170, 32736),
    ("rotation", 5120, 208650, 10240),
    ("adaptive-rotation", 5120, 208650, 10240),
])
def test_default_parameter_and_operation_counts(name, core, total, ops):
    model = make_model(name)
    counts = parameter_counts(model)
    assert counts["core_trainable_parameters"] == core
    assert counts["total_parameters"] == counts["trainable_parameters"] == total
    assert operation_counts(model)["core_multiply_add_ops_per_sample"] == ops


def test_adaptive_arithmetic_accounts_for_repeated_classifier():
    model = make_model("adaptive-rotation")
    counts = operation_counts(model, average_stages=2.5)
    assert counts["core_multiply_add_ops_per_sample"] == 3200
    assert counts["classifier_multiply_add_ops_per_sample"] == 12800
    assert counts["whole_model_multiply_add_ops_per_sample"] == 417408


def test_shared_initialization_is_comparable():
    models = []
    for name in ("dense", "low-rank", "rotation", "adaptive-rotation"):
        seed_everything(42)
        models.append(make_model(name))
    for model in models[1:]:
        torch.testing.assert_close(model.projection.weight, models[0].projection.weight)
        torch.testing.assert_close(model.classifier.weight, models[0].classifier.weight)
    for a, b in zip(models[2].core.parameters(), models[3].core.parameters()):
        torch.testing.assert_close(a, b)


def test_empty_difficulty_group_is_none():
    rows = [{"exit_stage": 1, "correct": True, "confidence": 0.95},
            {"exit_stage": 8, "correct": True, "confidence": 0.6}]
    result = adaptive_statistics(rows, 8)
    assert result["average_stages"] == 4.5
    assert result["average_stages_incorrect"] is None
    assert result["confidence_by_exit_stage"][1]["mean_exit_confidence"] is None
    assert sum(result["exit_distribution"].values()) == 2
    json.dumps(result, allow_nan=False)


def test_mocked_dataset_smoke_pipeline_and_plot_exports(monkeypatch, tmp_path):
    # Synthetic data is only a test fixture; no network or research accuracy claims.
    class FakeDataset:
        def __init__(self, root, train, download):
            self.data = torch.arange(16 * 28 * 28).reshape(16, 28, 28).remainder(256).byte()
            self.targets = torch.arange(16).remainder(10)

        def __len__(self):
            return 16

    monkeypatch.setattr("rotnet.data.MNIST", FakeDataset)
    tensors, splits = load_data("mnist", tmp_path, 42, train_limit=12, test_limit=8)
    other_tensors, other_splits = load_data("mnist", tmp_path, 42, train_limit=12, test_limit=8)
    assert splits == other_splits
    assert tensors["train"].tensors[0].min() >= -1
    assert tensors["train"].tensors[0].max() <= 1
    torch.testing.assert_close(tensors["test"].tensors[0], other_tensors["test"].tensors[0])
    models, predictions = {}, []
    device = torch.device("cpu")
    for name in ("dense", "low-rank", "rotation", "adaptive-rotation"):
        seed_everything(42)
        model = make_model(name, hidden_dim=8, rank=2)
        training = train_model(model, make_loader(tensors["train"], 4, 42, True), 1, 0.001, 0, device)
        loader = make_loader(tensors["test"], 4, 42, False)
        result, rows = evaluate_model(model, loader, device, threshold=0.9 if name == "adaptive-rotation" else None)
        predictions.extend({"model": name, **row} for row in rows)
        result["parameters"] = parameter_counts(model)
        result["operations"] = operation_counts(model)
        result.update(training)
        assert len(rows) == 8
        assert 0 <= result["test_accuracy"] <= 1
        assert all(len([key for key in row if key.startswith("logit_")]) == 10 for row in rows)
        timing = benchmark_inference(model, tensors["test"], device, 4, repeats=1, warmup=1,
                                     threshold=0.9 if name == "adaptive-rotation" else None)
        assert timing["median_wall_seconds"] > 0
        assert timing["samples"] == 8
        if name == "adaptive-rotation":
            result["threshold_sweep"] = [{**result, "inference": {"batched": timing, "single_sample": timing}}]
            result["full_depth"], _ = evaluate_model(model, loader, device, full_depth=True)
            assert sum(result["exit_distribution"].values()) == 8
        models[name] = result
    write_predictions(tmp_path / "predictions.csv", predictions)
    assert "logit_9" in (tmp_path / "predictions.csv").read_text()
    metrics = {"models": models, "config": {"threshold": 0.9, "hidden_dim": 8}}
    plots = create_plots(metrics, predictions, tmp_path / "plots")
    assert len(plots) == 5
    assert all((tmp_path / "plots" / name).stat().st_size > 100 for name in plots)
