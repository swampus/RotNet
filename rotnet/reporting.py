"""Standard-library reporting, including honest pending/failed run reports."""

import csv
import json
from pathlib import Path


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_predictions(path: Path, rows: list[dict]) -> None:
    fields = ["model", "sample_id", "target", "prediction", "correct", "confidence",
              "exit_stage", "threshold", "policy"] + [f"logit_{i}" for i in range(10)]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _number(value, digits=3):
    return "n/a (empty group)" if value is None else f"{value:.{digits}f}"


def render_report(metrics: dict) -> str:
    config, models = metrics["config"], metrics.get("models", {})
    lines = ["# RotNet measured results", "", f"Run: `{metrics['run_name']}`. Status: **{metrics['status']}**.", ""]
    if metrics["status"] != "completed":
        lines += ["Runtime results are pending. This run did not complete; missing values are not zeroes.",
                  "", f"Reason: {metrics.get('error', 'preflight only; no training requested')}", ""]
    lines += [f"Dataset: `{config['dataset']}`; seed: {config['seed']}; requested epochs: {config['epochs']}; "
              f"device: `{config['device']}`; CPU threads: {config['threads']}.", ""]
    if "data" in metrics:
        lines += [f"Executed data sizes: {metrics['data']['train']['size']} training and "
                  f"{metrics['data']['test']['size']} test samples.", ""]
    if models:
        lines += ["| Model | Test accuracy | Trainable / total params | Core params | Core multiply/add ops | Train s | Batched ms/sample | Batch=1 ms/sample |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for name, model in models.items():
            params, timing = model["parameters"], model["inference"]
            lines.append(f"| {name} | {100 * model['test_accuracy']:.2f}% | "
                         f"{params['trainable_parameters']} / {params['total_parameters']} | "
                         f"{params['core_trainable_parameters']} | "
                         f"{model['operations']['core_multiply_add_ops_per_sample']:.1f} | "
                         f"{model['training_seconds']:.2f} | {timing['batched']['median_ms_per_sample']:.4f} | "
                         f"{timing['single_sample']['median_ms_per_sample']:.4f} |")
        lines += ["", "Adaptive accuracy and core operations above use the configured default threshold. "
                  "Timings are medians of repeated model-only passes; evaluation wall time is recorded separately. "
                  "See metrics.json for batch sizes, sample counts, individual passes and exclusions.", ""]
        dense, rotation = models.get("dense"), models.get("rotation")
        if dense and rotation:
            difference = 100 * (rotation["test_accuracy"] - dense["test_accuracy"])
            ratio = rotation["inference"]["batched"]["median_ms_per_sample"] / dense["inference"]["batched"]["median_ms_per_sample"]
            lines += [f"Rotation accuracy differs from dense by {difference:+.2f} percentage points. "
                      f"Its batched inference takes {ratio:.2f} times the dense baseline time.", ""]
            if difference < 0:
                lines += ["RotationNet lost accuracy in this run.", ""]
            if ratio > 1:
                lines += ["**RotationNet was slower in wall-clock batched inference despite its lower theoretical core arithmetic count.**", ""]
    adaptive = models.get("adaptive-rotation")
    if adaptive:
        lines += ["## Adaptive threshold sweep", "",
                  "| Threshold | Accuracy | Mean stages | Median stages | Batched ms/sample | Batch=1 ms/sample | Mean stages, correct | Mean stages, incorrect |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for sweep in adaptive["threshold_sweep"]:
            lines.append(f"| {sweep['threshold']:.2f} | {100 * sweep['test_accuracy']:.2f}% | "
                         f"{sweep['average_stages']:.3f} | {sweep['median_stages']:.1f} | "
                         f"{sweep['inference']['batched']['median_ms_per_sample']:.4f} | "
                         f"{sweep['inference']['single_sample']['median_ms_per_sample']:.4f} | "
                         f"{_number(sweep['average_stages_correct'])} | {_number(sweep['average_stages_incorrect'])} |")
        full = adaptive["full_depth"]
        lines += ["", f"Final head with all stages: accuracy {100 * full['test_accuracy']:.2f}%; "
                  f"batched inference {full['inference']['batched']['median_ms_per_sample']:.4f} ms/sample.", "",
                  "This full-depth reference uses the same trained adaptive weights and just the final classifier call. "
                  "It is separate from RotationNet, which has a different training objective.", "",
                  "| Threshold | Exit counts, stages 1 through last |", "|---|---|"]
        for sweep in adaptive["threshold_sweep"]:
            lines.append(f"| {sweep['threshold']:.2f} | " + ", ".join(str(v) for v in sweep["exit_distribution"].values()) + " |")
        default = next(s for s in adaptive["threshold_sweep"] if s["threshold"] == config["threshold"])
        lines += ["", "## Confidence at the default-threshold exit", "",
                  "| Stage | Samples | Mean confidence | Exit accuracy |", "|---|---:|---:|---:|"]
        for row in default["confidence_by_exit_stage"]:
            lines.append(f"| {row['stage']} | {row['samples']} | {_number(row['mean_exit_confidence'])} | {_number(row['exit_accuracy'])} |")
        candidates = [s for s in adaptive["threshold_sweep"] if s["test_accuracy"] >= full["test_accuracy"]
                      and s["inference"]["batched"]["median_ms_per_sample"] < full["inference"]["batched"]["median_ms_per_sample"]]
        if not candidates:
            lines += ["", "No tested adaptive threshold matched the same model's full-depth accuracy while reducing measured batched inference time."]
    lines += ["", "## Methodological limits", "",
              "- One seed and a short training budget are exploratory, not evidence of a general advantage.",
              "- Dense/low-rank use one core GELU; rotation models use one per stage. Depth and expressivity differ.",
              "- The learned dense input projection remains the largest shared component; core savings are not whole-model savings.",
              "- Multiply/add counts exclude nonlinearities, trigonometry, gating, softmax and memory costs. Dense BLAS may be faster.",
              "- Softmax confidence is uncalibrated and explicitly controls exits. Confidence/stage association is expected and does not establish difficulty.",
              "- Thresholds are preset, evaluated on the same test set, and not selected for deployment using these test results.",
              "- Timing repeats measure runtime variability, not independent training seeds; test-prefix timing may differ from full-set timing.",
              "- No novelty claim, no guaranteed accuracy retention; negative results are valid.", "",
              "## Reproduction", "", "Command recorded for this run:", "", "```powershell", metrics["command"], "```", "",
              "Configuration, source hashes and environment are saved with every attempt. "
              "Executed experiments also save split IDs and raw timing repeats.", ""]
    return "\n".join(lines)


def write_report(run_dir: Path, metrics: dict, root_summary: Path | None = None) -> None:
    report = render_report(metrics)
    (run_dir / "RESULTS.md").write_text(report, encoding="utf-8")
    if root_summary is not None:
        root_summary.write_text(report, encoding="utf-8")
