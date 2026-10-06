"""Run orchestration; dependency failures are recorded without fabricated metrics."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import traceback

from .reporting import write_json, write_predictions, write_report

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAMES = ("dense", "low-rank", "rotation", "adaptive-rotation")


def positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def probability(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise argparse.ArgumentTypeError("must be finite and between 0 and 1")
    return result


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Exploratory RotNet experiment: train/evaluate four models.")
    p.add_argument("--dataset", choices=("mnist", "fashion-mnist"), default="mnist")
    p.add_argument("--model", choices=("all",) + MODEL_NAMES, default="all")
    p.add_argument("--epochs", type=positive_int, default=5)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hidden-dim", type=positive_int, default=256)
    p.add_argument("--rank", type=positive_int, default=32)
    p.add_argument("--threshold", type=probability, default=0.90)
    p.add_argument("--thresholds", type=probability, nargs="+", default=[0.70, 0.80, 0.90, 0.95, 0.99])
    p.add_argument("--batch-size", type=positive_int, default=128)
    p.add_argument("--eval-batch-size", type=positive_int, default=256)
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--threads", type=positive_int, default=2)
    p.add_argument("--train-limit", type=positive_int)
    p.add_argument("--test-limit", type=positive_int)
    p.add_argument("--benchmark-samples", type=positive_int, default=1000)
    p.add_argument("--single-sample-benchmark-samples", type=positive_int, default=1000)
    p.add_argument("--timing-repeats", type=positive_int, default=3)
    p.add_argument("--warmup-batches", type=positive_int, default=2)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--results-dir", type=Path, default=ROOT / "results")
    p.add_argument("--run-name", help="unique directory name; existing runs are never overwritten")
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--preflight", action="store_true", help="check dependencies only; do not train")
    p.add_argument("--no-update-summary", action="store_true", help="leave repository RESULTS.md unchanged")
    return p


def source_hashes() -> dict:
    paths = [ROOT / "requirements.txt", ROOT / "pyproject.toml"]
    for directory in ("rotnet", "scripts", "tests"):
        paths.extend((ROOT / directory).rglob("*.py"))
    return {str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths) if path.is_file()}


def environment_info() -> dict:
    packages = {}
    for name in ("torch", "torchvision", "numpy", "matplotlib", "pytest"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "processor": platform.processor(), "logical_cpu_count": os.cpu_count(), "packages": packages}


def dependency_check() -> dict:
    failures = {}
    for name in ("torch", "torchvision", "numpy", "matplotlib"):
        try:
            importlib.import_module(name)
        except Exception as error:
            failures[name] = f"{type(error).__name__}: {error}"
    return failures


def main(argv=None) -> int:
    p = parser()
    args = p.parse_args(argv)
    if args.hidden_dim < 2 or args.hidden_dim & (args.hidden_dim - 1):
        p.error("--hidden-dim must be a power of two >= 2")
    if args.rank > args.hidden_dim:
        p.error("--rank must not exceed --hidden-dim")
    if not 0 <= args.seed < 2**32:
        p.error("--seed must be in [0, 2**32)")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        p.error("--learning-rate must be finite and positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        p.error("--weight-decay must be finite and nonnegative")
    run_name = args.run_name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_name):
        p.error("--run-name must start with a letter/digit and contain only letters, digits, _, . or -")
    run_dir = args.results_dir.resolve() / run_name
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        p.error(f"run already exists: {run_dir}; choose a new --run-name")
    # Required preset thresholds are always present; the configured default is also evaluated.
    args.thresholds = sorted(set(args.thresholds + [0.70, 0.80, 0.90, 0.95, 0.99, args.threshold]))
    config = {key: str(value.resolve()) if isinstance(value, Path) else value for key, value in vars(args).items()}
    command_args = sys.argv if argv is None else [str(ROOT / "scripts" / "run_experiment.py"), *argv]
    quoted = subprocess.list2cmdline([sys.executable, *command_args])
    command = "& " + quoted if os.name == "nt" else quoted
    metrics = {"schema_version": 1, "run_name": run_name, "status": "started", "config": config,
               "command": command, "started_utc": datetime.now(timezone.utc).isoformat(),
               "models": {}, "environment": environment_info(), "source_sha256": source_hashes()}
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "environment.json", metrics["environment"])
    frozen = subprocess.run([sys.executable, "-m", "pip", "freeze", "--all"],
                            capture_output=True, text=True, check=False)
    if frozen.returncode == 0:
        (run_dir / "requirements-resolved.txt").write_text(frozen.stdout, encoding="utf-8")
    write_json(run_dir / "source_hashes.json", metrics["source_sha256"])
    write_predictions(run_dir / "predictions.csv", [])
    summary = None if args.no_update_summary else ROOT / "RESULTS.md"
    rows = []
    try:
        failures = dependency_check()
        metrics["dependency_check"] = failures
        if failures:
            metrics["status"] = "blocked"
            metrics["error"] = "Dependencies unavailable: " + "; ".join(f"{k}: {v}" for k, v in failures.items())
            print(metrics["error"], file=sys.stderr, flush=True)
            return 2
        if args.preflight:
            metrics["status"] = "preflight-only"
            print("Dependencies import successfully. No training requested.", flush=True)
            return 0
        execute(args, run_dir, metrics, rows)
        metrics["status"] = "completed"
        return 0
    except Exception as error:
        metrics["status"] = "failed"
        metrics["error"] = f"{type(error).__name__}: {error}"
        (run_dir / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        return 1
    finally:
        metrics["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(run_dir / "metrics.json", metrics)
        write_predictions(run_dir / "predictions.csv", rows)
        write_report(run_dir, metrics, summary)
        print(f"Run status: {metrics['status']}. Artifacts: {run_dir}", flush=True)


def execute(args, run_dir, metrics, rows) -> None:
    import torch

    from .data import load_data, make_loader
    from .evaluate import benchmark_inference, evaluate_model
    from .metrics import operation_counts, parameter_counts
    from .models import make_model
    from .plots import create_plots
    from .train import seed_everything, train_model

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu")
    metrics["environment"].update({"torch_num_threads": torch.get_num_threads(),
                                   "torch_num_interop_threads": torch.get_num_interop_threads(),
                                   "torch_build": torch.__config__.show(),
                                   "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor()})
    write_json(run_dir / "environment.json", metrics["environment"])
    tensors, splits = load_data(args.dataset, args.data_dir, args.seed, args.train_limit, args.test_limit,
                                download=not args.no_download)
    write_json(run_dir / "splits.json", splits)
    metrics["data"] = {key: {k: v for k, v in value.items() if k != "sample_ids"} for key, value in splits.items()}
    test_loader = make_loader(tensors["test"], args.eval_batch_size, args.seed, shuffle=False)
    selected = MODEL_NAMES if args.model == "all" else (args.model,)
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir()

    def timings(model, threshold=None, full_depth=False):
        common = {"model": model, "dataset": tensors["test"], "device": device,
                  "repeats": args.timing_repeats, "warmup": args.warmup_batches,
                  "threshold": threshold, "full_depth": full_depth}
        return {
            "batched": benchmark_inference(**common, batch_size=args.eval_batch_size, max_samples=args.benchmark_samples),
            "single_sample": benchmark_inference(**common, batch_size=1, max_samples=args.single_sample_benchmark_samples),
        }

    for name in selected:
        # This also gives identical projection/classifier initialization across models.
        seed_everything(args.seed)
        model = make_model(name, args.hidden_dim, args.rank, args.threshold).to(device)
        train_loader = make_loader(tensors["train"], args.batch_size, args.seed, shuffle=True)
        print(f"Training {name} on {len(tensors['train'])} samples", flush=True)
        training = train_model(model, train_loader, args.epochs, args.learning_rate, args.weight_decay, device)
        torch.save({"model_name": name, "state_dict": model.state_dict(), "config": metrics["config"]},
                   checkpoints / f"{name}.pt")
        if name == "adaptive-rotation":
            sweeps = []
            for threshold in args.thresholds:
                print(f"  evaluating threshold {threshold:.2f}", flush=True)
                result, predictions = evaluate_model(model, test_loader, device, threshold)
                result["inference"] = timings(model, threshold)
                result["operations"] = operation_counts(model, result["average_stages"])
                sweeps.append(result)
                rows.extend({"model": name, **row} for row in predictions)
            default = next(s for s in sweeps if s["threshold"] == args.threshold)
            result = {**default, "threshold_sweep": sweeps}
            full, predictions = evaluate_model(model, test_loader, device, full_depth=True)
            full["inference"] = timings(model, full_depth=True)
            # Full depth uses only one head (model.forward), not the adaptive gates.
            full["operations"] = operation_counts(model)
            full["operations"]["classifier_multiply_add_ops_per_sample"] = 2 * args.hidden_dim * 10
            full["operations"]["whole_model_multiply_add_ops_per_sample"] = (
                full["operations"]["projection_multiply_add_ops_per_sample"]
                + full["operations"]["core_multiply_add_ops_per_sample"] + 2 * args.hidden_dim * 10)
            result["full_depth"] = full
            rows.extend({"model": name, **row} for row in predictions)
        else:
            result, predictions = evaluate_model(model, test_loader, device)
            result["inference"] = timings(model)
            result["operations"] = operation_counts(model)
            rows.extend({"model": name, **row} for row in predictions)
        result.update(training)
        result["parameters"] = parameter_counts(model)
        metrics["models"][name] = result
        write_json(run_dir / "metrics.json", metrics)
        write_predictions(run_dir / "predictions.csv", rows)
        print(f"  {name}: test accuracy={result['test_accuracy']:.4f}", flush=True)
    metrics["plots"] = create_plots(metrics, rows, run_dir / "plots")
