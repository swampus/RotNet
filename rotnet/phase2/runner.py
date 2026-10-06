"""Resumable, append-only case execution for the Phase 2 experiment."""

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import traceback

import torch

from rotnet.data import load_data, make_loader
from rotnet.evaluate import benchmark_inference, evaluate_model
from rotnet.experiment import ROOT, environment_info, source_hashes
from rotnet.metrics import parameter_counts
from rotnet.reporting import write_json, write_predictions
from rotnet.train import seed_everything, train_model

from .analysis import collect_all_stages, select_policy, stage_transitions
from .data import resize_direct, split_record, validation_split
from .models import arithmetic, controlled_model
from .runtime import optimize_inference, profile_calls, verify_equivalence

FAMILIES = ("replication", "stages", "components", "validation", "direct16", "runtime")
MODEL_NAMES = ("dense", "low-rank", "rotation", "adaptive-rotation")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def phase1_manifest():
    paths = [ROOT / name for name in ("README.md", "RESULTS.md", "requirements.txt", "pyproject.toml", ".gitignore")]
    paths += [p for p in (ROOT / "rotnet").rglob("*.py") if "phase2" not in p.parts]
    paths += [p for p in (ROOT / "tests").glob("*.py") if not p.name.startswith("test_phase2")]
    paths += [ROOT / "scripts" / "run_experiment.py"]
    paths += [p for p in (ROOT / "results").rglob("*") if p.is_file() and "phase2" not in p.relative_to(ROOT / "results").parts]
    return {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in sorted(set(paths)) if p.is_file()}


def verify_manifest(manifest):
    changed = [name for name, digest in manifest.items() if not (ROOT / name).exists() or sha(ROOT / name) != digest]
    return {"status": "passed" if not changed else "failed", "files_checked": len(manifest), "changed": changed}


def plan(seeds, datasets, families):
    jobs = []
    for dataset in datasets:
        for family in families:
            for seed in seeds:
                if family in ("replication", "direct16"):
                    specs = [(name, 8, "learned") for name in MODEL_NAMES]
                elif family == "stages":
                    specs = [("rotation", stages, "learned") for stages in (1, 2, 4)]
                elif family == "components":
                    specs = [("rotation", 8, variant) for variant in
                             ("identity", "random-fixed", "final-gelu", "coordinate-only", "no-affine")]
                    specs += [("projection-only", 8, "learned")]
                elif family == "validation":
                    specs = [("adaptive-rotation", 8, "learned")]
                else:
                    specs = [("runtime", 8, "learned")]
                for name, stages, variant in specs:
                    jobs.append({"dataset": dataset, "family": family, "seed": seed, "model": name,
                                 "stages": stages, "variant": variant,
                                 "representation": "16x16" if family == "direct16" else "28x28"})
    return jobs


def case_id(job):
    return "__".join(str(job[k]) for k in ("dataset", "family", "seed", "model", "stages", "variant"))


def attempt_history(previous):
    if not previous:
        return []
    history = list(previous.get("previous_attempts", []))
    if previous.get("path"):
        history.append({key: previous.get(key) for key in
                        ("status", "path", "error", "attempt", "started_utc", "finished_utc")})
    return history


def parser():
    p = argparse.ArgumentParser(description="Phase 2 falsification suite; original Phase 1 files are untouched.")
    p.add_argument("--phase1-run", type=Path, default=ROOT / "results" / "mnist-seed42-5epochs")
    p.add_argument("--run-dir", type=Path, default=ROOT / "results" / "phase2" / "mnist-five-seeds")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    p.add_argument("--datasets", choices=("mnist", "fashion-mnist"), nargs="+", default=["mnist", "fashion-mnist"])
    p.add_argument("--families", choices=FAMILIES, nargs="+", default=list(FAMILIES))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--validation-size", type=int, default=5000)
    p.add_argument("--split-seed", type=int, default=202602)
    p.add_argument("--tolerance-pp", type=float, default=0.1)
    p.add_argument("--smoke", action="store_true", help="test-only 128/64 sample, one-epoch run; never pooled with full experiments")
    return p


def timing(model, dataset, config, threshold=None, full_depth=False):
    kwargs = {"model": model, "dataset": dataset, "device": torch.device("cpu"),
              "threshold": threshold, "full_depth": full_depth,
              "repeats": config["timing_repeats"], "warmup": config["warmup_batches"]}
    return {"batched": benchmark_inference(**kwargs, batch_size=config["eval_batch_size"], max_samples=config["benchmark_samples"]),
            "single_sample": benchmark_inference(**kwargs, batch_size=1, max_samples=config["single_sample_benchmark_samples"])}


def save_stage_data(path, logits, targets, ids):
    torch.save({"logits": logits, "targets": targets, "sample_ids": ids}, path / "stage_logits.pt")
    write_json(path / "stage_analysis.json", stage_transitions(logits, targets))
    import csv
    with (path / "stage_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        stages = logits.shape[1]
        writer.writerow(["sample_id", "target"] + [f"prediction_stage_{s}" for s in range(1, stages + 1)]
                        + [f"confidence_stage_{s}" for s in range(1, stages + 1)])
        confidences, predictions = logits.softmax(-1).max(-1)
        for sample, target, prediction, confidence in zip(ids.tolist(), targets.tolist(), predictions.tolist(), confidences.tolist()):
            writer.writerow([sample, target, *prediction, *confidence])


def train_case(job, path, tensors, config, args):
    seed = job["seed"]
    seed_everything(seed)
    model = controlled_model(job["model"], stages=job["stages"], variant=job["variant"],
                             representation=job["representation"], hidden_dim=config["hidden_dim"],
                             rank=config["rank"], threshold=config["threshold"])
    train, test = tensors["train"], tensors["test"]
    if job["representation"] == "16x16":
        train, test = resize_direct(train), resize_direct(test)
    validation = None
    if job["family"] == "validation":
        validation_size = min(args.validation_size, len(train) // 4) if args.smoke else args.validation_size
        train, validation = validation_split(train, validation_size, args.split_seed)
    write_json(path / "splits.json", {"train": split_record(train), "test": split_record(test),
                                      **({"validation": split_record(validation)} if validation is not None else {})})
    training = train_model(model, make_loader(train, config["batch_size"], seed, True),
                           config["epochs"], config["learning_rate"], config["weight_decay"], torch.device("cpu"))
    torch.save({"state_dict": model.state_dict(), "job": job, "config": config}, path / "checkpoint.pt")
    test_loader = make_loader(test, config["eval_batch_size"], seed, False)
    threshold = config["threshold"] if job["model"] == "adaptive-rotation" else None
    full_depth = False
    selection = None
    if validation is not None:
        val_loader = make_loader(validation, config["eval_batch_size"], seed, False)
        val_logits, val_targets, val_ids = collect_all_stages(model, val_loader, torch.device("cpu"))
        thresholds = [round(i / 100, 2) for i in range(10, 100, 5)] + [0.975, 0.99, 0.995, 0.999, 1.0]
        selection = select_policy(val_logits, val_targets, thresholds, args.tolerance_pp)
        threshold = selection["selected"]["threshold"]
        full_depth = selection["selected"]["policy"] == "full-depth"
        # The policy artifact is written before obtaining any official test labels/results.
        write_json(path / "selected_policy.json", selection)
        torch.save({"logits": val_logits, "targets": val_targets, "sample_ids": val_ids}, path / "validation_logits.pt")
    metrics, rows = evaluate_model(model, test_loader, torch.device("cpu"), threshold, full_depth=full_depth)
    write_predictions(path / "predictions.csv", [{"model": job["model"], **row} for row in rows])
    metrics.update({**training, "parameters": parameter_counts(model),
                    "inference": timing(model, test, config, threshold, full_depth),
                    "executed_train_samples": len(train), "test_samples": len(test)})
    mean_stages = metrics.get("average_stages")
    if full_depth:
        mean_stages = len(model.core.stages)
        metrics.update({"average_stages": mean_stages, "median_stages": mean_stages,
                        "exit_distribution": {str(i): len(test) if i == mean_stages else 0 for i in range(1, mean_stages + 1)}})
    metrics["operations"] = arithmetic(model, mean_stages, final_head_only=full_depth)
    if selection is not None:
        metrics["selection"] = selection
    if job["model"] == "adaptive-rotation":
        logits, targets, ids = collect_all_stages(model, test_loader, torch.device("cpu"))
        metrics["stage_analysis"] = stage_transitions(logits, targets)
        save_stage_data(path, logits, targets, ids)
    return metrics


def load_trained_case(suite, dataset, seed, name):
    job = {"dataset": dataset, "family": "replication", "seed": seed, "model": name, "stages": 8, "variant": "learned"}
    record = suite["cases"].get(case_id(job))
    if record is None or record["status"] != "completed":
        raise RuntimeError(f"runtime requires completed replication checkpoint: {case_id(job)}")
    saved = torch.load(Path(suite["run_dir"]) / record["path"] / "checkpoint.pt", map_location="cpu", weights_only=True)
    model = controlled_model(name)
    model.load_state_dict(saved["state_dict"])
    return model.eval()


def runtime_case(job, path, tensors, config, suite):
    loader = make_loader(tensors["test"], config["eval_batch_size"], job["seed"], False)
    models = {"dense": load_trained_case(suite, job["dataset"], job["seed"], "dense"),
              "original-rotation": load_trained_case(suite, job["dataset"], job["seed"], "rotation"),
              "original-adaptive": load_trained_case(suite, job["dataset"], job["seed"], "adaptive-rotation")}
    models["optimized-rotation"] = optimize_inference(models["original-rotation"])
    models["optimized-adaptive"] = optimize_inference(models["original-adaptive"])
    verification = {name: verify_equivalence(models["original-" + name], models["optimized-" + name],
                                           loader, torch.device("cpu"), threshold=config["threshold"])
                    for name in ("rotation", "adaptive")}
    write_json(path / "equivalence.json", verification)
    # Counterbalance all five implementations within each timing repetition.
    import random
    generator = random.Random(job["seed"])
    benchmark = {name: {} for name in models}
    for batch, label, count in ((1, "single_sample", config["single_sample_benchmark_samples"]),
                                (config["eval_batch_size"], "batched", config["benchmark_samples"])):
        passes = {name: [] for name in models}
        orders = []
        for repeat in range(config["timing_repeats"]):
            order = list(models)
            generator.shuffle(order)
            orders.append(order)
            for name in order:
                measured = benchmark_inference(models[name], tensors["test"], torch.device("cpu"), batch,
                                               repeats=1, warmup=config["warmup_batches"],
                                               max_samples=count, threshold=config["threshold"] if "adaptive" in name else None)
                benchmark[name][label] = measured
                passes[name].extend(measured["pass_seconds"])
        import statistics
        for name in models:
            measured = benchmark[name][label]
            measured["pass_seconds"] = passes[name]
            measured["repeats"] = len(passes[name])
            measured["median_wall_seconds"] = statistics.median(passes[name])
            measured["median_ms_per_sample"] = 1000 * measured["median_wall_seconds"] / measured["samples"]
            measured["samples_per_second"] = measured["samples"] / measured["median_wall_seconds"]
            measured["counterbalanced_orders"] = orders
    profiles = {}
    if job["seed"] == 42:
        for name, model in models.items():
            for batch in (1, config["eval_batch_size"]):
                try:
                    profiles[f"{name}__batch{batch}"] = profile_calls(model, tensors["test"].tensors[0][:batch], "adaptive" in name)
                except Exception as error:
                    profiles[f"{name}__batch{batch}"] = {"status": "failed", "error": str(error)}
    write_json(path / "profiles.json", profiles)
    return {"verification": verification, "benchmarks": benchmark,
            "torch_compile": {"status": "not_attempted", "reason": "optional compiler path omitted; portable eager cache optimization tested"},
            "profiles": profiles}


def persist(suite, args):
    run_dir = Path(suite["run_dir"])
    write_json(run_dir / "suite.json", suite)
    from .report import generate_reports
    generate_reports(suite, run_dir, ROOT / "PHASE2_RESULTS.md")


def main(argv=None):
    args = parser().parse_args(argv)
    run_dir = args.run_dir.resolve()
    if not run_dir.is_relative_to((ROOT / "results" / "phase2").resolve()):
        raise ValueError("--run-dir must be a new descendant of results/phase2, protecting Phase 1 runs")
    if len(set(args.seeds)) != len(args.seeds) or any(not 0 <= s < 2**32 for s in args.seeds):
        raise ValueError("seeds must be unique uint32 values")
    if args.validation_size < 1 or not math.isfinite(args.tolerance_pp) or args.tolerance_pp < 0:
        raise ValueError("validation size must be positive and tolerance finite/nonnegative")
    if args.resume:
        suite = json.loads((run_dir / "suite.json").read_text(encoding="utf-8"))
        if suite["settings"]["smoke"] != args.smoke:
            raise ValueError("smoke and full runs must not share artifacts")
        for key in ("seeds", "split_seed", "validation_size", "tolerance_pp"):
            if suite["settings"][key] != getattr(args, key):
                raise ValueError(f"resume {key} differs from recorded configuration")
        current_hashes = source_hashes()
        changed = [name for name, digest in suite["source_sha256"].items() if current_hashes.get(name) != digest]
        protected = [name for name in changed if
                     name.startswith("rotnet/models/") or name in
                     {"rotnet/train.py", "rotnet/data.py", "rotnet/evaluate.py", "rotnet/metrics.py",
                      "rotnet/phase2/models.py", "rotnet/phase2/data.py", "rotnet/phase2/analysis.py", "rotnet/phase2/runtime.py"}]
        if protected:
            raise ValueError(f"training/evaluation source changed; use a new suite: {protected}")
        suite.setdefault("resume_source_snapshots", []).append({"utc": datetime.now(timezone.utc).isoformat(),
                                                                "source_sha256": current_hashes,
                                                                "changed_since_start": changed})
        config = suite["config"]
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        config = json.loads((args.phase1_run / "config.json").read_text(encoding="utf-8"))
        if config["device"] != "cpu" or config["hidden_dim"] != 256 or config["eval_batch_size"] != 256:
            raise ValueError("this controlled suite expects Phase 1 CPU, width 256, evaluation batch 256")
        if args.smoke:
            config.update({"epochs": 1, "train_limit": 128, "test_limit": 64,
                           "benchmark_samples": 16, "single_sample_benchmark_samples": 8, "timing_repeats": 1})
        suite = {"schema_version": 1, "run_dir": str(run_dir), "status": "running",
                 "phase1_run": str(args.phase1_run.resolve()), "config": config,
                 "settings": {"seeds": args.seeds, "split_seed": args.split_seed,
                              "validation_size": args.validation_size, "tolerance_pp": args.tolerance_pp, "smoke": args.smoke},
                 "environment": environment_info(), "source_sha256": source_hashes(),
                 "phase1_manifest": phase1_manifest(), "cases": {}, "dataset_failures": {},
                 "started_utc": datetime.now(timezone.utc).isoformat()}
        write_json(run_dir / "phase1_manifest.json", suite["phase1_manifest"])
        frozen = subprocess.run([sys.executable, "-m", "pip", "freeze", "--all"], capture_output=True, text=True)
        (run_dir / "requirements-resolved.txt").write_text(frozen.stdout, encoding="utf-8")
    # A resumed worker is active even if its saved state was interrupted.
    # This also defers plot generation until measurements finish.
    suite["status"] = "running"
    torch.set_num_threads(config["threads"])
    torch.set_num_interop_threads(1)
    suite["environment"].update({"torch_num_threads": torch.get_num_threads(), "torch_build": torch.__config__.show()})
    suite["requested_jobs"] = plan(args.seeds, args.datasets, args.families)
    tensors_by_dataset = {}
    try:
        for dataset in args.datasets:
            try:
                tensors, _ = load_data(dataset, args.data_dir, 42,
                                       config.get("train_limit"), config.get("test_limit"), download=False)
                tensors_by_dataset[dataset] = tensors
                suite["dataset_failures"].pop(dataset, None)
            except Exception as error:
                suite["dataset_failures"][dataset] = {"status": "blocked", "error": f"{type(error).__name__}: {error}",
                                                       "reason": "local dataset unavailable; network download is unavailable in this session"}
                print(f"BLOCKED dataset {dataset}: {error}", flush=True)
        persist(suite, args)
        for index, job in enumerate(suite["requested_jobs"], start=1):
            identifier = case_id(job)
            previous = suite["cases"].get(identifier)
            if previous and previous["status"] == "completed":
                continue
            if job["dataset"] not in tensors_by_dataset:
                suite["cases"][identifier] = {"job": job, "status": "blocked", "error": suite["dataset_failures"][job["dataset"]]["error"]}
                continue
            if previous and previous["status"] == "failed" and not args.retry_failed:
                continue
            case_root = run_dir / "cases" / identifier
            case_root.mkdir(parents=True, exist_ok=True)
            attempt = len(list(case_root.glob("attempt*"))) + 1
            path = case_root / f"attempt{attempt:03d}"
            path.mkdir(exist_ok=False)
            record = {"job": job, "status": "running", "path": str(path.relative_to(run_dir)).replace("\\", "/"),
                      "started_utc": datetime.now(timezone.utc).isoformat(), "attempt": attempt,
                      "previous_attempts": attempt_history(previous)}
            suite["cases"][identifier] = record
            write_json(path / "job.json", job)
            write_json(run_dir / "suite.json", suite)
            print(f"CASE {index}/{len(suite['requested_jobs'])}: {identifier}", flush=True)
            try:
                if job["family"] == "runtime":
                    result = runtime_case(job, path, tensors_by_dataset[job["dataset"]], config, suite)
                else:
                    result = train_case(job, path, tensors_by_dataset[job["dataset"]], config, args)
                record.update({"status": "completed", "metrics": result})
                if "test_accuracy" in result:
                    print(f"  TEST {100 * result['test_accuracy']:.2f}%", flush=True)
            except Exception as error:
                record.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
                (path / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
                print(traceback.format_exc(), flush=True)
            record["finished_utc"] = datetime.now(timezone.utc).isoformat()
            write_json(path / "metrics.json", record)
            persist(suite, args)
            gc.collect()
        suite["status"] = "completed-with-blocked-dataset" if suite["dataset_failures"] else "completed"
        if any(r["status"] == "failed" for r in suite["cases"].values()):
            suite["status"] = "completed-with-failures"
    except KeyboardInterrupt:
        suite["status"] = "interrupted"
        raise
    finally:
        suite["phase1_preservation"] = verify_manifest(suite["phase1_manifest"])
        suite["updated_utc"] = datetime.now(timezone.utc).isoformat()
        persist(suite, args)
    return 0 if not any(r["status"] == "failed" for r in suite["cases"].values()) else 1
