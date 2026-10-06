"""Descriptive five-seed aggregates and evidence-based report, without p-values."""

import csv
import statistics
from pathlib import Path

from rotnet.reporting import write_json


def summary(values):
    values = list(values)
    return {"n": len(values), "mean": statistics.mean(values) if values else None,
            "std": statistics.stdev(values) if len(values) > 1 else None,
            "min": min(values) if values else None, "max": max(values) if values else None}


def completed(suite, dataset=None, family=None):
    return [r for r in suite["cases"].values() if r["status"] == "completed"
            and (dataset is None or r["job"]["dataset"] == dataset)
            and (family is None or r["job"]["family"] == family)]


def aggregate(records):
    return {"seeds": [r["job"]["seed"] for r in records],
            "accuracy": summary(r["metrics"]["test_accuracy"] for r in records),
            "training_seconds": summary(r["metrics"]["training_seconds"] for r in records),
            "core_ops": summary(r["metrics"]["operations"]["core_multiply_add_ops_per_sample"] for r in records),
            "whole_ops": summary(r["metrics"]["operations"]["whole_model_multiply_add_ops_per_sample"] for r in records),
            "core_parameters": summary(r["metrics"]["parameters"]["core_trainable_parameters"] for r in records),
            "core_total_parameters": summary(r["metrics"]["parameters"]["core_total_parameters"] for r in records),
            "batched_ms": summary(r["metrics"]["inference"]["batched"]["median_ms_per_sample"] for r in records),
            "single_ms": summary(r["metrics"]["inference"]["single_sample"]["median_ms_per_sample"] for r in records)}


def groups(suite, dataset, family, key="model"):
    records = completed(suite, dataset, family)
    keys = sorted(set(str(r["job"][key]) for r in records))
    return {k: aggregate([r for r in records if str(r["job"][key]) == k]) for k in keys}


def fmt(value, digits=3):
    return "pending" if value is None else f"{value:.{digits}f}"


def mean_std(stat, factor=1, digits=3):
    if stat["mean"] is None:
        return "pending"
    return f"{factor * stat['mean']:.{digits}f} ± " + (f"{factor * stat['std']:.{digits}f}" if stat["std"] is not None else "pending SD")


def model_table(group):
    if not group:
        return ["No completed measurements in this group.", ""]
    lines = ["| Model/control | Seeds | Mean accuracy ± sample SD (%) | Min–max (%) | Mean core ops | Mean core trainable/total params | Mean train s | Mean batch=256 ms/sample | Mean batch=1 ms/sample |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, row in group.items():
        acc = row["accuracy"]
        lines.append(f"| {name} | {acc['n']} | {mean_std(acc, 100)} | "
                     f"{100 * acc['min']:.2f}–{100 * acc['max']:.2f} | {fmt(row['core_ops']['mean'], 1)} | "
                     f"{fmt(row['core_parameters']['mean'], 0)}/{fmt(row['core_total_parameters']['mean'], 0)} | "
                     f"{fmt(row['training_seconds']['mean'], 2)} | {fmt(row['batched_ms']['mean'], 5)} | {fmt(row['single_ms']['mean'], 5)} |")
    return lines + [""]


def evidence_answers(suite):
    replication = groups(suite, "mnist", "replication")
    components = groups(suite, "mnist", "components", "variant")
    # Components key must distinguish projection-only from learned variants.
    identity = [r for r in completed(suite, "mnist", "components") if r["job"]["variant"] == "identity"]
    answers = {}
    if "rotation" in replication and "dense" in replication:
        dense = {r["job"]["seed"]: r["metrics"]["test_accuracy"] for r in completed(suite, "mnist", "replication") if r["job"]["model"] == "dense"}
        deltas = [100 * (r["metrics"]["test_accuracy"] - dense[r["job"]["seed"]])
                  for r in completed(suite, "mnist", "replication") if r["job"]["model"] == "rotation" and r["job"]["seed"] in dense]
        answers["1"] = (f"Across {len(deltas)} paired MNIST seeds, RotationNet minus Dense accuracy averaged "
                        f"{statistics.mean(deltas):+.3f} pp, range {min(deltas):+.3f} to {max(deltas):+.3f} pp. "
                        "This is descriptive consistency evidence, not a significance or noninferiority test.")
    else:
        answers["1"] = "MNIST replication is pending or incomplete."
    answers["2"] = ("Unknown: Fashion-MNIST is BLOCKED by unavailable local data/network. Generalization beyond MNIST has not been tested."
                     if "fashion-mnist" in suite["dataset_failures"] else "See the separate Fashion-MNIST five-seed table; five seeds alone do not establish broad generalization.")
    if identity and "rotation" in replication:
        learned = replication["rotation"]["accuracy"]["mean"]
        frozen = aggregate(identity)["accuracy"]["mean"]
        random = [r for r in completed(suite, "mnist", "components") if r["job"]["variant"] == "random-fixed"]
        answers["3"] = f"Learned minus identity mean accuracy is {100 * (learned - frozen):+.3f} pp. "
        if random:
            answers["3"] += f"Learned minus frozen-random is {100 * (learned - aggregate(random)['accuracy']['mean']):+.3f} pp. "
        answers["3"] += "These controls, including coordinate-only and no-affine variants, test necessity; they do not uniquely identify a causal mechanism."
    else:
        answers["3"] = "Component controls are pending."
    adaptive = [r for r in completed(suite, "mnist", "replication") if r["job"]["model"] == "adaptive-rotation"]
    if adaptive:
        rescues = [r["metrics"]["stage_analysis"]["stage1_errors_correct_at_final"] for r in adaptive]
        damages = [r["metrics"]["stage_analysis"]["stage1_correct_wrong_at_final"] for r in adaptive]
        answers["4"] = (f"Per seed, mean {statistics.mean(rescues):.1f} stage-1 errors are correct at the final stage, "
                        f"while mean {statistics.mean(damages):.1f} stage-1 correct predictions are wrong at the final stage. "
                        "Error rescue is observed; 'difficulty' is not independently measured.")
    else:
        answers["4"] = "Stage rescue/damage measurements are pending."
    selected = completed(suite, "mnist", "validation")
    if selected:
        stages = [r["metrics"]["average_stages"] for r in selected]
        deltas = [100 * (r["metrics"]["test_accuracy"] - r["metrics"]["stage_analysis"]["accuracy_by_stage"][-1]) for r in selected]
        savings = []
        for r in selected:
            m = r["metrics"]
            full = (0 if r["job"]["representation"] == "16x16" else 401408) + 10240 + 5120
            savings.append(100 * (1 - m["operations"]["whole_model_multiply_add_ops_per_sample"] / full))
        answers["5"] = (f"Validation-selected policies use mean {statistics.mean(stages):.3f}/8 test stages. "
                        f"Mean test accuracy difference from the same weights' final head is {statistics.mean(deltas):+.3f} pp "
                        f"(range {min(deltas):+.3f} to {max(deltas):+.3f}); mean whole-model multiply/add savings "
                        f"are {statistics.mean(savings):.2f}%. A validation tolerance does not guarantee test accuracy retention.")
    else:
        answers["5"] = "Validation-selected policies are pending."
    runtime = completed(suite, "mnist", "runtime")
    if runtime:
        vals = {name: statistics.mean(r["metrics"]["benchmarks"][name]["batched"]["median_ms_per_sample"] for r in runtime)
                for name in runtime[0]["metrics"]["benchmarks"]}
        ratio = vals["optimized-rotation"] / vals["dense"]
        answers["6"] = f"Optimized RotationNet takes {ratio:.2f}× Dense's batch=256 time on this CPU. "
        answers["6"] += "A dense latency advantage remains." if ratio > 1 else "A measured batched latency advantage exists on this machine; verify other workloads/hardware."
        original = statistics.mean(r["metrics"]["benchmarks"]["original-adaptive"]["batched"]["median_ms_per_sample"] for r in runtime)
        cached = statistics.mean(r["metrics"]["benchmarks"]["optimized-adaptive"]["batched"]["median_ms_per_sample"] for r in runtime)
        if cached > original:
            answers["6"] += f" The cached adaptive attempt was {100 * (cached/original-1):.2f}% slower in batched inference."
    else:
        answers["6"] = "Runtime comparisons are pending."
    direct = groups(suite,"mnist","direct16")
    if "dense" in direct and "rotation" in direct:
        answers["1"] += (f" In the separate 16x16/no-projection comparison, Dense averaged "
                          f"{100*direct['dense']['accuracy']['mean']:.3f}% and Rotation "
                          f"{100*direct['rotation']['accuracy']['mean']:.3f}%; the structured replacement lost "
                          f"{100*(direct['dense']['accuracy']['mean']-direct['rotation']['accuracy']['mean']):.3f} pp.")
    if "adaptive-rotation" in direct and "rotation" in direct:
        extra = 100*(direct["adaptive-rotation"]["whole_ops"]["mean"]/direct["rotation"]["whole_ops"]["mean"]-1)
        answers["5"] += (f" In the separate direct-input preset-0.90 experiment, adaptive whole-model arithmetic "
                          f"was {extra:.2f}% HIGHER than fixed Rotation, due to repeated classifier calls, "
                          "with lower mean accuracy. Fewer stages did not imply cheaper whole-model execution.")
    answers["7"] = ("Only a targeted follow-up is justified: test whether the mixing is necessary after the input-projection control, "
                    "then assess a fused structured kernel against optimized dense/low-rank baselines and include a harder dataset. "
                    "The current prototype is not evidence of a practical CPU speed advantage or a general architectural benefit.")
    return answers


def generate_reports(suite, run_dir: Path, root_report: Path | None):
    aggregates = {}
    for dataset in ("mnist", "fashion-mnist"):
        for family in ("replication", "direct16"):
            aggregates[f"{dataset}/{family}"] = groups(suite, dataset, family)
        aggregates[f"{dataset}/validation"] = groups(suite,dataset,"validation")
        stages = groups(suite,dataset,"stages",key="stages")
        replica = aggregates[f"{dataset}/replication"]
        if "rotation" in replica:
            stages["8"] = replica["rotation"]
        aggregates[f"{dataset}/stages"] = stages
        controls = completed(suite,dataset,"components")
        control_keys = set(r["job"]["model"] if r["job"]["model"] == "projection-only" else r["job"]["variant"] for r in controls)
        components = {key:aggregate([r for r in controls if
                      (r["job"]["model"] if r["job"]["model"] == "projection-only" else r["job"]["variant"]) == key]) for key in sorted(control_keys)}
        if "rotation" in replica:
            components["learned"] = replica["rotation"]
        aggregates[f"{dataset}/components"] = components
    records = completed(suite)
    write_json(run_dir / "aggregates.json", aggregates)
    with (run_dir / "cases.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["dataset", "family", "seed", "model", "stages", "variant", "representation", "status", "test_accuracy",
                  "core_trainable_parameters", "core_ops", "whole_ops", "training_seconds", "batch256_ms", "batch1_ms", "path", "error"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in suite["cases"].values():
            row = {**record["job"], "status": record["status"], "path": record.get("path"), "error": record.get("error")}
            metrics = record.get("metrics", {})
            if "test_accuracy" in metrics:
                row.update({"test_accuracy": metrics["test_accuracy"], "core_trainable_parameters": metrics["parameters"]["core_trainable_parameters"],
                            "core_ops": metrics["operations"]["core_multiply_add_ops_per_sample"], "whole_ops": metrics["operations"]["whole_model_multiply_add_ops_per_sample"],
                            "training_seconds": metrics["training_seconds"], "batch256_ms": metrics["inference"]["batched"]["median_ms_per_sample"],
                            "batch1_ms": metrics["inference"]["single_sample"]["median_ms_per_sample"]})
            writer.writerow(row)
    answers = evidence_answers(suite)
    write_json(run_dir / "evidence_answers.json", answers)
    lines = ["# Phase 2 RotNet: replication and falsification", "", "## 1. Executive summary", "",
             f"Status: **{suite['status']}**. Run directory: `{run_dir}`.", "",
             f"Completed cases: {len(records)}; failed cases: {sum(r['status'] == 'failed' for r in suite['cases'].values())}; "
             f"blocked cases: {sum(r['status'] == 'blocked' for r in suite['cases'].values())}.", "",
             "**TEST-ONLY SMOKE RUN: excluded from research conclusions.**" if suite["settings"]["smoke"] else
             "Accuracy, training/inference times and exits are measured. Parameter counts are enumerated; scalar multiply/add arithmetic is theoretical accounting. "
             "SD is the sample standard deviation across seeds (ddof=1), not a confidence interval.", "",
             "Seeds: " + ", ".join(map(str, suite["settings"]["seeds"])) + ". "
             f"Each model trains for {suite['config']['epochs']} epochs with unchanged Phase 1 AdamW settings, "
             f"batches {suite['config']['batch_size']}/{suite['config']['eval_batch_size']}, width 256, rank 32, two CPU threads.", ""]
    if not suite["settings"]["smoke"]:
        lines += [answers[number] for number in ("1", "3", "5", "6") if "pending" not in answers[number].lower()]
        lines.append("")
    if "fashion-mnist" in suite["dataset_failures"]:
        lines += ["**Fashion-MNIST: BLOCKED.** No local dataset; network sockets are unavailable. No Fashion-MNIST accuracy or timing is fabricated.", ""]
    lines += ["## 2. Multi-seed MNIST replication", ""] + model_table(aggregates["mnist/replication"])
    lines += ["Adaptive replication uses the unchanged preset 0.90 threshold and full 60,000-example training split. "
              "It is kept separate from validation-selected policies.", "", "## 3. Multi-seed Fashion-MNIST replication", ""]
    if "fashion-mnist" in suite["dataset_failures"]:
        lines += ["**BLOCKED** — " + suite["dataset_failures"]["fashion-mnist"]["error"], "",
                  "The same suite supports Fashion-MNIST when the official dataset is cached; no dataset-specific tuning is implemented.", ""]
    else:
        lines += model_table(aggregates["fashion-mnist/replication"])
    stage_records = completed(suite, "mnist", "stages") + [r for r in completed(suite, "mnist", "replication") if r["job"]["model"] == "rotation"]
    stage_groups = {str(s): aggregate([r for r in stage_records if r["job"]["stages"] == s]) for s in (1, 2, 4, 8)
                    if any(r["job"]["stages"] == s for r in stage_records)}
    lines += ["## 4. Rotation stage-count ablation", ""] + model_table(stage_groups)
    lines += ["Stages are deterministic prefixes (strides 1,2,...). Fewer stages change connectivity as well as nonlinear depth. "
              "Eight-stage values reuse freshly measured Phase 2 replication cases.", ""]
    adaptive = [r for r in completed(suite, "mnist", "replication") if r["job"]["model"] == "adaptive-rotation"]
    if adaptive:
        lines += ["Already-trained adaptive classifier at each fixed stage (no threshold):", "",
                  "| Stage | Mean accuracy ± SD (%) |", "|---|---:|"]
        for s in range(8):
            lines.append(f"| {s+1} | {mean_std(summary(r['metrics']['stage_analysis']['accuracy_by_stage'][s] for r in adaptive),100)} |")
        lines.append("")
    component_records = completed(suite, "mnist", "components")
    component_groups = {}
    for variant in ("learned", "identity", "random-fixed", "final-gelu", "coordinate-only", "no-affine", "projection-only"):
        selected = ([r for r in stage_records if r["job"]["stages"] == 8] if variant == "learned" else
                    [r for r in component_records if (r["job"]["model"] == "projection-only" if variant == "projection-only" else r["job"]["variant"] == variant)])
        if selected:
            component_groups[variant] = aggregate(selected)
    lines += ["## 5. Rotation-component controls", ""] + model_table(component_groups)
    lines += ["Identity retains theta=0 as frozen parameters and executes the same rotation arithmetic. Coordinate-only removes mixing entirely; "
              "it is the same eight-stage coordinate-affine/GELU depth. Random-fixed freezes the original random angles. "
              "Final-GELU uses one GELU after all eight affine rotation stages. No-affine removes scale/bias entirely. "
              "Projection-only removes the hidden core, retaining input projection/GELU and classifier. "
              "Frozen parameters are included in total but excluded from trainable counts.", "",
              "## 6. Proper validation-selected early exit", "",
              "A fixed seeded 55,000/5,000 partition of official training data is shared across all five seeds. "
              "Only validation chooses the policy: minimum mean stages subject to accuracy within 0.1 pp of the same weights' final head. "
              "The policy JSON is saved before evaluating the official test labels. Full-depth/no-gating is a fallback. "
              "This separate experiment has fewer training updates than the 60,000-example replication; it is not pooled with that table.", ""]
    validation = completed(suite, "mnist", "validation")
    if validation:
        lines += ["| Seed | Selected threshold/policy | Validation accuracy (%) | Full validation (%) | Test accuracy (%) | Full test (%) | Mean/median test stages | Core / whole ops | Batch256 / batch1 ms |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for r in validation:
            m = r["metrics"]
            selected = m["selection"]["selected"]
            lines.append(f"| {r['job']['seed']} | {selected['threshold'] if selected['threshold'] is not None else 'full-depth'} | "
                         f"{100 * selected['validation_accuracy']:.2f} | {100*m['selection']['full_depth_validation_accuracy']:.2f} | "
                         f"{100*m['test_accuracy']:.2f} | {100*m['stage_analysis']['accuracy_by_stage'][-1]:.2f} | "
                         f"{m['average_stages']:.3f}/{m['median_stages']:.1f} | "
                         f"{m['operations']['core_multiply_add_ops_per_sample']:.1f}/{m['operations']['whole_model_multiply_add_ops_per_sample']:.1f} | "
                         f"{m['inference']['batched']['median_ms_per_sample']:.5f}/{m['inference']['single_sample']['median_ms_per_sample']:.5f} |")
        lines += ["", "Exit counts for each frozen policy (stages 1..8):", ""]
        for r in validation:
            lines.append(f"- Seed {r['job']['seed']}: " + ", ".join(str(v) for v in r["metrics"]["exit_distribution"].values()))
        violations = [r for r in validation if 100*(r["metrics"]["stage_analysis"]["accuracy_by_stage"][-1]-r["metrics"]["test_accuracy"])
                      > r["metrics"]["selection"]["tolerance_percentage_points"]+1e-10]
        lines += ["", f"**The validation tolerance failed to transfer on {len(violations)}/{len(validation)} test seeds.**"]
        for r in violations:
            drop=100*(r["metrics"]["stage_analysis"]["accuracy_by_stage"][-1]-r["metrics"]["test_accuracy"])
            lines.append(f"Seed {r['job']['seed']} lost {drop:.2f} pp versus its own full-depth test accuracy; the selected policy was not changed afterward.")
        lines.append("")
    else:
        lines += ["Pending.", ""]
    lines += ["## 7. Stage rescue/damage analysis", ""]
    if adaptive:
        lines += ["Counts below are means across seeds on the same 10,000 official test images; these are not independent new datasets.", "",
                  "| Transition | Rescued mean ± SD | Damaged mean ± SD | Incorrect→incorrect mean | Correct→correct mean |",
                  "|---|---:|---:|---:|---:|"]
        for s in range(7):
            rows = [r["metrics"]["stage_analysis"]["transitions"][s] for r in adaptive]
            lines.append(f"| {s+1}→{s+2} | {mean_std(summary(x['rescued'] for x in rows),digits=1)} | "
                         f"{mean_std(summary(x['damaged'] for x in rows),digits=1)} | "
                         f"{statistics.mean(x['incorrect_to_incorrect'] for x in rows):.1f} | {statistics.mean(x['correct_to_correct'] for x in rows):.1f} |")
        lines += ["", "| Seed | Stage1 correct | Stage1 errors eventually correct at final | Errors ever corrected | Stage1 correct broken at final | Correct ever broken |",
                  "|---|---:|---:|---:|---:|---:|"]
        for r in adaptive:
            a = r["metrics"]["stage_analysis"]
            lines.append(f"| {r['job']['seed']} | {a['stage1_correct']} | {a['stage1_errors_correct_at_final']} | "
                         f"{a['stage1_errors_ever_corrected']} | {a['stage1_correct_wrong_at_final']} | {a['stage1_correct_ever_broken']} |")
        lines.append("")
    else:
        lines += ["Pending.", ""]
    lines += ["## 8. 16x16 MNIST: no dense input projection", ""] + model_table(aggregates["mnist/direct16"])
    direct = aggregates["mnist/direct16"]
    if direct:
        lines += ["| Model | Mean core ops | Mean whole-model ops |", "|---|---:|---:|"]
        for name, row in direct.items():
            lines.append(f"| {name} | {row['core_ops']['mean']:.1f} | {row['whole_ops']['mean']:.1f} |")
        lines.append("")
        if "rotation" in direct and "adaptive-rotation" in direct:
            extra=100*(direct["adaptive-rotation"]["whole_ops"]["mean"]/direct["rotation"]["whole_ops"]["mean"]-1)
            lines += [f"**Direct-input adaptive execution used {extra:.2f}% more whole-model multiply/add arithmetic than fixed Rotation**, "
                      "despite fewer stages, and achieved lower mean accuracy. The classifier is four times the arithmetic cost of one rotation/affine stage.", ""]
    lines += ["Fixed bilinear antialiased resizing; flatten directly to 256. No input projection or input GELU remains. "
              "All models share this representation. Adaptive uses the preset 0.90 policy. "
              "These results are separate from 28x28; resolution and model capacity both change.", "",
              "## 9. Runtime optimization", "",
              "Inference-only copies cache sin/cos and reuse adaptive exit confidences. Learned mathematics and original weights are preserved. "
              "All-stage logits on every official test image must match at atol=1e-6, rtol=1e-5; predictions/exits must match exactly. "
              "Five implementations are measured in seeded shuffled order within each timing repetition, at batch 1 and 256. "
              "Compilation is optional and not attempted; this comparison measures the portable eager cache path.", ""]
    runtime = completed(suite, "mnist", "runtime")
    if runtime:
        lines += ["| Implementation | Seeds | Mean batch256 ms/sample ± SD | Mean batch1 ms/sample ± SD |", "|---|---:|---:|---:|"]
        for name in runtime[0]["metrics"]["benchmarks"]:
            lines.append(f"| {name} | {len(runtime)} | "
                         f"{mean_std(summary(r['metrics']['benchmarks'][name]['batched']['median_ms_per_sample'] for r in runtime),digits=5)} | "
                         f"{mean_std(summary(r['metrics']['benchmarks'][name]['single_sample']['median_ms_per_sample'] for r in runtime),digits=5)} |")
        max_error = max(v["max_absolute_logit_error"] for r in runtime for v in r["metrics"]["verification"].values())
        lines += ["", f"Maximum observed absolute logit error across verified test passes: {max_error:.9g}. "
                  "Exact exit and predicted-class agreement passed for completed runtime cases. "
                  "Seed-42 CPU operator profiles (including calls/time for trig, GELU, indexing and matrix multiplies) are saved per batch size.", ""]
        for kind in ("rotation","adaptive"):
            original=statistics.mean(r["metrics"]["benchmarks"][f"original-{kind}"]["single_sample"]["median_ms_per_sample"] for r in runtime)
            optimized=statistics.mean(r["metrics"]["benchmarks"][f"optimized-{kind}"]["single_sample"]["median_ms_per_sample"] for r in runtime)
            direction="faster" if optimized < original else "slower"
            percent=100*abs(optimized/original-1)
            lines += [f"Cached {kind} was {percent:.2f}% {direction} at batch=1. This optimization attempt is retained regardless of the outcome.", ""]
        lines += ["Profiling diagnosis: fixed Rotation still makes hundreds of elementwise calls, repeated stacks/concatenations and nine GELUs including the input activation. "
                  "Caching removes sin/cos but leaves those costs. The adaptive cache path adds indexing/allocation to save selected confidences; "
                  "seed-42 profiles show 480 versus 400 index calls at batch=256 and 60 versus 50 at batch=1. "
                  "These instrumented ten-forward profiles diagnose overhead; they are not the latency benchmark itself.", ""]
    else:
        lines += ["Pending.", ""]
    lines += ["## 10. Failures and negative findings", ""]
    for name, failure in suite["dataset_failures"].items():
        lines.append(f"- {name}: BLOCKED. {failure['reason']}. {failure['error']}")
    for r in suite["cases"].values():
        if r["status"] == "failed":
            lines.append(f"- FAILED `{r['path']}`: {r['error']}. Original attempt/error log preserved.")
        for attempt in r.get("previous_attempts", []):
            if attempt["status"] in ("failed", "interrupted", "running"):
                lines.append(f"- Earlier {attempt['status'].upper()} attempt `{attempt['path']}`: "
                             f"{attempt.get('error') or 'worker stopped before completion'}. "
                             "Its artifacts remain; only the completed replacement is included in aggregates.")
    for finding in suite.get("implementation_findings",[]):
        lines.append("- "+finding)
    if not suite["dataset_failures"] and not any(r["status"] == "failed" for r in suite["cases"].values()):
        lines.append("No recorded execution failures; numerical and methodological limitations still apply.")
    lines += ["", "Evidence-based answers to the seven research questions:", ""]
    for number, answer in answers.items():
        lines.append(f"{number}. {answer}")
    lines += ["", "## 11. Methodological limitations", "",
              "- Five seeds use the same official test set. Sample SD reflects initialization/training variation, not independent dataset uncertainty.",
              "- No significance test, equivalence claim, or novelty claim. Small mean differences should not be treated as established superiority.",
              "- Fixed five-epoch budgets may disadvantage deep stacks; tuning each model is deliberately excluded here.",
              "- Replication has 60,000 training images; validation experiments have 55,000. Their outcomes are not pooled.",
              "- Validation searches a finite preset threshold grid; it can overfit validation and does not guarantee the test tolerance.",
              "- Confidence is uncalibrated; later error rescue is descriptive, not an independent definition of difficulty.",
              "- Arithmetic excludes nonlinearities, trigonometry, allocation, softmax, gating and memory. Whole-model counts include repeated classifiers.",
              "- The identity control has redundant zero-angle arithmetic; coordinate-only removes it and supplies the true no-mixing depth control.",
              "- 16x16 changes resolution as well as removing the learned projection; it is a separate controlled comparison, not a direct 28x28 accuracy claim.",
              "- Timing includes final output selection and adaptive gating; it excludes loading, transfers, CSV and optimizer work. Default timed prefix is 1,000 fixed test samples.",
              "- CPU cache/thermal/background-load variation and kernel/library specificity remain; timer repeats are not independent training seeds.",
              "- Interrupted attempts are retained but excluded from accuracy/timing aggregates; completed replacement runs restart the same deterministic five-epoch budget.",
              "- Fashion-MNIST is unexecuted when marked blocked. No conclusion beyond MNIST is supported.", "",
              "## 12. Recommended Phase 3", "",
              "First run this unchanged protocol on cached Fashion-MNIST and one harder dataset. Predefine an accuracy-retention margin and a meaningful latency target. "
              "Prioritize comparisons against projection-only, coordinate-depth and low-rank controls. "
              "If mixing remains useful without a dense input projection, implement a genuinely fused CPU/GPU structured kernel, "
              "verify numerical/exit equivalence, and compare both accuracy and end-to-end latency over more hardware and training budgets. "
              "If no-mixing controls match RotNet or direct-input accuracy collapses, narrow or stop the architectural claim.", "",
              "## Reproduction and preservation", "", "```powershell",
              ".\\.venv\\Scripts\\python.exe -m pytest -q -p no:cacheprovider --basetemp .phase2-tests-reproduction",
              ".\\.venv\\Scripts\\python.exe scripts\\run_phase2.py --run-dir results\\phase2\\reproduction-five-seeds", "```", "",
              "Use a new run directory for fresh training. To resume the measured run and skip completed cases:", "", "```powershell",
              f".\\.venv\\Scripts\\python.exe scripts\\run_phase2.py --run-dir \"{run_dir}\" --resume", "```", "",
              "Completed cases are skipped on resume. Failures retain original attempts; --retry-failed creates a new numbered attempt. "
              "Phase 1 results and source files are protected by a before/after SHA-256 manifest.", "",
              f"Phase 1 preservation: {suite.get('phase1_preservation', {'status':'verification pending'})}.", ""]
    validation_results=suite.get("validation_results")
    if validation_results:
        lines += [f"Final tests: **{validation_results['tests_passed']} passed**, including all "
                  f"{validation_results['original_tests_passed']} original tests. Test command and XML evidence are saved in validation.json/test-results.xml.", ""]
    if suite.get("resume_source_snapshots"):
        lines += ["Resume source snapshots are saved. Resumes reject changes to model mathematics, training, data or evaluation code; "
                  "runner/reporting recovery changes are recorded separately from the original source snapshot.", ""]
    report = "\n".join(lines)
    (run_dir / "PHASE2_RESULTS.md").write_text(report, encoding="utf-8")
    if root_report is not None and not suite["settings"]["smoke"]:
        root_report.write_text(report, encoding="utf-8")
    if suite["status"] != "running":
        from .plots import create_phase2_plots
        create_phase2_plots(suite, run_dir / "plots")
