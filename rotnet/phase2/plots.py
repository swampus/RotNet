"""Descriptive plots; accuracy axes show 0..100 and all data remain exported."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .report import completed, groups, summary


def create_phase2_plots(suite, directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    outputs = []

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(directory / name, dpi=150)
        plt.close(fig)
        outputs.append(name)

    for dataset in ("mnist", "fashion-mnist"):
        group = groups(suite, dataset, "replication")
        if not group:
            continue
        names = list(group)
        accuracy = [100 * group[n]["accuracy"]["mean"] for n in names]
        std = [100 * (group[n]["accuracy"]["std"] or 0) for n in names]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(names, accuracy, yerr=std, capsize=5)
        ax.set(ylim=(0, 100), ylabel="Test accuracy (%)", title=f"{dataset}: mean ± sample SD across seeds")
        save(fig, f"{dataset}_accuracy_mean_std.png")
        for axis, field, xlabel in (("core_ops", "core_ops", "Core scalar multiply/add operations per sample"),
                                     ("real_latency", "batched_ms", "Batch=256 milliseconds per sample")):
            fig, ax = plt.subplots(figsize=(7, 4))
            for name in names:
                row = group[name]
                ax.errorbar(row[field]["mean"], 100 * row["accuracy"]["mean"], yerr=100 * (row["accuracy"]["std"] or 0),
                            fmt="o", capsize=4, label=name)
            ax.set(xlabel=xlabel, ylabel="Test accuracy (%)", ylim=(0, 100), title=f"{dataset}: accuracy versus {axis}")
            ax.set_xlim(left=0)
            ax.legend()
            save(fig, f"{dataset}_accuracy_vs_{axis}.png")
    adaptive = [r for r in completed(suite, "mnist", "replication") if r["job"]["model"] == "adaptive-rotation"]
    if adaptive:
        accuracy = np.array([r["metrics"]["stage_analysis"]["accuracy_by_stage"] for r in adaptive]) * 100
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.errorbar(range(1, 9), accuracy.mean(0), yerr=accuracy.std(0, ddof=1) if len(adaptive) > 1 else None, fmt="o-", capsize=4)
        ax.set(xticks=range(1, 9), ylim=(0, 100), xlabel="Fixed stage (same trained adaptive weights)", ylabel="Accuracy (%)")
        save(fig, "mnist_adaptive_accuracy_by_stage.png")
        fig, ax = plt.subplots(figsize=(8, 4))
        for offset, label in ((-0.18, "rescued"), (0.18, "damaged")):
            counts = np.array([[t[label] for t in r["metrics"]["stage_analysis"]["transitions"]] for r in adaptive])
            ax.bar(np.arange(1, 8) + offset, counts.mean(0), width=0.36,
                   yerr=counts.std(0, ddof=1) if len(adaptive) > 1 else None, capsize=3, label=label)
        ax.set(xticks=range(1, 8), xticklabels=[f"{s}→{s+1}" for s in range(1, 8)],
               ylabel="Mean samples per seed ± SD", title="MNIST stage transitions")
        ax.legend()
        save(fig, "mnist_rescue_damage_by_stage.png")
    validation = completed(suite, "mnist", "validation")
    if validation:
        exits = np.array([[r["metrics"]["exit_distribution"][str(s)] for s in range(1, 9)] for r in validation])
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(range(1, 9), exits.mean(0), yerr=exits.std(0, ddof=1) if len(validation) > 1 else None, capsize=4)
        ax.set(xticks=range(1, 9), xlabel="Exit stage", ylabel="Mean test sample count ± SD", title="Frozen validation-selected policies")
        save(fig, "mnist_validation_selected_exits.png")
        fig, ax = plt.subplots(figsize=(8, 4))
        for r in validation:
            m = r["metrics"]
            candidates = m["selection"]["candidates"]
            ax.plot([c["validation_mean_stages"] for c in candidates], [100*c["validation_accuracy"] for c in candidates], alpha=0.25, color="gray")
            ax.scatter(m["average_stages"], 100*m["test_accuracy"], label=f"Frozen policy, seed {r['job']['seed']}")
        ax.set(xlim=(0.5, 8.5), ylim=(0, 100), xlabel="Mean stages", ylabel="Accuracy (%)",
               title="Gray: validation candidates; points: frozen-policy test results")
        ax.legend(fontsize=8)
        save(fig, "mnist_validation_selected_accuracy_vs_stages.png")
    direct = groups(suite, "mnist", "direct16")
    if direct:
        fig, ax = plt.subplots(figsize=(8, 4))
        names = list(direct)
        x = np.arange(len(names))
        ax.bar(x - 0.18, [direct[n]["core_ops"]["mean"] for n in names], width=0.36, label="Core arithmetic")
        ax.bar(x + 0.18, [direct[n]["whole_ops"]["mean"] for n in names], width=0.36, label="Whole model arithmetic")
        ax.set(xticks=x, xticklabels=names, ylabel="Scalar multiply/add operations per sample", title="16x16 MNIST, no learned input projection")
        ax.legend()
        save(fig, "mnist_16x16_whole_model_compute.png")
    return outputs
