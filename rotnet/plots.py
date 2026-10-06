"""Plots of measured data only, with visible scales and threshold labels."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def create_plots(metrics: dict, predictions: list[dict], output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    models = metrics["models"]
    paths = []

    def save(fig, name):
        fig.tight_layout()
        path = output_dir / name
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path.name))

    labels = list(models)
    if labels:
        fig, ax = plt.subplots(figsize=(8, 4))
        accuracies = [100 * models[name]["test_accuracy"] for name in labels]
        bars = ax.bar(labels, accuracies)
        ax.bar_label(bars, fmt="%.2f%%", padding=3)
        ax.set(ylim=(0, 105), ylabel="Test accuracy (%)", title="Accuracy by model (one training seed)")
        if "adaptive-rotation" in models:
            ax.set_xlabel(f"Adaptive model uses preset threshold {metrics['config']['threshold']:.2f}")
        save(fig, "accuracy_by_model.png")

        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(labels, [models[n]["parameters"]["core_trainable_parameters"] for n in labels])
        ax.bar_label(bars, fmt="%.0f", padding=3)
        ax.set(ylabel="Core trainable parameters", title="Hidden/core parameter count (zero-based axis)")
        ax.set_ylim(bottom=0)
        save(fig, "core_parameters_by_model.png")

    adaptive = models.get("adaptive-rotation")
    if adaptive:
        sweeps = adaptive["threshold_sweep"]
        stages = metrics["config"]["hidden_dim"].bit_length() - 1
        fig, ax = plt.subplots(figsize=(7, 4))
        x = [s["average_stages"] for s in sweeps]
        y = [100 * s["test_accuracy"] for s in sweeps]
        ax.scatter(x, y, label="Adaptive thresholds")
        for sweep, xx, yy in zip(sweeps, x, y):
            ax.annotate(f"{sweep['threshold']:.2f}", (xx, yy), xytext=(4, 4), textcoords="offset points")
        ax.scatter([stages], [100 * adaptive["full_depth"]["test_accuracy"]], marker="x", label="Final head, full depth")
        ax.set(xlim=(0.5, stages + 0.5), ylim=(0, 100), xlabel="Average executed stages per sample",
               ylabel="Test accuracy (%)", title="Adaptive accuracy versus stages (not wall time)")
        ax.legend()
        save(fig, "adaptive_accuracy_vs_stages.png")

        threshold = metrics["config"]["threshold"]
        rows = [r for r in predictions if r["model"] == "adaptive-rotation" and r["threshold"] == threshold]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(range(1, stages + 1), [sum(r["exit_stage"] == s for r in rows) for s in range(1, stages + 1)])
        ax.set(xticks=range(1, stages + 1), xlabel="Exit stage", ylabel="Number of test samples",
               title=f"Adaptive exits at threshold {threshold:.2f}")
        ax.set_ylim(bottom=0)
        save(fig, "adaptive_exit_histogram.png")

        fig, ax = plt.subplots(figsize=(7, 4))
        positions, values = [], []
        for stage in range(1, stages + 1):
            selected = [r["confidence"] for r in rows if r["exit_stage"] == stage]
            if selected:
                positions.append(stage)
                values.append(selected)
        if values:
            ax.boxplot(values, positions=positions, widths=0.55, showfliers=False)
        ax.axhline(threshold, color="gray", linestyle="--", label="Exit threshold")
        ax.set(xlim=(0.5, stages + 0.5), ylim=(0, 1.02), xticks=range(1, stages + 1),
               xlabel="Exit stage", ylabel="Maximum softmax probability at exit",
               title="Confidence is part of the stopping rule; association is expected")
        ax.legend()
        save(fig, "adaptive_confidence_vs_exit_stage.png")
    return paths
