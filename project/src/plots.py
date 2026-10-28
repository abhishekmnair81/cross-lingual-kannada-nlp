"""Plotting module: Generates light-themed, publication-ready figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .utils import resolve_path

# Light pastel color palette as required
PALETTE = {
    "PER": "#93C5FD",     # light blue
    "LOC": "#86EFAC",     # light green
    "ORG": "#FDBA74",     # light orange
    "O": "#E5E7EB",       # light grey
    "accent": "#A78BFA",  # soft purple
    "bg": "#FFFFFF",      # pure white
    "grid": "#E5E7EB",    # subtle border/grid
    "text": "#1F2937",    # dark slate text
}


def apply_light_theme() -> None:
    """Apply consistent light, soft styling to Matplotlib."""
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        # Fallback if specific style name not present
        plt.style.use("default")

    plt.rcParams.update(
        {
            "figure.facecolor": PALETTE["bg"],
            "axes.facecolor": PALETTE["bg"],
            "grid.color": PALETTE["grid"],
            "grid.linestyle": "--",
            "grid.linewidth": 0.75,
            "text.color": PALETTE["text"],
            "axes.labelcolor": PALETTE["text"],
            "xtick.color": PALETTE["text"],
            "ytick.color": PALETTE["text"],
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial", "Helvetica"],
            "figure.dpi": 200,
            "savefig.dpi": 200,
            "savefig.facecolor": PALETTE["bg"],
            "savefig.edgecolor": PALETTE["bg"],
        }
    )


def plot_loss_curves(
    train_losses: List[float],
    val_losses: List[float],
    eval_steps: List[int],
    output_path: str | Path,
) -> None:
    """Plot training and validation loss progression over steps."""
    apply_light_theme()
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=200)

    if train_losses:
        steps = [i + 1 for i in range(len(train_losses))]
        ax.plot(steps, train_losses, label="Train Loss", color=PALETTE["accent"], linewidth=2.0)

    if val_losses and eval_steps:
        ax.plot(
            eval_steps[: len(val_losses)],
            val_losses,
            label="Val Loss",
            color=PALETTE["ORG"],
            marker="o",
            markersize=5,
            linewidth=2.0,
        )

    ax.set_title("Training and Validation Loss Curve (MuRIL Kannada NER)", fontsize=13, weight="bold", pad=12)
    ax.set_xlabel("Steps", fontsize=11, labelpad=8)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=11, labelpad=8)
    ax.legend(frameon=True, facecolor="#F8FAFC", edgecolor=PALETTE["grid"])
    fig.tight_layout()

    out = resolve_path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def plot_val_f1_curve(
    eval_steps: List[int],
    val_f1_scores: List[float],
    output_path: str | Path,
) -> None:
    """Plot validation Seqeval micro F1 curve."""
    apply_light_theme()
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=200)

    ax.plot(
        eval_steps[: len(val_f1_scores)],
        val_f1_scores,
        label="Validation F1 (Seqeval)",
        color=PALETTE["PER"],
        marker="s",
        markersize=6,
        linewidth=2.2,
    )

    if val_f1_scores:
        best_idx = int(np.argmax(val_f1_scores))
        best_step = eval_steps[best_idx]
        best_f1 = val_f1_scores[best_idx]
        ax.scatter([best_step], [best_f1], color="#EF4444", s=80, zorder=5, label=f"Best F1: {best_f1:.4f}")
        ax.annotate(
            f"Best: {best_f1:.4f}",
            (best_step, best_f1),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="#EFF6FF", ec=PALETTE["PER"], lw=1),
        )

    ax.set_title("Validation F1 Progression (Kannada Dedup Val)", fontsize=13, weight="bold", pad=12)
    ax.set_xlabel("Steps", fontsize=11, labelpad=8)
    ax.set_ylabel("Entity F1 Score", fontsize=11, labelpad=8)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=True, facecolor="#F8FAFC", edgecolor=PALETTE["grid"])
    fig.tight_layout()

    out = resolve_path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def plot_entity_distribution_train(
    label_counts: Dict[str, int],
    output_path: str | Path,
) -> None:
    """Bar chart of entity class counts in kn_train."""
    apply_light_theme()
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=200)

    # Consolidate IOB tags into entity types
    entity_types = ["PER", "LOC", "ORG", "O"]
    counts = [
        label_counts.get("B-PER", 0) + label_counts.get("I-PER", 0),
        label_counts.get("B-LOC", 0) + label_counts.get("I-LOC", 0),
        label_counts.get("B-ORG", 0) + label_counts.get("I-ORG", 0),
        label_counts.get("O", 0),
    ]
    colors = [PALETTE["PER"], PALETTE["LOC"], PALETTE["ORG"], PALETTE["O"]]

    bars = ax.bar(entity_types, counts, color=colors, edgecolor="#CBD5E1", width=0.55)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:,}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            weight="semibold",
        )

    ax.set_title("Entity Class Distribution in kn_preprocessed_train", fontsize=13, weight="bold", pad=12)
    ax.set_xlabel("Entity Category", fontsize=11, labelpad=8)
    ax.set_ylabel("Token Count", fontsize=11, labelpad=8)
    fig.tight_layout()

    out = resolve_path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def plot_per_class_f1_comparison(
    kannada_scores: Dict[str, float],
    telugu_scores: Dict[str, float],
    output_path: str | Path,
) -> None:
    """Bar chart comparing per-class and overall F1 for Kannada test vs Telugu zero-shot test."""
    apply_light_theme()
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=200)

    categories = ["Overall", "PER", "LOC", "ORG"]
    kn_vals = [kannada_scores.get(c, 0.0) for c in categories]
    te_vals = [telugu_scores.get(c, 0.0) for c in categories]

    x = np.arange(len(categories))
    width = 0.35

    rects1 = ax.bar(x - width / 2, kn_vals, width, label="Kannada (In-Language)", color=PALETTE["PER"], edgecolor="#93C5FD")
    rects2 = ax.bar(x + width / 2, te_vals, width, label="Telugu (Zero-Shot)", color=PALETTE["ORG"], edgecolor="#FDBA74")

    for rect in list(rects1) + list(rects2):
        height = rect.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            weight="bold",
        )

    ax.set_title("Named Entity Recognition F1: In-Language vs Cross-Lingual Zero-Shot", fontsize=12, weight="bold", pad=12)
    ax.set_ylabel("Seqeval F1 Score", fontsize=11, labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10, weight="semibold")
    ax.set_ylim(0, 1.1)
    ax.legend(frameon=True, facecolor="#F8FAFC", edgecolor=PALETTE["grid"])
    fig.tight_layout()

    out = resolve_path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def plot_confusion_matrix_heatmap(
    cm: np.ndarray,
    labels: List[str],
    output_path: str | Path,
    title: str = "Confusion Matrix (Entity Type Level)",
) -> None:
    """Confusion matrix heatmap using light Blues colormap."""
    apply_light_theme()
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=200)

    # Normalize row-wise for display percentage
    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)

    annot_matrix = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot_matrix[i, j] = f"{cm[i, j]:,}\n({cm_norm[i, j]:.1%})"

    sns.heatmap(
        cm_norm,
        annot=annot_matrix,
        fmt="",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        ax=ax,
        linewidths=0.5,
        linecolor="#E2E8F0",
    )

    ax.set_title(title, fontsize=12, weight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=10, labelpad=8)
    ax.set_ylabel("Gold Label", fontsize=10, labelpad=8)
    fig.tight_layout()

    out = resolve_path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def plot_subword_length_histogram(
    lengths: List[int],
    output_path: str | Path,
    cutoff: int = 128,
) -> None:
    """Histogram of subword sequence lengths with cutoff reference line."""
    apply_light_theme()
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=200)

    sns.histplot(
        lengths,
        bins=50,
        color=PALETTE["accent"],
        edgecolor="#FFFFFF",
        alpha=0.85,
        kde=True,
        ax=ax,
    )

    coverage = np.mean(np.array(lengths) <= cutoff) * 100
    ax.axvline(
        cutoff,
        color="#EF4444",
        linestyle="--",
        linewidth=2.0,
        label=f"Max Length Cutoff = {cutoff} ({coverage:.2f}% coverage)",
    )

    ax.set_title("Subword Sequence Length Distribution (google/muril-base-cased)", fontsize=12, weight="bold", pad=12)
    ax.set_xlabel("Sequence Length (Subwords)", fontsize=11, labelpad=8)
    ax.set_ylabel("Number of Sentences", fontsize=11, labelpad=8)
    ax.legend(frameon=True, facecolor="#F8FAFC", edgecolor=PALETTE["grid"])
    fig.tight_layout()

    out = resolve_path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
