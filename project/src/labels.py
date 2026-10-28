"""Entity labels and mappings for Kannada-Telugu NER."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

# Exactly 7 labels in fixed order
LABELS: List[str] = [
    "O",
    "B-PER",
    "I-PER",
    "B-LOC",
    "I-LOC",
    "B-ORG",
    "I-ORG",
]

LABEL2ID: Dict[str, int] = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL: Dict[int, str] = {idx: label for idx, label in enumerate(LABELS)}


def get_labels() -> List[str]:
    """Return the fixed list of 7 NER labels."""
    return list(LABELS)


def get_label2id() -> Dict[str, int]:
    """Return label-to-id mapping."""
    return dict(LABEL2ID)


def get_id2label() -> Dict[int, str]:
    """Return id-to-label mapping."""
    return dict(ID2LABEL)


def save_labels(filepath: str | Path) -> None:
    """Save label configurations to JSON format for interoperability.

    Args:
        filepath: Target JSON file path.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "labels": LABELS,
        "label2id": LABEL2ID,
        "id2label": {str(k): v for k, v in ID2LABEL.items()},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_labels(filepath: str | Path) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    """Load labels and mappings from JSON file.

    Args:
        filepath: Path to labels.json.

    Returns:
        Tuple of (labels, label2id, id2label).
    """
    path = Path(filepath)
    if not path.exists():
        # Fallback to defaults and persist
        save_labels(path)
        return list(LABELS), dict(LABEL2ID), dict(ID2LABEL)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    labels = data["labels"]
    label2id = {k: int(v) for k, v in data["label2id"].items()}
    id2label = {int(k): v for k, v in data["id2label"].items()}
    return labels, label2id, id2label
