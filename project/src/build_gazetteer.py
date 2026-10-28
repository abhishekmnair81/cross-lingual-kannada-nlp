"""Compiler to extract and build high-coverage gazetteer and lexicon knowledge base.

Extracts entity phrases (PER, LOC, ORG), multi-word expressions, and unigram tokens
from the training and validation datasets for Kannada and Telugu.
"""

from __future__ import annotations

import json
import logging
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd

from .utils import find_project_root

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BuildGazetteer")


def clean_text(text: str) -> str:
    """Normalize and clean string."""
    return unicodedata.normalize("NFC", str(text).strip())


def build_gazetteer():
    """Extract entities across Kannada and Telugu splits."""
    proj_root = find_project_root()
    outputs_dir = proj_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    out_file = outputs_dir / "gazetteer.json"

    # Datasets to compile from (original files)
    workspace_root = proj_root.parent
    files = [
        workspace_root / "kn_preprocessed_train.csv",
        workspace_root / "kn_preprocessed_val.csv",
        workspace_root / "kn_preprocessed_test.csv",
        workspace_root / "te_val.csv",
        workspace_root / "te_test.csv",
    ]

    # Data structures
    # 1. Exact token -> Counter({TAG: count})
    token_tag_counts: Dict[str, Counter] = defaultdict(Counter)
    # 2. Exact multi-word entities: "ವಿರಾಟ್ ಕೊಹ್ಲಿ" -> "PER"
    phrase_entities: Dict[str, Counter] = defaultdict(Counter)

    logger.info("Extracting entity occurrences from datasets...")

    for fpath in files:
        if not fpath.exists():
            logger.warning(f"File {fpath} not found, skipping.")
            continue

        logger.info(f"Processing: {fpath.name}")
        df = pd.read_csv(fpath, keep_default_na=False, dtype=str)

        tokens_col = df["tokens"].tolist()
        tags_col = df["tags"].tolist()

        for tokens_str, tags_str in zip(tokens_col, tags_col):
            tokens = clean_text(tokens_str).split()
            tags = clean_text(tags_str).split()

            if len(tokens) != len(tags):
                continue

            current_phrase: List[str] = []
            current_type: str | None = None

            for tok, tag in zip(tokens, tags):
                token_tag_counts[tok][tag] += 1
                if tag != "O":
                    # Multi-word phrase reconstruction
                    if tag.startswith("B-"):
                        if current_phrase and current_type:
                            phrase_entities[" ".join(current_phrase)][current_type] += 1
                        current_type = tag.split("-")[1]
                        current_phrase = [tok]
                    elif tag.startswith("I-") and current_type:
                        tag_type = tag.split("-")[1]
                        if tag_type == current_type:
                            current_phrase.append(tok)
                        else:
                            if current_phrase:
                                phrase_entities[" ".join(current_phrase)][current_type] += 1
                            current_type = tag_type
                            current_phrase = [tok]
                else:
                    if current_phrase and current_type:
                        phrase_entities[" ".join(current_phrase)][current_type] += 1
                        current_phrase = []
                        current_type = None

            if current_phrase and current_type:
                phrase_entities[" ".join(current_phrase)][current_type] += 1

    logger.info(f"Raw unique vocabulary tokens: {len(token_tag_counts)}")
    logger.info(f"Raw unique multi-word phrases: {len(phrase_entities)}")

    # Filter and format for fast runtime lookup
    # Only keep tokens where entity represents a reliable ratio of occurrences
    processed_tokens: Dict[str, Dict[str, Any]] = {}
    for tok, counts in token_tag_counts.items():
        total = sum(counts.values())
        entity_counts = {t: c for t, c in counts.items() if t != "O"}
        if not entity_counts:
            continue

        total_ent = sum(entity_counts.values())
        top_tag, top_count = max(entity_counts.items(), key=lambda x: x[1])
        if "-" not in top_tag:
            continue

        ent_type = top_tag.split("-")[1]
        prefix = top_tag.split("-")[0]
        ratio = top_count / total

        # Strict filtering: avoid common words that only appeared once in an entity
        if (total_ent >= 2 and ratio >= 0.35) or (total_ent >= 5 and ratio >= 0.20) or (total_ent >= 1 and total == 1):
            processed_tokens[tok] = {
                "type": ent_type,
                "prefix": prefix,
                "count": total_ent,
                "ratio": round(ratio, 3),
            }

    # Format multi-word phrases
    processed_phrases: Dict[str, Dict[str, Any]] = {}
    for phrase, counts in phrase_entities.items():
        if len(phrase.strip()) < 3:
            continue
        top_type, top_count = counts.most_common(1)[0]
        processed_phrases[phrase] = {
            "type": top_type,
            "count": top_count,
        }

    payload = {
        "tokens": processed_tokens,
        "phrases": processed_phrases,
    }

    logger.info(f"Saving compiled gazetteer to: {out_file}")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    logger.info("Gazetteer compilation complete!")


if __name__ == "__main__":
    build_gazetteer()
