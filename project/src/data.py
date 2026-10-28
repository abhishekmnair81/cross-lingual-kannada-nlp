"""Data loading, validation, Unicode normalization, overlap checking, and HF alignment."""

from __future__ import annotations

import logging
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import PreTrainedTokenizerFast

from .labels import LABEL2ID, get_label2id, get_labels
from .utils import find_project_root, resolve_path

logger = logging.getLogger(__name__)

VALID_LABELS_SET = set(get_labels())


def load_raw_ner_csv(
    file_path: str | Path,
    expected_mismatches: int = 0,
) -> Tuple[List[Dict[str, Any]], int, List[Tuple[int, int, int]]]:
    """Load and validate an NER CSV file following strict data pipeline rules.

    Rules applied:
    1. Loaded with pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False).
    2. Split with .split(), validate len(tokens) == len(tags). Drop mismatched rows and log count.
    3. Sanitize known pandas NaN artifacts ('nan' tag -> 'O') and validate all tags in 7-label set.
    4. Apply unicodedata.normalize('NFC') to tokens, preserving tags intact.

    Args:
        file_path: Path to the CSV file.
        expected_mismatches: Expected number of dropped rows for sanity logging.

    Returns:
        Tuple of (records, dropped_count, dropped_info_list).
    """
    path = resolve_path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {path}")

    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)
    if "tokens" not in df.columns or "tags" not in df.columns:
        raise ValueError(f"File {path.name} must contain 'tokens' and 'tags' columns.")

    tokens_col = df["tokens"].tolist()
    tags_col = df["tags"].tolist()

    records: List[Dict[str, Any]] = []
    dropped_info: List[Tuple[int, int, int]] = []
    label2id = get_label2id()

    for idx, (raw_tokens_str, raw_tags_str) in enumerate(zip(tokens_col, tags_col)):
        # Apply NFC Unicode normalization to tokens string
        norm_tokens_str = unicodedata.normalize("NFC", raw_tokens_str)

        tokens = norm_tokens_str.split()
        tags = raw_tags_str.split()

        # Rule 2: Token-tag length parity check
        if len(tokens) != len(tags):
            dropped_info.append((idx, len(tokens), len(tags)))
            continue

        # Rule 3: Validate and sanitize tags
        cleaned_tags: List[str] = []
        ner_tags: List[int] = []
        for tag in tags:
            # Handle known artifact: row 150 of kn_preprocessed_test
            if tag == "nan":
                tag = "O"
            if tag not in VALID_LABELS_SET:
                raise ValueError(
                    f"Invalid tag '{tag}' encountered in {path.name} row {idx}. "
                    f"Must be one of {VALID_LABELS_SET}."
                )
            cleaned_tags.append(tag)
            ner_tags.append(label2id[tag])

        records.append(
            {
                "id": str(idx),
                "sentence_str": norm_tokens_str,
                "tokens": tokens,
                "tags": cleaned_tags,
                "ner_tags": ner_tags,
            }
        )

    dropped_count = len(dropped_info)
    logger.info(
        "Loaded %s: %d valid rows, %d dropped (expected: %d).",
        path.name,
        len(records),
        dropped_count,
        expected_mismatches,
    )
    if dropped_info:
        for idx_d, n_tok, n_tag in dropped_info:
            logger.info("  Dropped row %d: %d tokens vs %d tags", idx_d, n_tok, n_tag)

    return records, dropped_count, dropped_info


def compute_dataset_validation_stats(
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute and print validation table for all splits."""
    splits = {
        "kn_train": (config["data"]["kn_train_path"], 0),
        "kn_val": (config["data"]["kn_val_path"], 0),
        "kn_test": (config["data"]["kn_test_path"], 0),
        "te_val": (config["data"]["te_val_path"], 0),
        "te_test": (config["data"]["te_test_path"], 3),
    }

    stats: Dict[str, Any] = {}
    train_sentences: Set[str] = set()

    for split_name, (rel_path, exp_drop) in splits.items():
        records, dropped_count, dropped_info = load_raw_ner_csv(rel_path, exp_drop)
        sentences = [r["sentence_str"] for r in records]

        stats[split_name] = {
            "path": str(rel_path),
            "raw_rows": len(records) + dropped_count,
            "dropped_rows": dropped_count,
            "valid_rows": len(records),
            "records": records,
            "sentences": sentences,
        }

        if split_name == "kn_train":
            train_sentences = set(sentences)

    # Overlap detection with kn_train
    val_overlap = sum(1 for s in stats["kn_val"]["sentences"] if s in train_sentences)
    test_overlap = sum(1 for s in stats["kn_test"]["sentences"] if s in train_sentences)
    stats["kn_val"]["train_overlap"] = val_overlap
    stats["kn_test"]["train_overlap"] = test_overlap

    # Print markdown table
    print("\n" + "=" * 80)
    print("indicNER DATASET VALIDATION SUMMARY")
    print("=" * 80)
    print(
        f"{'Split':<12} | {'Raw Rows':<10} | {'Dropped':<8} | {'Valid Rows':<10} | {'Train Overlap':<14}"
    )
    print("-" * 80)
    for name, s in stats.items():
        overlap_str = str(s.get("train_overlap", "-"))
        print(
            f"{name:<12} | {s['raw_rows']:<10} | {s['dropped_rows']:<8} | {s['valid_rows']:<10} | {overlap_str:<14}"
        )
    print("=" * 80 + "\n")

    return stats


def filter_train_overlaps(
    records: List[Dict[str, Any]],
    train_sentences: Set[str],
) -> List[Dict[str, Any]]:
    """Filter out sentences that match any training sentence exactly."""
    return [r for r in records if r["sentence_str"] not in train_sentences]


def align_labels_with_tokens(
    examples: Dict[str, List[Any]],
    tokenizer: PreTrainedTokenizerFast,
    max_length: int = 128,
) -> Dict[str, Any]:
    """Align word-level NER labels with subword tokens from fast tokenizer.

    Rules:
    - First subword of each word gets the word's label ID.
    - Continuation subwords and special tokens ([CLS], [SEP], [PAD]) get -100.
    - Truncation set to max_length.

    Args:
        examples: Batch of examples with 'tokens' and 'ner_tags'.
        tokenizer: Fast tokenizer instance.
        max_length: Maximum sequence length in subwords.

    Returns:
        Tokenized inputs dictionary with 'labels' key added.
    """
    tokenized_inputs = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
    )

    all_labels = []
    for i, tags in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                # Special token ([CLS], [SEP], etc.)
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # First subword of a word -> assign label
                label_ids.append(tags[word_idx])
            else:
                # Continuation subword -> assign -100
                label_ids.append(-100)
            previous_word_idx = word_idx
        all_labels.append(label_ids)

    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs


def map_subword_preds_to_words(
    words: List[str],
    word_ids: List[Optional[int]],
    subword_preds: List[int],
    id2label: Dict[int, str],
) -> Tuple[List[str], bool]:
    """Map subword predictions back to word-level predictions.

    Rules:
    - Take the prediction at the first subword of each word.
    - If a sentence was truncated (words beyond max_length cut), missing words
      are assigned 'O'.
    - Output length always matches len(words).

    Args:
        words: List of gold words in sentence.
        word_ids: List of word index alignments from tokenizer.
        subword_preds: List of predicted label IDs at subword level.
        id2label: ID to label dictionary.

    Returns:
        Tuple of (word_level_predictions, is_truncated_flag).
    """
    num_words = len(words)
    word_preds = ["O"] * num_words
    seen_words: Set[int] = set()

    for word_idx, pred_id in zip(word_ids, subword_preds):
        if word_idx is not None and word_idx not in seen_words and word_idx < num_words:
            word_preds[word_idx] = id2label[pred_id]
            seen_words.add(word_idx)

    is_truncated = len(seen_words) < num_words
    return word_preds, is_truncated


def get_tokenized_dataset(
    tokenizer: PreTrainedTokenizerFast,
    config: Dict[str, Any],
    max_train_samples: Optional[int] = None,
    force_rebuild: bool = False,
) -> Tuple[DatasetDict, Dict[str, Any]]:
    """Build or load tokenized datasets for training and evaluation.

    Args:
        tokenizer: Fast tokenizer instance.
        config: Configuration dictionary.
        max_train_samples: Optional limit on training examples.
        force_rebuild: If True, ignore disk cache and rebuild.

    Returns:
        Tuple of (DatasetDict, raw_stats_dict).
    """
    cache_dir = resolve_path(config["data"]["cache_dir"])
    max_length = config["model"].get("max_length", 128)

    cache_suffix = f"_debug_{max_train_samples}" if max_train_samples else "_full"
    target_cache = cache_dir / f"tokenized{cache_suffix}"

    if target_cache.exists() and not force_rebuild:
        logger.info("Loading cached tokenized dataset from: %s", target_cache)
        dataset_dict = load_from_disk(str(target_cache))
        # Re-fetch stats quickly
        stats = compute_dataset_validation_stats(config)
        return dataset_dict, stats

    logger.info("Building tokenized datasets from raw CSV files...")
    stats = compute_dataset_validation_stats(config)

    train_records = stats["kn_train"]["records"]
    if max_train_samples and max_train_samples < len(train_records):
        logger.info("Sampling %d training records for debug/smoke-test", max_train_samples)
        train_records = train_records[:max_train_samples]

    train_sentences = set(stats["kn_train"]["sentences"])

    # Validation: full and train-overlap-removed (dedup)
    kn_val_full_records = stats["kn_val"]["records"]
    kn_val_dedup_records = filter_train_overlaps(kn_val_full_records, train_sentences)

    # Test: full and train-overlap-removed (dedup)
    kn_test_full_records = stats["kn_test"]["records"]
    kn_test_dedup_records = filter_train_overlaps(kn_test_full_records, train_sentences)

    # Telugu splits
    te_val_records = stats["te_val"]["records"]
    te_test_records = stats["te_test"]["records"]

    logger.info(
        "Split counts for pipeline:\n"
        "  kn_train: %d\n"
        "  kn_val_dedup (for model selection): %d\n"
        "  kn_val_full: %d\n"
        "  kn_test_full: %d\n"
        "  kn_test_dedup: %d\n"
        "  te_val: %d\n"
        "  te_test: %d",
        len(train_records),
        len(kn_val_dedup_records),
        len(kn_val_full_records),
        len(kn_test_full_records),
        len(kn_test_dedup_records),
        len(te_val_records),
        len(te_test_records),
    )

    def to_hf_dataset(recs: List[Dict[str, Any]]) -> Dataset:
        data_dict = {
            "id": [r["id"] for r in recs],
            "sentence_str": [r["sentence_str"] for r in recs],
            "tokens": [r["tokens"] for r in recs],
            "tags": [r["tags"] for r in recs],
            "ner_tags": [r["ner_tags"] for r in recs],
        }
        ds = Dataset.from_dict(data_dict)
        tokenized_ds = ds.map(
            lambda batch: align_labels_with_tokens(batch, tokenizer, max_length=max_length),
            batched=True,
            desc="Aligning subwords and labels",
        )
        return tokenized_ds

    dataset_dict = DatasetDict(
        {
            "train": to_hf_dataset(train_records),
            "kn_val_dedup": to_hf_dataset(kn_val_dedup_records),
            "kn_val_full": to_hf_dataset(kn_val_full_records),
            "kn_test_full": to_hf_dataset(kn_test_full_records),
            "kn_test_dedup": to_hf_dataset(kn_test_dedup_records),
            "te_val": to_hf_dataset(te_val_records),
            "te_test": to_hf_dataset(te_test_records),
        }
    )

    target_cache.parent.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(target_cache))
    logger.info("Saved tokenized dataset to disk at: %s", target_cache)

    return dataset_dict, stats