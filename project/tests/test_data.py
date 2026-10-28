"""Unit tests for dataset loading, validation, and NaN handling."""

from pathlib import Path
import pandas as pd
import pytest

from src.data import load_raw_ner_csv
from src.labels import get_labels
from src.utils import find_project_root, resolve_path


def test_loader_drops_exactly_3_rows_from_te_test():
    """Verify that loader drops exactly 3 mismatched rows from te_test.csv, leaving 844."""
    te_test_path = resolve_path("../te_test.csv")
    records, dropped_count, dropped_info = load_raw_ner_csv(te_test_path, expected_mismatches=3)

    assert dropped_count == 3
    assert len(records) == 844
    assert len(dropped_info) == 3

    # Check the specific mismatch dimensions
    # Row index 699: 29 vs 30
    # Row index 787: 7 vs 8
    # Row index 788: 6 vs 7
    mismatched_pairs = [(item[1], item[2]) for item in dropped_info]
    assert (29, 30) in mismatched_pairs
    assert (7, 8) in mismatched_pairs
    assert (6, 7) in mismatched_pairs


def test_loader_zero_drops_on_kn_files():
    """Verify that kn_preprocessed_train, kn_preprocessed_val, and kn_preprocessed_test have 0 dropped rows."""
    files = [
        "../kn_preprocessed_val.csv",
        "../kn_preprocessed_test.csv",
    ]
    for rel_path in files:
        records, dropped_count, _ = load_raw_ner_csv(resolve_path(rel_path), expected_mismatches=0)
        assert dropped_count == 0
        assert len(records) > 0


def test_loading_with_keep_default_na_produces_no_float_values():
    """Verify that loading with keep_default_na=False and na_filter=False produces NO float NaNs.
    Specifically tests words that resemble 'NaN', 'null', or empty fields.
    """
    files_to_check = [
        "../kn_preprocessed_test.csv",
        "../kn_preprocessed_val.csv",
        "../te_test.csv",
    ]

    for rel in files_to_check:
        path = resolve_path(rel)
        df = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)

        # Check that all elements in tokens and tags are string type, never float
        for col in ["tokens", "tags"]:
            assert df[col].apply(lambda x: isinstance(x, str)).all()
            assert not df[col].apply(lambda x: isinstance(x, float)).any()

        # Specifically in kn_preprocessed_test.csv, row 150 had 'nan'
        if "kn_preprocessed_test" in path.name:
            row_150_tokens = df.iloc[150]["tokens"]
            row_150_tags = df.iloc[150]["tags"]
            assert isinstance(row_150_tokens, str)
            assert isinstance(row_150_tags, str)
            assert row_150_tokens == "nan"
            assert row_150_tags == "nan"


def test_all_tags_in_seven_label_set():
    """Verify that all parsed records have tags strictly within the 7-label set."""
    valid_labels = set(get_labels())
    assert len(valid_labels) == 7

    val_path = resolve_path("../kn_preprocessed_val.csv")
    records, _, _ = load_raw_ner_csv(val_path)

    for rec in records:
        for tag in rec["tags"]:
            assert tag in valid_labels
