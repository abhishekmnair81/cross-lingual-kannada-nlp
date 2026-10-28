"""Unit tests for token alignment and prediction truncation handling."""

import pytest
from transformers import AutoTokenizer

from src.data import align_labels_with_tokens, map_subword_preds_to_words
from src.labels import LABEL2ID, ID2LABEL


@pytest.fixture(scope="module")
def tokenizer():
    """Load fast tokenizer for MuRIL."""
    return AutoTokenizer.from_pretrained("google/muril-base-cased", use_fast=True)


def test_alignment_first_subword_and_continuation(tokenizer):
    """Test that:
    1. First subword of each word receives the word's label ID.
    2. Continuation subwords receive -100.
    3. Special tokens ([CLS], [SEP]) receive -100.
    """
    # Multi-subword Kannada words
    # ರಾಜನಾಥ್ -> [CLS], ರಾಜ, ##ನಾಥ್, ಸಿಂಗ್, [SEP]
    tokens = ["ರಾಜನಾಥ್", "ಸಿಂಗ್"]
    tags = ["B-PER", "I-PER"]
    ner_tags = [LABEL2ID[t] for t in tags]

    batch = {
        "tokens": [tokens],
        "ner_tags": [ner_tags],
    }

    tokenized = align_labels_with_tokens(batch, tokenizer, max_length=128)
    labels = tokenized["labels"][0]
    input_ids = tokenized["input_ids"][0]
    word_ids = tokenized.word_ids(batch_index=0)

    # First and last must be special tokens -> label -100
    assert word_ids[0] is None
    assert labels[0] == -100
    assert word_ids[-1] is None
    assert labels[-1] == -100

    # Inspect word 0 ('ರಾಜನಾಥ್')
    word_0_subword_indices = [i for i, w in enumerate(word_ids) if w == 0]
    assert len(word_0_subword_indices) >= 1

    # First subword must have the exact label ID
    first_sub_idx = word_0_subword_indices[0]
    assert labels[first_sub_idx] == LABEL2ID["B-PER"]

    # All continuation subwords of word 0 must have -100
    for cont_sub_idx in word_0_subword_indices[1:]:
        assert labels[cont_sub_idx] == -100

    # Inspect word 1 ('ಸಿಂಗ್')
    word_1_subword_indices = [i for i, w in enumerate(word_ids) if w == 1]
    assert len(word_1_subword_indices) >= 1
    assert labels[word_1_subword_indices[0]] == LABEL2ID["I-PER"]
    for cont_sub_idx in word_1_subword_indices[1:]:
        assert labels[cont_sub_idx] == -100


def test_truncated_sentence_prediction_length(tokenizer):
    """Test that a truncated sentence still yields predictions of the same length as gold.
    Words beyond the cut get predicted 'O'.
    """
    # Create a long sentence with 50 words
    words = [f"ಪದ_{i}" for i in range(50)]
    gold_tags = ["O"] * 50
    gold_tags[0] = "B-LOC"
    gold_tags[45] = "B-PER"

    # Tokenize with a very short max_length=10 to force heavy truncation
    max_len = 10
    enc = tokenizer(
        words,
        is_split_into_words=True,
        truncation=True,
        max_length=max_len,
    )

    word_ids = enc.word_ids()
    subword_preds = [0] * len(word_ids)  # dummy subword predictions (all O = 0)
    # Simulate predicting B-LOC (3) on the first token
    if len(word_ids) > 1 and word_ids[1] == 0:
        subword_preds[1] = LABEL2ID["B-LOC"]

    word_preds, is_truncated = map_subword_preds_to_words(
        words=words,
        word_ids=word_ids,
        subword_preds=subword_preds,
        id2label=ID2LABEL,
    )

    # 1. Output length MUST match original gold word count exactly
    assert len(word_preds) == len(words)
    # 2. Since max_length=10 < length of 50 words, is_truncated must be True
    assert is_truncated is True
    # 3. Truncated words beyond the cut must be 'O'
    assert word_preds[45] == "O"
    # 4. First word should retain its predicted tag
    assert word_preds[0] == "B-LOC"
