"""CLI prediction interface for Kannada and Telugu NER inference."""

from __future__ import annotations

import argparse
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForTokenClassification, AutoTokenizer

from .data import map_subword_preds_to_words
from .gazetteer import extract_entities_with_gazetteer
from .labels import ID2LABEL, get_id2label
from .utils import find_project_root, get_device, resolve_path


def predict_text(
    text: str,
    model: AutoModelForTokenClassification,
    tokenizer: Any,
    device: torch.device,
    id2label: Dict[int, str],
) -> Dict[str, Any]:
    """Run inference on a raw Kannada or Telugu sentence.

    Args:
        text: Input string.
        model: Token classification model.
        tokenizer: Fast tokenizer.
        device: Torch device.
        id2label: Label ID to string map.

    Returns:
        Dictionary with tokens, predicted tags, confidence scores, and entity spans.
    """
    model.eval()
    norm_text = unicodedata.normalize("NFC", text.strip())
    words = norm_text.split()

    if not words:
        return {"tokens": [], "tags": [], "confidences": [], "entities": []}

    encodings = tokenizer(
        words,
        is_split_into_words=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[0]  # (seq_len, num_labels)
        probs = F.softmax(logits, dim=-1)
        subword_preds = torch.argmax(probs, dim=-1).cpu().tolist()
        subword_confs = torch.max(probs, dim=-1).values.cpu().tolist()

    word_ids = encodings.word_ids(batch_index=0)

    # Word-level mapping
    num_words = len(words)
    word_preds = ["O"] * num_words
    word_confs = [1.0] * num_words
    seen_words = set()

    for sub_idx, (w_idx, p_id) in enumerate(zip(word_ids, subword_preds)):
        if w_idx is not None and w_idx not in seen_words and w_idx < num_words:
            word_preds[w_idx] = id2label[p_id]
            word_confs[w_idx] = float(subword_confs[sub_idx])
            seen_words.add(w_idx)

    # Refine predictions with Dravidian Gazetteer & Morphological Engine
    word_preds, word_confs = extract_entities_with_gazetteer(words, word_preds, word_confs)

    # Extract entity spans
    entities: List[Dict[str, Any]] = []
    current_entity: Dict[str, Any] | None = None

    for idx, (word, tag, conf) in enumerate(zip(words, word_preds, word_confs)):
        if tag.startswith("B-"):
            if current_entity:
                entities.append(current_entity)
            ent_type = tag.split("-")[1]
            current_entity = {
                "entity": ent_type,
                "words": [word],
                "start_idx": idx,
                "end_idx": idx + 1,
                "confidences": [conf],
            }
        elif tag.startswith("I-") and current_entity:
            ent_type = tag.split("-")[1]
            if current_entity["entity"] == ent_type:
                current_entity["words"].append(word)
                current_entity["end_idx"] = idx + 1
                current_entity["confidences"].append(conf)
            else:
                entities.append(current_entity)
                current_entity = {
                    "entity": ent_type,
                    "words": [word],
                    "start_idx": idx,
                    "end_idx": idx + 1,
                    "confidences": [conf],
                }
        else:
            if current_entity:
                entities.append(current_entity)
                current_entity = None

    if current_entity:
        entities.append(current_entity)

    # Format entity objects
    formatted_entities = []
    for ent in entities:
        formatted_entities.append(
            {
                "text": " ".join(ent["words"]),
                "type": ent["entity"],
                "start_token": ent["start_idx"],
                "end_token": ent["end_idx"],
                "confidence": float(sum(ent["confidences"]) / len(ent["confidences"])),
            }
        )

    return {
        "text": norm_text,
        "tokens": words,
        "tags": word_preds,
        "confidences": [round(c, 4) for c in word_confs],
        "entities": formatted_entities,
    }


def main():
    parser = argparse.ArgumentParser(description="Predict NER tags for input text")
    parser.add_argument("--text", type=str, required=True, help="Input sentence in Kannada or Telugu")
    parser.add_argument("--model_path", type=str, default=None, help="Path to fine-tuned model checkpoint")
    args = parser.parse_args()

    # Reconfigure stdout for UTF-8 on Windows
    sys.stdout.reconfigure(encoding="utf-8")

    proj_root = find_project_root()
    default_ckpt = proj_root / "outputs" / "checkpoints" / "best_model"
    model_dir = Path(args.model_path) if args.model_path else default_ckpt

    if not model_dir.exists():
        # Fallback to backbone if checkpoint not yet trained
        model_name = "google/muril-base-cased"
        print(f"Notice: Checkpoint {model_dir} not found. Using pretrained backbone '{model_name}'.")
    else:
        model_name = str(model_dir)

    device = get_device()
    id2label = get_id2label()

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(id2label),
        id2label=id2label,
        label2id={v: k for k, v in id2label.items()},
    ).to(device)

    result = predict_text(args.text, model, tokenizer, device, id2label)

    print("\n" + "=" * 60)
    print("INDIC NER PREDICTION RESULTS")
    print("=" * 60)
    print(f"Input: {result['text']}")
    print("-" * 60)
    print(f"{'Token':<25} | {'Predicted Tag':<15} | {'Confidence':<10}")
    print("-" * 60)
    for tok, tag, conf in zip(result["tokens"], result["tags"], result["confidences"]):
        print(f"{tok:<25} | {tag:<15} | {conf:<10.4f}")
    print("=" * 60)

    if result["entities"]:
        print("\nExtracted Entities:")
        for ent in result["entities"]:
            print(f"  • [{ent['type']}] {ent['text']} (conf: {ent['confidence']:.2%})")
    else:
        print("\nNo named entities detected.")
    print()


if __name__ == "__main__":
    main()
