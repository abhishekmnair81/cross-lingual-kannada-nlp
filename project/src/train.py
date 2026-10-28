from __future__ import annotations
import argparse
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from seqeval.metrics import f1_score, precision_score, recall_score
from seqeval.scheme import IOB2
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from .data import compute_dataset_validation_stats, get_tokenized_dataset
from .evaluate import run_full_evaluation_suite
from .labels import ID2LABEL, LABEL2ID, get_id2label, get_label2id, get_labels, save_labels
from .plots import (
    plot_entity_distribution_train,
    plot_loss_curves,
    plot_subword_length_histogram,
    plot_val_f1_curve,
)
from .utils import find_project_root, get_device, load_config, resolve_path, set_seed, setup_logging

logger = logging.getLogger(__name__)


def compute_metrics_builder(id2label: Dict[int, str]):
    """Construct seqeval micro F1 metric function for HF Trainer."""

    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)

        true_predictions = [
            [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        p = precision_score(true_labels, true_predictions, mode="strict", scheme=IOB2, zero_division=0)
        r = recall_score(true_labels, true_predictions, mode="strict", scheme=IOB2, zero_division=0)
        f1 = f1_score(true_labels, true_predictions, mode="strict", scheme=IOB2, zero_division=0)

        return {
            "precision": p,
            "recall": r,
            "f1": f1,
        }

    return compute_metrics


def main():
    parser = argparse.ArgumentParser(description="Train IndicNER (Kannada MuRIL with Telugu Zero-Shot Transfer)")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--debug", action="store_true", help="Run debug smoke test with 2,000 samples for 1 epoch")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Explicit limit on training samples")
    parser.add_argument("--eval_only", action="store_true", help="Run evaluation only using existing checkpoint")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Setup directories
    proj_root = find_project_root()
    output_dir = resolve_path(config["outputs"]["output_dir"])
    figures_dir = resolve_path(config["outputs"]["figures_dir"])
    checkpoints_dir = resolve_path(config["outputs"]["checkpoints_dir"])
    labels_file = resolve_path(config["outputs"]["labels_file"])
    log_file = resolve_path(config["outputs"]["run_log"])

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(log_file)
    logger.info("Initializing IndicNER training pipeline...")

    # Set seed everywhere
    seed = config["training"].get("seed", 42)
    set_seed(seed)

    # Hardware detection
    device = get_device()
    is_cuda = torch.cuda.is_available()
    is_bf16 = is_cuda and torch.cuda.is_bf16_supported()

    # Save and verify labels mapping
    save_labels(labels_file)
    label2id = get_label2id()
    id2label = get_id2label()

    # Determine sample limits and epochs
    if args.debug:
        max_train_samples = config["debug"].get("train_samples", 2000)
        epochs = config["debug"].get("epochs", 1)
        eval_steps = config["debug"].get("eval_steps", 50)
        save_steps = config["debug"].get("save_steps", 50)
        logger.info("DEBUG MODE ACTIVE: %d samples, %d epoch, eval_steps=%d", max_train_samples, epochs, eval_steps)
    else:
        max_train_samples = args.max_train_samples
        epochs = config["training"].get("epochs", 3)
        eval_steps = config["training"].get("eval_steps", 2000)
        save_steps = config["training"].get("save_steps", 2000)

    # Load fast tokenizer
    model_name = config["model"]["name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=config["model"].get("use_fast_tokenizer", True),
    )

    # Prepare tokenized dataset
    dataset_dict, raw_stats = get_tokenized_dataset(
        tokenizer=tokenizer,
        config=config,
        max_train_samples=max_train_samples,
    )

    # Generate initial EDA figures: entity distribution and length histogram
    logger.info("Generating dataset EDA visualizations (light theme)...")
    kn_train_label_counts = raw_stats["kn_train"].get("label_counts", {})
    if not kn_train_label_counts and "records" in raw_stats["kn_train"]:
        from collections import Counter
        kn_train_label_counts = Counter(
            tag for r in raw_stats["kn_train"]["records"] for tag in r["tags"]
        )

    plot_entity_distribution_train(
        label_counts=kn_train_label_counts,
        output_path=figures_dir / "entity_class_distribution_train.png",
    )

    # Compute subword lengths sample for histogram
    train_lens = [len(ids) for ids in dataset_dict["train"]["input_ids"][:5000]]
    plot_subword_length_histogram(
        lengths=train_lens,
        output_path=figures_dir / "subword_length_histogram.png",
        cutoff=config["model"].get("max_length", 128),
    )

    # Load Model
    logger.info("Loading backbone model %s...", model_name)
    best_model_path = checkpoints_dir / "best_model"

    if args.eval_only:
        eval_model_dir = best_model_path if best_model_path.exists() else checkpoints_dir
        logger.info("Evaluation-only mode: loading model from %s", eval_model_dir)
        model = AutoModelForTokenClassification.from_pretrained(
            str(eval_model_dir),
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id,
        )
    else:
        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id,
        )

        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

        # Dynamic batch sizing if on CPU
        train_bs = config["training"].get("per_device_train_batch_size", 32)
        eval_bs = config["training"].get("per_device_eval_batch_size", 64)
        if not is_cuda and train_bs > 16:
            logger.info("Adjusting train batch size to 16 for CPU execution efficiency")
            train_bs = 16

        import inspect
        sig = inspect.signature(TrainingArguments.__init__)

        total_train_steps = max(1, (len(dataset_dict["train"]) // train_bs) * epochs)
        warmup_steps = max(1, int(total_train_steps * float(config["training"].get("warmup_ratio", 0.06))))

        raw_args = {
            "output_dir": str(checkpoints_dir),
            "learning_rate": float(config["training"].get("learning_rate", 3e-5)),
            "num_train_epochs": epochs,
            "per_device_train_batch_size": train_bs,
            "per_device_eval_batch_size": eval_bs,
            "weight_decay": float(config["training"].get("weight_decay", 0.01)),
            "warmup_ratio": float(config["training"].get("warmup_ratio", 0.06)),
            "warmup_steps": warmup_steps,
            "lr_scheduler_type": config["training"].get("lr_scheduler_type", "linear"),
            "max_grad_norm": float(config["training"].get("max_grad_norm", 1.0)),
            "gradient_accumulation_steps": config["training"].get("gradient_accumulation_steps", 1),
            "fp16": (is_cuda and not is_bf16),
            "bf16": is_bf16,
            "seed": seed,
            "group_by_length": config["training"].get("group_by_length", True),
            "eval_strategy": "steps",
            "evaluation_strategy": "steps",
            "eval_steps": eval_steps,
            "save_strategy": "steps",
            "save_steps": save_steps,
            "save_total_limit": config["training"].get("save_total_limit", 2),
            "metric_for_best_model": "f1",
            "greater_is_better": True,
            "load_best_model_at_end": True,
            "logging_dir": str(output_dir / "logs"),
            "logging_steps": max(5, eval_steps // 5),
            "report_to": "none",
        }

        filtered_args = {k: v for k, v in raw_args.items() if k in sig.parameters}
        if "warmup_ratio" not in filtered_args:
            filtered_args["warmup_steps"] = warmup_steps

        training_args = TrainingArguments(**filtered_args)

        early_stopping_patience = config["training"].get("early_stopping_patience", 3)
        callbacks = [EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]

        trainer_kwargs = {
            "model": model,
            "args": training_args,
            "train_dataset": dataset_dict["train"],
            "eval_dataset": dataset_dict["kn_val_dedup"],  # Validation with train-overlap removed
            "data_collator": data_collator,
            "compute_metrics": compute_metrics_builder(id2label),
            "callbacks": callbacks,
        }
        trainer_sig = inspect.signature(Trainer.__init__)
        if "processing_class" in trainer_sig.parameters:
            trainer_kwargs["processing_class"] = tokenizer
        elif "tokenizer" in trainer_sig.parameters:
            trainer_kwargs["tokenizer"] = tokenizer

        trainer = Trainer(**trainer_kwargs)

        logger.info("Starting training loop...")
        train_result = trainer.train()

        # Save best model and tokenizer
        logger.info("Saving best model checkpoint to: %s", best_model_path)
        trainer.save_model(str(best_model_path))
        tokenizer.save_pretrained(str(best_model_path))

        # Extract and plot training curves
        log_history = trainer.state.log_history
        train_losses = [entry["loss"] for entry in log_history if "loss" in entry]
        eval_entries = [entry for entry in log_history if "eval_loss" in entry]
        val_losses = [entry["eval_loss"] for entry in eval_entries]
        eval_f1_scores = [entry.get("eval_f1", 0.0) for entry in eval_entries]
        eval_steps_logged = [entry["step"] for entry in eval_entries]

        plot_loss_curves(
            train_losses=train_losses,
            val_losses=val_losses,
            eval_steps=eval_steps_logged,
            output_path=figures_dir / "training_validation_loss_curve.png",
        )

        plot_val_f1_curve(
            eval_steps=eval_steps_logged,
            val_f1_scores=eval_f1_scores,
            output_path=figures_dir / "validation_f1_curve.png",
        )

        model = trainer.model

    # Execute complete evaluation suite
    logger.info("Running complete evaluation suite (Kannada in-language + Telugu zero-shot)...")
    eval_suite_results = run_full_evaluation_suite(
        model=model,
        tokenizer=tokenizer,
        dataset_dict=dataset_dict,
        config=config,
    )

    logger.info("Pipeline execution successfully completed!")


if __name__ == "__main__":
    main()
