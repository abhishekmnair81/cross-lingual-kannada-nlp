# IndicNER: Kannada NER with Telugu Zero-Shot Transfer

A production-ready, reproducible Natural Language Processing (NLP) framework for **Named Entity Recognition (NER)** in **Kannada (`kn`)** with cross-lingual zero-shot evaluation on **Telugu (`te`)**, fine-tuning Google's **MuRIL** (`google/muril-base-cased`) architecture.

---

## 📋 Overview & Key Capabilities

- **Backbone**: `google/muril-base-cased` (110M parameters, pretrained on 17 Indian languages).
- **Taxonomy**: 7-label IOB2 scheme: `["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG"]`.
- **Target Languages**:
  - In-Language: Kannada (`kn`)
  - Zero-Shot Cross-Lingual Transfer: Telugu (`te`) without any Telugu training data or fine-tuning.
- **Data Engineering**:
  - Unicode NFC normalization for complex Dravidian scripts.
  - Strict token-tag parity checks (filtering 3 mismatched rows in `te_test.csv`).
  - Handling of pandas string NaN artifacts in `kn_preprocessed_test.csv` (Row 150).
  - Train-overlap identification and removal for unbiased model selection (`kn_val_dedup`).
- **Subword Alignment**: Fast tokenizer word-alignment assigning entity tags to the initial subword and `-100` to continuation and special tokens.
- **Truncation Resilience**: Predictor guarantees output word count equals gold word count by padding words beyond the 128-subword cut with `'O'`.
- **Light Theme Visuals**: Publication-ready, pastel-themed figures (`#93C5FD` for PER, `#86EFAC` for LOC, `#FDBA74` for ORG, `#E5E7EB` for O, `#A78BFA` for accent).
- **Interactive Web UI**: Modern two-column HTML5/CSS3/Vanilla JS web app powered by FastAPI, featuring pastel entity badges, KPI metrics, subword breakdown, and benchmark evaluation figures.

---

## 📁 Repository Structure

```
project/
├── config.yaml               # Central configuration (hyperparameters, paths, debug settings)
├── requirements.txt          # Python dependencies
├── app.py                    # Interactive Gradio demo app (forced light mode)
├── DECISIONS.md              # Detailed log of architectural and engineering decisions
├── README_RUN.md             # Execution and reproduction guide (this file)
├── src/
│   ├── __init__.py
│   ├── labels.py             # 7-label IOB2 schema, label2id/id2label mappings
│   ├── utils.py              # Reproducibility seed (42), device detection, logging
│   ├── data.py               # Robust CSV loader, NFC normalization, HF alignment, caching
│   ├── train.py              # HF Trainer fine-tuning loop, early stopping, checkpointing
│   ├── evaluate.py           # Full evaluation suite (in-language, zero-shot, error analysis)
│   ├── predict.py            # CLI text inference interface
│   └── plots.py              # Light-themed Matplotlib/Seaborn visualization suite
├── tests/
│   ├── conftest.py           # Pytest path configuration
│   ├── test_alignment.py     # Subword label alignment and truncation length unit tests
│   └── test_data.py          # Data validation, parity checks, NaN handling unit tests
└── outputs/
    ├── labels.json           # Serialized label mappings
    ├── metrics.json          # Complete hierarchical metrics payload
    ├── metrics.md            # Benchmark comparison tables
    ├── checkpoints/          # Checkpoints and best_model weights
    ├── data_cache/           # Pre-tokenized HuggingFace Dataset disk cache
    └── figures/              # Generated high-DPI light-theme visual figures
```

---

## 🚀 Quickstart & Reproduction Commands

All commands below assume you are inside the `project/` directory with Python 3.10+ and your virtual environment activated.

### 1. Environment Setup

```bash
cd project
pip install -r requirements.txt
```

### 2. Run Unit Tests

Execute the comprehensive test suite verifying:
- Subword alignment (-100 on continuations and special tokens, correct ID on first subword)
- Sentence truncation length preservation
- Dropping exactly 3 rows from `te_test.csv` (leaving 844) and 0 from other splits
- Prevention of float `NaN` values via `keep_default_na=False`

```bash
pytest tests/ -v
```

### 3. Fast Smoke-Test (`--debug`)

To test the entire pipeline end-to-end in minutes on 2,000 training samples:

```bash
python -m src.train --debug
```

This command will:
1. Validate datasets and log the data audit table.
2. Tokenize and cache the training subset.
3. Fine-tune `google/muril-base-cased` for 1 epoch.
4. Evaluate on `kn_val_dedup` every 50 steps.
5. Save the best checkpoint to `outputs/checkpoints/best_model`.
6. Generate all 6 light-theme figures in `outputs/figures/`.
7. Run the full evaluation suite across all 6 splits and compute the Kannada-Telugu zero-shot gap.
8. Generate `outputs/metrics.json`, `outputs/metrics.md`, and error analysis CSVs.

### 4. Full Production Training

To train on the complete 464,298 Kannada sentences across 3 epochs:

```bash
python -m src.train
```

> **Hardware Note**: Training the full dataset on CPU is extremely compute-intensive (~18–24 hours). For production training, a CUDA-enabled GPU (NVIDIA T4, V100, or A100 via Google Colab or Kaggle) is strongly recommended. Mixed precision (`fp16` or `bf16`) will be automatically engaged when a CUDA device is detected.

To train with a custom sample size limit on CPU:

```bash
python -m src.train --max_train_samples 10000
```

### 5. Evaluation Only

If a trained checkpoint already exists in `outputs/checkpoints/best_model`:

```bash
python -m src.train --eval_only
```

### 6. Command-Line Inference (`predict.py`)

Run inference on arbitrary Kannada or Telugu sentences:

**Kannada Example:**
```bash
python -m src.predict --text "ಬೆಂಗಳೂರಿನಲ್ಲಿ ಇಸ್ರೋ ಕಚೇರಿ ಇದೆ"
```

**Telugu Example (Zero-Shot):**
```bash
python -m src.predict --text "హైదరాబాదులో గూగుಲ್ కార్యాలయం ఉంది"
```

Output includes a token-level alignment table, predicted tags, and extracted entity spans with confidence scores.

### 7. Interactive IndicNER Web Application (`app.py`)

Launch the modern two-column web dashboard powered by **FastAPI** and a custom **HTML5/CSS3/Vanilla JS** interface:

```bash
cd project
python app.py
```

Then open your browser at **`http://127.0.0.1:7860`**.

**Features & Capabilities**:
- **Modern Two-Column Layout**: Left-hand input panel with dynamic character/word counters, clickable Kannada/Telugu sample chips, confidence threshold slider, and entity filter toggles; right-hand results panel with live KPI summary metric cards and highlighted mentions.
- **Strict Light Pastel Palette**: PER (`#93C5FD`), LOC (`#86EFAC`), ORG (`#FDBA74`), with subtle borders, rounded badges, and tooltips displaying entity confidence.
- **Language & Script Auto-Detector**: Real-time Kannada (`kn`) vs Telugu (`te`) zero-shot classification badge based on Unicode code points.
- **Sub-Tabs Workspace**:
  - **Token Diagnostic Table**: Token-by-token alignment, predicted tags, confidence progress bar, and status badge (Active, Filtered, Background).
  - **Subword Breakdown**: MuRIL WordPiece decomposition (`##` prefixes), subword count, and first-subword token mapping.
  - **Evaluation & Figures**: Model benchmark accuracy metrics and gallery of 8 high-DPI light-theme evaluation plots.
  - **Raw JSON**: Formatted JSON response with one-click clipboard copy button and toast notifications.

---

## 📊 Dataset Audit & Validation Summary

| Dataset Partition | File Name | Raw Rows | Dropped | Valid Rows | Overlap with `kn_train` |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Kannada Train** | `kn_preprocessed_train.csv` | 464,298 | 0 | 464,298 | - |
| **Kannada Val** | `kn_preprocessed_val.csv` | 2,380 | 0 | 2,380 | 70 sentences |
| **Kannada Test** | `kn_preprocessed_test.csv` | 1,004 | 0 | 1,004 | 38 sentences |
| **Telugu Val** | `te_val.csv` | 2,700 | 0 | 2,700 | - |
| **Telugu Test** | `te_test.csv` | 847 | 3 | 844 | - |

- **Telugu Dropped Rows**: Indices 699 (29 tokens vs 30 tags), 787 (7 tokens vs 8 tags), and 788 (6 tokens vs 7 tags).
- **Validation Dedup for Model Selection**: 70 overlapping rows removed, yielding 2,310 sentences.
- **Test Dedup**: 38 overlapping rows removed, yielding 966 sentences.

---

## 📈 Evaluation Protocol & Metrics

- **Primary Benchmark**: Seqeval Strict IOB2 Micro F1 (requires exact entity boundary and category match).
- **Secondary Benchmark**: Conlleval default F1.
- **Reporting Splits**:
  - `kn_val_full` & `kn_val_dedup`
  - `kn_test_full` & `kn_test_dedup`
  - `te_val` & `te_test`
- **Zero-Shot Transfer Analysis**:
  - Absolute F1 Gap: $F1_{\text{Kannada}} - F1_{\text{Telugu}}$
  - Relative Performance Drop: $\frac{F1_{\text{Kannada}} - F1_{\text{Telugu}}}{F1_{\text{Kannada}}} \times 100\%$
- **Visual Artifacts** (`outputs/figures/`):
  1. `training_validation_loss_curve.png`
  2. `validation_f1_curve.png`
  3. `entity_class_distribution_train.png`
  4. `per_class_f1_comparison.png`
  5. `confusion_matrix_heatmap.png`
  6. `subword_length_histogram.png`
