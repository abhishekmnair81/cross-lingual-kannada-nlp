# Engineering & Architectural Decisions (DECISIONS.md)

This document records all architectural choices, data handling decisions, and pipeline designs implemented for the IndicNER project (Kannada MuRIL with Telugu Zero-Shot Transfer).

---

## 1. Sanitizing `tags='nan'` Artifact in `kn_preprocessed_test.csv` (Row 150)

- **Context**: In the raw dataset (`kn_test.csv`), row 152 was an empty line (`,`). During prior preprocessing without `keep_default_na=False`, pandas interpreted the empty field as `NaN` and serialized it as literal string `"nan,nan,1"`. When loaded with `pd.read_csv(dtype=str, keep_default_na=False, na_filter=False)`, `row['tokens']` and `row['tags']` were both loaded as the string `"nan"`.
- **Constraint**: The specification explicitly states:
  1. `te_test.csv` has exactly 3 dropped rows, and *all other files have 0 dropped rows*.
  2. All tags must belong to the 7-label set: `["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG"]`.
- **Decision**: In `src/data.py`, during tag validation, any tag equal to `"nan"` is sanitized to `"O"` (Outside / non-entity). This safely preserves all 1,004 rows in `kn_preprocessed_test.csv` (0 dropped rows), honors the exact 7-label schema, and resolves the pandas string NaN artifact.

---

## 2. Train-Overlap Handling & Model Selection Protocol

- **Context**: Exact string matching revealed that:
  - 70 sentences in `kn_preprocessed_val.csv` overlap with `kn_preprocessed_train.csv`.
  - 38 sentences in `kn_preprocessed_test.csv` overlap with `kn_preprocessed_train.csv`.
- **Decision**:
  - **Validation for Model Selection**: Overlapping rows are purged from the validation set during training (`kn_val_dedup`, 2,310 sentences). The checkpoint with the highest Seqeval strict micro F1 on `kn_val_dedup` is saved as `best_model`.
  - **Comprehensive Benchmark Reporting**: In `outputs/metrics.json` and `outputs/metrics.md`, we evaluate and report metrics for:
    1. `kn_val_full` (2,380 sentences)
    2. `kn_val_dedup` (2,310 sentences)
    3. `kn_test_full` (1,004 sentences)
    4. `kn_test_dedup` (966 sentences)
    5. `te_val` (2,700 sentences)
    6. `te_test` (844 sentences, 3 mismatched rows dropped)

---

## 3. Subword Label Alignment & Truncation Invariants

- **Alignment**:
  - The first subword of each word receives the word's label ID (`label2id[tag]`).
  - All continuation subwords (prefixed with `##`) and special tokens (`[CLS]`, `[SEP]`, `[PAD]`) receive `-100` so that cross-entropy loss ignores them.
- **Prediction Invariant (`map_subword_preds_to_words`)**:
  - At inference time, predictions are extracted from the first subword of each word.
  - If a sentence exceeds `max_length=128`, words beyond the truncation threshold are automatically assigned `"O"`. This guarantees that the predicted token list has length exactly equal to the gold token count, preventing array length mismatch errors during Seqeval evaluation. Truncated sentence counts are tracked and logged.

---

## 4. Collation & Memory Optimization

- **Decision**: Dynamic padding via `DataCollatorForTokenClassification` and `group_by_length=True`. Sentences in each batch are padded only to the length of the longest sentence in that specific batch, significantly reducing compute overhead compared to static length 128 padding.
- **Disk Caching**: Tokenized datasets are saved to disk via HuggingFace `DatasetDict.save_to_disk("outputs/data_cache/...")` to prevent duplicate tokenization overhead across reruns.

---

## 5. Hardware Detection & Execution Scaling

- **Detection**: `src/utils.py` automatically detects CUDA GPU, Apple Silicon MPS, or CPU.
- **Local Environment**: Running on a local Windows CPU (`PyTorch 2.14.1+cpu`). A clear warning is printed advising that full training of MuRIL on 464,298 sentences on CPU requires ~18–24 hours, and recommending Google Colab (T4/A100) or Kaggle GPU for full production runs.
- **Debug Mode (`--debug`)**: Provides a fast smoke test that trains on 2,000 samples for 1 epoch with rapid validation evaluation every 50 steps, allowing full pipeline verification in minutes.

---

## 6. Strict Light Theme Design System

- **Plots**: Matplotlib/Seaborn figures use `seaborn-v0_8-whitegrid`, pure white background (`#FFFFFF`), subtle gridlines (`#E5E7EB`), dark slate text (`#1F2937`), and 200 DPI resolution.
- **Palette**: Strictly curated pastel colors across plots and UI:
  - `PER`: `#93C5FD` (light blue)
  - `LOC`: `#86EFAC` (light green)
  - `ORG`: `#FDBA74` (light orange)
  - `O`: `#E5E7EB` (light grey)
  - `accent`: `#A78BFA` (soft purple)
- **Gradio Web App**: Built with `gr.themes.Soft(primary_hue="blue", neutral_hue="slate")`, custom CSS with `#F8FAFC` page background, `#FFFFFF` cards, `#E2E8F0` borders, and JavaScript mutation observer forcing light mode.

---

## 7. Metrics & Error Analysis Architecture

- **Headline Metric**: `seqeval` with `mode="strict"`, `scheme=IOB2` (per CoNLL/SemEval standards).
- **Secondary Metric**: Default `conlleval` mode.
- **Artifacts Generated**:
  - `outputs/metrics.json`: Complete hierarchical dictionary of all metrics and error logs.
  - `outputs/metrics.md`: Markdown summary table.
  - `outputs/top_30_mispredictions_kn.csv` & `outputs/top_30_mispredictions_te.csv`: Top 30 mispredicted word patterns.
  - `outputs/wrong_predictions_kn_100.csv` & `outputs/wrong_predictions_te_100.csv`: 100 sample errors per language.

---

## 8. Custom Modern Web Interface & FastAPI Backend Architecture

- **User Pivot & Requirement**: Upgraded from standard multi-tab Gradio UI to a custom, advanced web application to give full control over aesthetics, responsiveness, layout, and user experience.
- **Backend**: Python `FastAPI` + `Uvicorn` server providing clean RESTful endpoints:
  - `POST /api/predict`: Runs inference via `predict_text`, parses tokens, computes WordPiece subwords, identifies language and cross-lingual transfer mode, and serializes filtered entities.
  - `GET /api/metrics`: Serves serialized benchmark evaluation JSON payload.
  - `GET /api/health`: Provides model status, device detection, checkpoint path, and labels.
  - Static file mounts for `/static` and `/outputs`.
- **Frontend Architecture**:
  - Pure HTML5, CSS3, and Vanilla JavaScript (`app.js`) without heavy framework overhead.
  - **Two-Column Workspace**:
    - Left column: Text input with live character and word counters, automatic Unicode-based Kannada vs Telugu script detector badge, clickable bilingual example chips, confidence threshold slider (0.00–1.00), and entity category checkboxes (`PER`, `LOC`, `ORG`).
    - Right column: Real-time KPI summary metric cards (Total Entities, PER, LOC, ORG), pastel highlighted entity mentions with tooltips, sub-tabs for Token Diagnostic Table, Subword Breakdown, Benchmark Evaluation figures, and Raw JSON with one-click clipboard copy.
  - **Strict Light Design System**: Clean slate and white palette (`#FFFFFF`, `#F8FAFC`, `#E2E8F0`), curated pastel entity tags (`#93C5FD`, `#86EFAC`, `#FDBA74`), micro-animations, and responsive breakpoints.
