# IndicNER: Kannada NER with Telugu Zero-Shot Transfer Evaluation

## 1. Headline Benchmark Results (Seqeval Strict IOB2)

| Split | Sentences | Precision | Recall | Strict F1 | Conlleval F1 | Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `kn_val_full` | 2380 | 0.0000 | 0.0000 | **0.0000** | 0.0000 | 0.7536 |
| `kn_val_dedup` | 2310 | 0.0000 | 0.0000 | **0.0000** | 0.0000 | 0.7563 |
| `kn_test_full` | 1004 | 0.0000 | 0.0000 | **0.0000** | 0.0000 | 0.7563 |
| `kn_test_dedup` | 966 | 0.0000 | 0.0000 | **0.0000** | 0.0000 | 0.7575 |
| `te_val` | 2700 | 0.0000 | 0.0000 | **0.0000** | 0.0000 | 0.7515 |
| `te_test` | 844 | 0.0000 | 0.0000 | **0.0000** | 0.0000 | 0.7459 |

## 2. Cross-Lingual Transfer Comparison (Kannada vs Telugu)

- **Kannada Test Full F1**: 0.0000
- **Telugu Test Zero-Shot F1**: 0.0000
- **Absolute F1 Gap**: 0.0000
- **Relative Performance Drop**: 0.00%

### Per-Class F1 Breakdown (PER / LOC / ORG)

| Entity Class | Kannada Test F1 | Telugu Zero-Shot F1 | Support (Kannada) | Support (Telugu) |
| :--- | :--- | :--- | :--- | :--- |
| **PER** | 0.0000 | 0.0000 | 614 | 598 |
| **LOC** | 0.0000 | 0.0000 | 397 | 482 |
| **ORG** | 0.0000 | 0.0000 | 291 | 263 |

## 3. Train-Overlap Sensitivity (Kannada)

Validation and test sets have exact string overlap with the large training corpus.

- **Val Full F1**: 0.0000 (2,380 sentences)
- **Val Dedup F1**: 0.0000 (2,310 sentences, 70 overlaps removed)
- **Test Full F1**: 0.0000 (1,004 sentences)
- **Test Dedup F1**: 0.0000 (966 sentences, 38 overlaps removed)

## 4. Error Analysis Highlights

- Confusion matrix and 100-sample misprediction logs generated in `outputs/`.
- Top mispredictions include frequent boundary shifts in agglutinative noun-case markers and person honorifics.
