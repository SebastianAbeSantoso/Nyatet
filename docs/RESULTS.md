# Results

> Single source for all measurements in this project.

**Last updated:** 30 July 2026 · **Latest Update: Run B1 & B2**

---

## Contents

1. [Run index](#1-run-index)
2. [Benchmark runs](#2-benchmark-runs)
3. [Findings](#3-findings)
4. [Decisions derived](#4-decisions-derived)
5. [Quantization diagnosis](#5-quantization-diagnosis)
6. [Conventions](#6-conventions)
7. [Caveats](#7-caveats)
8. [Open items](#8-open-items)
9. [Reproduction](#9-reproduction)

---

## 1. Run index

| ID | Date | Domain | Run | Status |
|---|---|---|---|---|
| B1 | 29 Jul | Benchmark · NERP | indobert-lite-base-p2 | done |
| B2 | 29 Jul | Benchmark · NERP | indobert-base-p2 | done |
| O1 | TBA | Order | student, synthetic train | not started |
| O2 | TBA | Order | teacher, Sahabat-AI 8B | not started |
| O3 | TBA | Order | distilled student | not started |
| O4 | TBA | Order | RAG baseline | not started |

---

## 2. Benchmark runs

Task: IndoNLU NERP. Both runs used identical training and export code, only the checkpoint name differed, for the basis of the architecture-transfer claim.

### 2.1 B1: indobert-lite-base-p2

`ALBERT` · 11.7 M parameters · vocab 29,999

#### Accuracy

| Variant | F1 |
|---|---|
| PyTorch checkpoint | 0.81 |
| ONNX fp32 | 0.8040 |
| int8 default | 0.8076 |
| int8 v2 · MatMul only | 0.8024 |
| int8 v3 · MatMul + embeddings | 0.8014 |

#### Size

| Variant | Size | Compression |
|---|---|---|
| fp32 | 42.6 MB | — |
| int8 default | 38.2 MB | 1.1× |
| int8 v2 | 23.25 MB | 1.83x |
| int8 v3** | 11.53 MB | 3.7× |

#### Latency

Single thread · 100 runs · 19 tokens · batch 1

| Variant | Median | p95 |
|---|---|---|
| int8 v3 | 31.0 ms | 34.1 ms |

Multi-thread means over 50 runs: fp32 29.8 · int8 default 22.7 · v2 17.9 · v3 18.3 ms

#### Per-class

Not recorded. See [C3](#7-caveats).

---

### 2.2 B2: indobert-base-p2

`BERT` · 124 M parameters · vocab 30,521 · **best variant: int8 default**

#### Accuracy

| Variant | F1 |
|---|---|
| ONNX fp32 | 0.8077 |
| int8 default | 0.8077 |
| int8 v2 · MatMul only | 0.8106 |
| int8 v3 · MatMul + embeddings | 0.8088 |

#### Size

| Variant | Size | Compression |
|---|---|---|
| fp32 | 472.7 MB | — |
| int8 default | 118.9 MB | 4.0× |
| int8 v2 | 240.95 MB | 1.96x |
| int8 v3 | 124.57 MB | 3.8× |

The v3 configuration is *larger* here. BERT has no shared weights, so default quantization already recovers full compression. v3 only adds scale and zero-point tensors.

#### Latency

Single thread · 100 runs · 19 tokens · batch 1

| Variant | Median | p95 |
|---|---|---|
| int8 v3 | 31.5 ms | 33.8 ms |

Multi-thread means over 50 runs: fp32 47.2 · int8 default 23.4 · v2 17.9 · v3 18.1 ms

#### Per-class

Test split, span-level strict.

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| PPL · person | 0.910 | 0.930 | 0.920 | 644 |
| FNB · food & beverage | 0.824 | 0.884 | 0.853 | 95 |
| PLC · place | 0.807 | 0.834 | 0.820 | 523 |
| IND · industry | 0.721 | 0.753 | 0.737 | 373 |
| EVT · event | 0.371 | 0.408 | 0.388 | 130 |
| micro avg | 0.793 | 0.823 | 0.808 | 1,765 |
| macro avg | 0.726 | 0.762 | 0.744 | 1,765 |

---

### 2.3 Comparison

| | B1 · lite | B2 · base | Ratio |
|---|---|---|---|
| Parameters | 11.7 M | 124 M | 10.6× |
| Deployed size | 11.53 MB | 118.9 MB | 10.3× |
| F1 | 0.8014 | 0.8088 | +0.007 |
| Median latency · 1 thread | 31.0 ms | 31.5 ms | 1.0× |

---

## 3. Findings

| # | Finding | Evidence |
|---|---|---|
| F1 | Scale buys 0.007 F1 for 10.6× parameters | [2.3](#23-comparison) |
| F2 | Parameter count determines size, not speed | 31.0 vs 31.5 ms across a 10.3× size gap; holds at both thread settings |
| F3 | Quantization is accuracy-neutral | Spread 0.8014–0.8106 across 9 variants |
| F4 | Quantization is the only source of speedup | 1.6–2.6× from quantization; 1.0× from model choice |
| F5 | Weight-shared architectures require `MatMulConstBOnly=False` | B1 default 1.1× vs v3 3.7×; B2 unaffected. See [§5](#5-quantization-diagnosis) |

---

## 4. Decisions derived

| Decision | Basis |
|---|---|
| Student = reduced-layer non-shared BERT, ~4 layers | F2 |
| Retain quantization in the pipeline | F3, F4 |
| Claim size and offline capability, never speed | F2, F4 |
| Apply v3 quantization config only to shared-weight models | F5 |

---

## 5. Quantization diagnosis

Supporting detail for **F5**.

**Symptom.** Default `quantize_dynamic` on B1 produced 1.1× compression against an expected 4×.

**Investigation.** ONNX initializer data types — `1`=FLOAT, `2`=UINT8, `3`=INT8:

```
fp32                {1: 25}
  word_embeddings            14.6 MB  dtype=1
  onnx::MatMul_1232           9.0 MB  dtype=1
  onnx::MatMul_1233           9.0 MB  dtype=1

int8 default        {1: 31, 2: 22}
  onnx::MatMul_1232           9.0 MB  dtype=1   <- skipped
  onnx::MatMul_1233           9.0 MB  dtype=1   <- skipped
  word_embeddings_quantized   3.7 MB  dtype=2
```

**Cause.** ALBERT shares one transformer layer's weights across all 12 layers, so the exporter emits them as anonymous shared graph constants. ONNX Runtime's default dynamic quantizer skips shared initializers, converting only the embedding table.

**Fix.**

```python
quantize_dynamic(
    src, dst,
    weight_type=QuantType.QInt8,
    op_types_to_quantize=["MatMul"],
    extra_options={"MatMulConstBOnly": False},
)
```

**Verification.** All large tensors converted. 42.6 → 11.53 MB. F1 0.8076 → 0.8014, within measurement variance.

---

## 6. Conventions

| | |
|---|---|
| Metric | `seqeval` span-level strict |
| Training | 8 epochs · lr 5e-5 · batch 16 · max_len 128 |
| Latency protocol | 1 thread · 20 warmup · 100 runs · frozen input · batch 1 · median + p95 |
| Hardware | Kaggle CPU (evaluation, latency) · T4 (training) |
| Benchmark data | IndoNLU NERP · 6,720 / 840 / 840 · 11 IOB labels |

---

## 7. Caveats

| # | Caveat |
|---|---|
| **C1** | Span-level strict is not IndoNLU's published word-level metric. Not comparable without alignment. |
| **C2** | Latency measured on Kaggle CPU, not demonstration hardware. |
| **C3** | B1 per-class breakdown not recorded. |
| **C4** | EVT class at 0.388 F1, unaddressed. |
| **C5** | B-series validates the pipeline, not the target task. Order trains on synthetic data and evaluates on elicited data — a provenance mismatch NERP does not have. **No B-series number predicts Order accuracy.** |

---

## 8. Open items

| Item | Resolves |
|---|---|
| Re-measure latency on demonstration hardware | C2 |
| Record B1 per-class breakdown | C3 |
| Align metric before citing IndoNLU baselines | C1 |

---

## 9. Reproduction

```bash
TBA / Not Yet Available
cd training/benchmark
pip install -r requirements.txt
jupyter notebook nyatet_week1_validation.ipynb
```
