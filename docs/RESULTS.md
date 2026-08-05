# Results

> Single source for all measurements in this project.

**Last updated:** 6 August 2026 · **Latest run:** O1

---

## Contents

1. [Run index](#1-run-index)
2. [Benchmark runs](#2-benchmark-runs)
3. [Order domain](#3-order-domain)
4. [Findings](#4-findings)
5. [Decisions derived](#5-decisions-derived)
6. [Quantization recipe](#6-quantization-recipe)
7. [Conventions](#7-conventions)
8. [Caveats](#8-caveats)
9. [Open items](#9-open-items)
10. [Reproduction](#10-reproduction)

---

## 1. Run index

| ID | Date | Domain | Run | Status |
|---|---|---|---|---|
| B1 | 29 Jul | Benchmark · NERP | indobert-lite-base-p2 | done |
| B2 | 29 Jul | Benchmark · NERP | indobert-base-p2 | done |
| -  | 30 Jul | Order | Sahabat-AI 8B + token-classification head | abandoned, see [C7](#8-caveats) |
| O1 | **6 Aug** | **Order** | **indobert-lite-p2, synthetic train** | **done — shipped** |
| O2 | TBA | Order | distilled 4-layer student | not started |
| O3 | TBA | Order | RAG baseline | not started |

---

## 2. Benchmark runs

Task: IndoNLU NERP. Both runs used identical training and export code, only the checkpoint name differed, the basis of the architecture-transfer claim.

### 2.1 B1: indobert-lite-base-p2

`ALBERT` · 11.7 M parameters · vocab 29,999

#### Accuracy

| Variant | F1 |
|---|---|
| PyTorch checkpoint | 0.81 |
| ONNX fp32 | 0.8040 |
| int8 default | 0.8076 |
| int8 v2 · MatMul only | 0.8024 |
| int8 v3 · MatMul + Gather | 0.8014 |

#### Size

| Variant | Size | Compression |
|---|---|---|
| fp32 | 42.6 MB | - |
| int8 default | 38.2 MB | 1.1× |
| int8 v2 | 23.25 MB | 1.83× |
| **int8 v3** | **11.53 MB** | **3.7×** |

#### Latency

Single thread · 100 runs · 19 tokens · batch 1

| Variant | Median | p95 |
|---|---|---|
| int8 v3 | 31.0 ms | 34.1 ms |

Multi-thread means over 50 runs: fp32 29.8 · int8 default 22.7 · v2 17.9 · v3 18.3 ms

#### Per-class

Not recorded. See [C3](#8-caveats).

---

### 2.2 B2: indobert-base-p2

`BERT` · 124 M parameters · vocab 30,521 · **best variant: int8 default**

#### Accuracy

| Variant | F1 |
|---|---|
| ONNX fp32 | 0.8077 |
| int8 default | 0.8077 |
| int8 v2 · MatMul only | 0.8106 |
| int8 v3 · MatMul + Gather | 0.8088 |

#### Size

| Variant | Size | Compression |
|---|---|---|
| fp32 | 472.7 MB | - |
| **int8 default** | **118.9 MB** | **4.0×** |
| int8 v2 | 240.95 MB | 1.96× |
| int8 v3 | 124.57 MB | 3.8× |

The v3 configuration is *larger* here. BERT has no shared weights, so default quantization already recovers full compression; v3 only adds scale and zero-point tensors.

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

## 3. Order domain

### 3.1 O1: indobert-lite-p2, synthetic training

The target task. Trained on generated data, evaluated on **real held-out messages**.

| | |
|---|---|
| Model | `indobenchmark/indobert-lite-base-p2` |
| Labels | 5 span types → 11 BIO labels |
| Training data | ~10,800 generated rows (8,000 orders + 35% negatives) |
| Evaluation data | 31 real messages, hand-annotated, held out **by conversation** |
| Shipped variant | **int8 v2** |

#### Accuracy

| Variant | F1 | Note |
|---|---|---|
| Synthetic held-out split | 1.0000 | diagnostic only, the generator grading itself |
| PyTorch checkpoint | 0.9091–0.9247 | varies by run seed, see [C6](#8-caveats) |
| ONNX fp32 | 0.9110 | - |
| ONNX int8 v2 | 0.9158 | shipped |
| ONNX int8 v3 | 0.8958 | −0.02 for −11 MB, declined |

Per-class, int8 v2 · span-level strict · *n*=31 messages, 89 spans:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| QTY | 1.000 | 0.971 | 0.986 | 35 |
| UNIT | 0.955 | 1.000 | 0.977 | 21 |
| ITEM | 0.941 | 1.000 | 0.970 | 16 |
| VARIANT | 0.737 | 1.000 | 0.848 | 15 |
| ANAPHORIC | — | — | — | 2 |

`ANAPHORIC` has 2 held-out instances, both detected. Not reportable as an F1, two examples can only produce 0, 0.5 or 1.0.

`VARIANT` is the weakest measured class. Recall is perfect; precision is not the model over-predicts. Likely cause is the `di-` prefix: the generator lists `digoreng` / `di goreng` as variant forms while the untagged furniture contains few other `di-` words, so the prefix carries spurious signal against real passives (`diambil`, `dikirim`).

#### Size and latency

Single thread · 100 runs · 13 tokens · batch 1

| Variant | Size | Median | p95 |
|---|---|---|---|
| fp32 | 42.63 MB | 46.3 ms | 47.3 ms |
| int8 v2 | 22.17 MB | 24.6 ms | 25.7 ms |
| int8 v3 | 11.00 MB | 24.9 ms | 25.6 ms |

v3 halves the file for no speed gain and costs 0.02 F1, thus v2 is chosen.

#### Confidence saturation

Every predicted span returns tagging confidence ≥ 0.999995, including on deliberately ambiguous input (`kirim yg kmrn` → 0.999995). The model never expresses uncertainty.

Cause: templated training data contains no genuine ambiguity, so the model never learned that uncertainty exists, the same fact the 1.0000 synthetic F1 reports.

**Consequence.** Tagging confidence is uninformative on this model and no threshold on it will fire. Resolution confidence is unaffected, being a retrieval margin computed in `resolver.py` rather than a model output. The two-score architecture holds; one of the two scores is currently a constant. See [C8](#8-caveats).

### 3.2 Evaluation data

| | |
|---|---|
| Source | 60 real WhatsApp conversations, 682 messages, one seller, used with permission |
| Pseudonymized | names, business name, location landmarks replaced |
| Split | 45 conversations train / 15 evaluation, **by conversation** |
| Annotated | 31 messages (29 orders + 2 anaphoric non-orders), 89 spans |

Splitting by conversation rather than message prevents a customer's phrasing appearing on both sides.

Three standing-order conversations (a reseller sending weekly day-to-quantity schedules) are excluded from evaluation and retained as negatives.

---

## 4. Findings

| # | Finding | Evidence |
|---|---|---|
| F1 | Scale buys 0.007 F1 for 10.6× parameters | [2.3](#23-comparison) |
| F2 | Parameter count determines size, not speed | 31.0 vs 31.5 ms across a 10.3× size gap; holds at both thread settings |
| F3 | Quantization is accuracy-neutral | B-series spread 0.8014–0.8106 across 9 variants |
| F4 | Quantization is the only source of speedup | 1.6–2.6× from quantization; 1.0× from model choice |
| F5 | Weight-shared architectures need pre-processing + explicit op types | B1 default 1.1× vs v3 3.7×; B2 unaffected. See [§6](#6-quantization-recipe) |
| F6 | Synthetic-to-real transfer holds: 0.916 on real messages from generated training data | [3.1](#31-o1-indobert-lite-p2-synthetic-training) |
| F7 | Templated training produces a saturated, uninformative confidence signal | [3.1](#31-o1-indobert-lite-p2-synthetic-training) |
| F8 | Nearly half of real orders contain no product name (implicit item) | 14 of 29 annotated orders |

---

## 5. Decisions derived

| Decision | Basis |
|---|---|
| Student = reduced-layer non-shared BERT, ~4 layers | F2 |
| Retain quantization in the pipeline | F3, F4 |
| Claim size and offline capability, never speed | F2, F4 |
| Quantize Gather only where it pays: yes on B1, no on O1 | F5, [3.1](#31-o1-indobert-lite-p2-synthetic-training) |
| Resolver must handle "quantity + variant, no item" as a normal case | F8 |
| Do not surface tagging confidence in the UI | F7 |

---

## 6. Quantization recipe

Supporting detail for **F5**.

**Symptom.** Default `quantize_dynamic` on an ALBERT export produced 1.1× compression against an expected 4×.

**Investigation.** ONNX initializer data types, `1`=FLOAT, `2`=UINT8, `3`=INT8:

```
fp32                {1: 25}
  word_embeddings            14.6 MB  dtype=1
  onnx::MatMul_1232           9.0 MB  dtype=1
  onnx::MatMul_1233           9.0 MB  dtype=1

int8 default        {1: 31, 2: 22}
  onnx::MatMul_1232           9.0 MB  dtype=1   
  onnx::MatMul_1233           9.0 MB  dtype=1   
  word_embeddings_quantized   3.7 MB  dtype=2
```

Cause. ALBERT shares one transformer layer's weights across all 12 layers, so the exporter emits them as anonymous shared graph constants (`onnx::MatMul_*`). ONNX Runtime's default dynamic quantizer skips shared initializers, converting only the embedding table. Fix is three parts, all required. Passing `MatMulConstBOnly` alone does not work, the pre-processing step is what makes the shared constants reachable.

```python
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.quantization.shape_inference import quant_pre_process

quant_pre_process(
    input_model_path="onnx_fp32/model.onnx",
    output_model_path="onnx_fp32/model_preprocessed.onnx",
    skip_symbolic_shape=True,
)

quantize_dynamic(
    "onnx_fp32/model_preprocessed.onnx", "onnx_int8_v2/model.onnx",
    weight_type=QuantType.QInt8,
    op_types_to_quantize=["MatMul"],              # v2
    # op_types_to_quantize=["MatMul", "Gather"],  # v3
    extra_options={"MatMulConstBOnly": False},
)
```

| Variant | Converts |
|---|---|
| v2 | MatMul weights only; embedding table stays fp32 |
| v3 | MatMul **and** `Gather` (the embedding lookup) |

**Verification.** Check initializer dtypes rather than trusting file size, `quantize_dynamic` writes a pre-processed copy alongside the quantized model, and summing a directory double-counts.

**Which to use is task-dependent.** On B1, v3 gave 3.7× for no accuracy cost. On O1, v3 cost 0.02 F1, concentrated in ITEM and VARIANT precision, the classes most dependent on embedding fidelity — so v2 shipped.

---

## 7. Conventions

| | |
|---|---|
| Metric | `seqeval` span-level strict |
| Benchmark training | 8 epochs · lr 5e-5 · batch 16 · max_len 128 |
| Order training | 5 epochs · lr 5e-5 · batch 32 · max_len 96 |
| Latency protocol | 1 thread · 20 warmup · 100 runs · frozen input · batch 1 · median + p95 |
| Hardware | Kaggle CPU (evaluation, latency) · T4 (training) |
| Benchmark data | IndoNLU NERP · 6,720 / 840 / 840 · 11 IOB labels |
| Order data | generated training · 31 real held-out messages |

---

## 8. Caveats

| # | Caveat |
|---|---|
| **C1** | Span-level strict is not IndoNLU's published word-level metric. Not comparable without alignment. |
| **C2** | Latency measured on Kaggle CPU, not demonstration hardware. |
| **C3** | B1 per-class breakdown not recorded. |
| **C4** | EVT class at 0.388 F1, unaddressed. |
| **C5** | B-series validates the pipeline, not the target task. **No B-series number predicts Order accuracy.** |
| **C6** | O1 F1 varies ~±0.015 across training runs (seed nondeterminism). At *n*=31, one span flipping moves F1 by ~0.011. Report as a range, not a point. |
| **C7** | A Sahabat-AI 8B teacher was attempted on 2× T4 and abandoned. The architecture loads and LoRA attaches, but fp16 compute produces non-finite gradients from the freshly-initialized classification head and fp32 compute exceeds memory during loading. Turing provides no native bf16. A configuration that did train reached 0.0011 F1 after 500 examples, insufficient training to assess the approach. Not a measurement of the method, only of its feasibility on this hardware. |
| **C8** | Tagging confidence is saturated and uninformative (F7). |
| **C9** | O1 evaluation contains only messages that carry spans. Nothing currently measures whether the model correctly outputs *nothing* on non-order messages, the most likely live-demo failure. |
| **C10** | Evaluation data comes from a single seller in one city. Real, but narrow. Cake and pastry orders are undersampled: 4 conversations, 1 in evaluation. |

---

## 9. Open items

| Item | Resolves |
|---|---|
| Re-measure latency on demonstration hardware | C2 |
| Record B1 per-class breakdown | C3 |
| Align metric before citing IndoNLU baselines | C1 |
| Annotate non-order messages in the evaluation split | C9 |
| Add `di-` passives to generator furniture to lift VARIANT precision | [3.1](#31-o1-indobert-lite-p2-synthetic-training) |

---

## 10. Reproduction

**Benchmark (B1, B2)**

```bash
cd training/benchmark
pip install -r requirements.txt
jupyter notebook nyatet_week1_validation.ipynb
```

Loads IndoNLU NERP programmatically. No manual download.

**Order (O1)**

```bash
cd training/order
python generate_data.py          
jupyter notebook nyatet_order_O1.ipynb
```

Requires `eval_annotated.json` in the same directory.
