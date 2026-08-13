# Results

> Single source for all measurements in this project.

**Last updated:** 12 Aug **Latest run:** O3 generative baselines (Sahabat-AI 8B, SEA-LION v4 8B)

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
| O1 | 6 Aug | Order | indobert-lite-p2, synthetic train | done |
| O2 | 8 Aug | Order | distilled 4-layer student | done |
| O3 | 10 - 12 Aug | Order | RAG baseline | done |

---

## 2. Benchmark runs

- Task: IndoNLU NERP.
- Both runs used identical training and export code, only the checkpoint name differed, the basis of the architecture-transfer claim.

### 2.1 B1: indobert-lite-base-p2

#### Variants
- v1 = default
- v2 = MatMul only
- v3 = MatMul + Gather

| Variant | F1 | Size | Compression | Median | p95 |
|---|---|---|---|---|---|
| PyTorch checkpoint | 0.81 | - | - | - | - |
| ONNX fp32 | 0.8040 | 42.6 MB | - | - | - |
| int8 v1 | 0.8076 | 38.2 MB | 1.1× | - | - |
| int8 v2 | 0.8024 | 23.25 MB | 1.83× | - | - |
| int8 v3 | 0.8014 | 11.53 MB | 3.7× | 31.0 ms | 34.1 ms |

Multi-thread means over 50 runs:
| Variant | mean |
|---|---|
| fp32 | 29.8 ms |
| int8 v1 | 22.7 ms |
| int8 v2 | 17.9 ms |
| int8 v3 | 18.3 ms |

#### Per-class

Not recorded. See [C3](#8-caveats).

---

### 2.2 B2: indobert-base-p2

#### Variants
- v1 = default
- v2 = MatMul only
- v3 = MatMul + Gather

| Variant | F1 | Size | Compression | Median | p95 |
|---|---|---|---|---|---|
| fp32 | 0.8077 | 472.7 MB | - | - | - |
| int8 v1 | 0.8077 | 118.9 MB | 4.0× | - | - |
| int8 v2 | 0.8106 | 240.95 MB | 1.96× | - | - |
| int8 v3 | 0.8088 | 124.57 MB | 3.8× | 31.5 ms | 33.8 ms |

- Latency: (single thread 100 runs, 19 tokens, batch 1. Only v3 was measured, under this protocol. See [C12](#8-caveats). 
- Multi-thread means over 50 runs:
| Variant | mean |
|---|---|
| fp32 | 47.2 ms |
| int8 v1 |23.4 ms |
| int8 v2 | 17.9 ms |
| int8 v3 | 18.1 ms |

The v3 configuration is larger here. BERT has no shared weights, so default quantization already recovers full compression, v3 only adds scale and zero-point tensors.

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
 
- The target task. Trained on generated data, evaluated on real held-out messages.
 
| | |
|---|---|
| Model | `indobenchmark/indobert-lite-base-p2` |
| Labels | 5 span types → 11 BIO labels |
| Training data | ~10,800 generated rows (8,000 orders + 35% negatives) |
| Evaluation data | 31 real messages, hand-annotated, held out by conversation |
| Shipped variant | int8 v3 |
 
#### Variants
 
- Single thread · 100 runs · 13 tokens · batch 1
 
| Variant | F1 | Size | Median | p95 |
|---|---|---|---|---|
| fp32 | 0.8901 | 42.63 MB | 46.3 ms | 48.5 ms |
| int8 v2 | 0.8962 | 22.17 MB | 22.0 ms | 23.3 ms |
| int8 v3 | 0.9022 | 11.00 MB | 21.6 ms | 22.6 ms |
 
Synthetic held-out split: 1.0000, diagnostic only, the generator grading itself.
 
The three variants span 0.012 F1, inside the ±0.02 resolution limit at this
sample size ([C6](#8-caveats)). v3 is shipped for being smallest at no
measurable accuracy cost.
 
#### Per-class
 
- int8 v3 · span-level strict · *n*=31 messages, 89 spans:
 
| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| QTY | 1.000 | 0.971 | 0.986 | 35 |
| UNIT | 0.952 | 0.952 | 0.952 | 21 |
| ITEM | 0.941 | 1.000 | 0.970 | 16 |
| VARIANT | 0.524 | 0.733 | 0.611 | 15 |
| ANAPHORIC | 1.000 | 0.500 | 0.667 | 2 |
 
`ANAPHORIC` has 2 held-out instances, 1 detected. Not reportable as an F1,
two examples can only produce 0, 0.5 or 1.0.
 
`VARIANT` is the weakest measured class by a wide margin, and consistently so
across every run. Both precision (0.524) and recall (0.733) are low, so the
model both over-predicts and misses.
 
The `di-` prefix was the leading hypothesis: the generator lists `digoreng`
and `di goreng` as variant forms while the untagged furniture contained few
other `di-` words, so the prefix could carry spurious signal against ordinary
passives (`diambil`, `dikirim`). This was tested by adding passive verbs to
the generator at two ratios, see [F9](#4-findings). Neither improved the
score, and the 10:1 ratio was worse. The cause remains unidentified.

#### Confidence saturation
 
Every predicted span returns tagging confidence ≥ 0.999995, including on
deliberately ambiguous input (`kirim yg kmrn` → 0.999995). The model never
expresses uncertainty.
 
Cause: templated training data contains no genuine ambiguity, so the model
never learned that uncertainty exists, the same fact the 1.0000 synthetic F1
reports.
 
**Consequence.** Tagging confidence is uninformative on this model and no
threshold on it will fire. Resolution confidence is unaffected, being a
retrieval margin computed in `resolver.py` rather than a model output. The
two-score architecture holds; one of the two scores is currently a constant.
See [C8](#8-caveats).

### 3.1b O1b: re-evaluation on expanded set

- Same model, same weights, same `int8 v3` artifact as [3.1](#31-o1-indobert-lite-p2-synthetic-training). Only the evaluation set changed.

The 31-message set contained only messages carrying spans, so nothing measured
whether the model correctly outputs nothing on ordinary chatter, the failure
most likely to appear in a live demonstration. 41 non-order buyer messages from
the same 15 evaluation conversations were annotated with empty span lists, and
9 entity-bearing non-orders were annotated normally.

| | 31-message set | 81-message set |
|---|---|---|
| Messages | 31 | 81 |
| With spans | 31 | 40 |
| Empty (negatives) | 0 | 41 |
| Spans | 89 | 98 |

Annotation policy: Spans mark entity mentions, not order content. The
model sees one message with no history, so `mentah` in `Enggeh mentah` and in
`risol mentah 20 biji` are indistinguishable to it. Which mentions become order
lines is decided by the resolver, not by the annotation.

#### Results, int8 v3

- F1 0.8365 (n=81)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| QTY | 0.895 | 0.971 | 0.932 | 35 |
| UNIT | 0.909 | 0.952 | 0.930 | 21 |
| ITEM | 0.947 | 0.857 | 0.900 | 21 |
| VARIANT | 0.483 | 0.737 | 0.583 | 19 |
| ANAPHORIC | 0.500 | 0.500 | 0.500 | 2 |

Content classes held up despite ITEM gaining five instances in harder contexts
(`Masih ada kah risol`, `Besok jualanlah risol`  availability questions, not
orders). `VARIANT` remains the weakest class and now carries more weight.

#### Behaviour on non-order messages

`seqeval` scores only messages with gold spans, so the 41 negatives contribute
nothing to the F1 above. Measured separately:

| | |
|---|---|
| Messages with a spurious span | 5 of 41 (12%) |
| Spurious tokens | 10 |

```
Bu tinggalijam 7.30 ya bu → I-VARIANT, B-QTY, B-QTY
Iya jd 55 ribu kalo duitnya → B-QTY, B-QTY
Jam setengah 9 diambil bisa → B-QTY
Yg pian kawa pastikan ibu ambil yg jam 6 → B-ANAPHORIC, I-ANAPHORIC
Yg nia olah ukuran berapa → I-VARIANT, I-VARIANT
```

Three patterns, all fixable in the generator:

- Numbers in time and price contexts tagged `QTY`. The generator produces times
  only as `jam 7 pagi diambil`, never as prices or irregular formats.
- The token `yg` triggering `ANAPHORIC` most anaphoric forms in the generator
  begin `yg`/`ky`/`kaya`, so the leading token carries excess signal.
- One `I-VARIANT` with no preceding `B-`: malformed BIO output.

Not fixed. Changing the generator would invalidate the O2 comparison ([3.2](#32-o2-distillation-into-a-reduced-layer-student)). See [C14](#8-caveats).

### 3.2 O2: distillation into a reduced-layer student

`indobert-base-p2` fine-tuned on the order data as teacher, distilled into a
4-layer non-shared BERT. Same generator, same held-out set, same seed as O1,
one variable changed.

| | |
|---|---|
| Teacher | `indobert-base-p2`, 124 M, 12 layers × 768 hidden |
| Student | 4 layers × 384 hidden, 6 heads, ~20 M, trained from scratch |
| Initialization | Embeddings seeded from the teacher's first 384 dimensions; layers not copied |
| Loss | `0.5 · CE(hard) + 0.5 · T² · KL(student/T ‖ teacher/T)`, T = 3.0 |
| Training | 8 epochs · lr 1e-4 · batch 32 · warmup 0.1 |

#### Teacher variants

Single thread · 100 runs · 13 tokens · batch 1

| Variant | F1 | Size | Median | p95 |
|---|---|---|---|---|
| fp32 | 0.9016 | 472.73 MB | 46.2 ms | 52.2 ms |
| int8 v2 | 0.9016 | 229.79 MB | 24.1 ms | 24.9 ms |
| int8 v3 | 0.9016 | 118.80 MB | 24.0 ms | 27.2 ms |

Quantization is exactly accuracy-neutral here: identical F1 across all three.

#### Student variants

| Variant | F1 | Size | Median | p95 |
|---|---|---|---|---|
| fp32 | 0.8643 | 101.18 MB | 3.6 ms | 3.8 ms |
| int8 v2 | 0.8600 | 80.94 MB | 2.5 ms | 2.6 ms |
| int8 v3 | 0.8756 | 25.45 MB | 2.4 ms | 2.8 ms |

#### Three-way comparison

| | Params | Layers | F1 | Size (int8 v3) | Median |
|---|---|---|---|---|---|
| O2 teacher · base-p2 | 124 M | 12 | 0.9016 | 118.80 MB | 24.0 ms |
| O1 · lite-p2, shipped | 11.7 M | 12 | 0.9022 | 11.00 MB | 21.6 ms |
| O2 student · distilled | ~20 M | 4 | 0.8756 | 25.45 MB | 2.4 ms |

Two results, in opposite directions:

Parameters move size, not speed: A 10.6× parameter increase at constant
depth gives 0.9016 against 0.9022 and 24.0 ms against 21.6 ms. Size moved
10.8×; latency moved 11%.

Depth moves speed: Cutting 12 layers to 4 gives 2.4 ms against 21.6 ms, 
9× faster. This is [F2](#4-findings) confirmed from the other side, on the
target task rather than the benchmark.

Note the student is larger than O1 (25.45 MB vs 11.00) while being 9×
faster. Fewer layers, but no cross-layer weight sharing, so its parameters sit
in storage rather than in compute. The same point from a third angle.

#### Per-class, student int8 v3

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| QTY | 0.897 | 1.000 | 0.946 | 35 |
| ITEM | 0.842 | 1.000 | 0.914 | 16 |
| VARIANT | 0.875 | 0.933 | 0.903 | 15 |
| UNIT | 0.808 | 1.000 | 0.894 | 21 |
| ANAPHORIC | 0.167 | 1.000 | 0.286 | 2 |

The 0.027 gap to O1 is concentrated in one class. Every content class
holds up, and `VARIANT` is actually *better* on the student than on O1 (0.903
vs 0.611). `ANAPHORIC` collapses: 2 true positives against 10 false ones.

Likely cause: `ANAPHORIC` spans are long, multi-word and semantically loose
(`ky biasa`, `kaya kmrn`), and a 4-layer model has less context capacity to
bound them. The generator's 19 anaphoric surface forms may also be too varied
for a student of this size. Not tested, modifying the generator would change
two variables and invalidate the comparison against O1. See [C13](#8-caveats).

#### Decision

O1 ships. The student's 0.027 deficit is outside the resolution limit, so
it is a real difference rather than noise.

The distilled student is retained as measured evidence for the architecture
finding, not as a candidate for deployment: at sixty messages a night the
difference between 21.6 ms and 2.4 ms is imperceptible, while 0.027 F1 is not.

Consequence for the contribution claim. The project describes a small
model that runs offline. It does not describe a distilled one. Wording that
implied distillation has been corrected in the PRD and README.
### 3.3 O3: generative baselines

The organisers' 22 July clarification permits RAG, agentic workflows and tool
calling as alternatives to fine-tuning. This measures the alternative rather
than asserting it was the wrong choice.

Two small instruction-tuned LLMs, few-shot prompted to extract the same five
span types as JSON, evaluated on the same 81 messages with the same `seqeval`
scoring. Both run locally, so the comparison isolates task formulation,
bounded tagging against unbounded generation, rather than confounding it with
cloud against local.

Few-shot examples are retrieved per message by TF-IDF over character n-grams
from a 2,700-row generated pool. That is the retrieval half of "RAG" in the
only form this task admits: there are no documents to retrieve, only labelled
examples.

#### Comparison

| | F1 | Size | sec/msg | Schema failures | Hallucinated spans | FP on 41 negatives |
|---|---|---|---|---|---|---|
| **Nyatet tagger, shipped** | 0.8365 | **11.00 MB** | **0.022** | **0** | **0** | 5 |
| Sahabat-AI 8B | **0.8556** | ~8 GB | 136.7 | 0 | 10 | **1** |
| SEA-LION v4 8B | 0.7623 | ~8 GB | 13.0 | 9 | 72 | 14 |

Per class:

| Class | Tagger | Sahabat-AI | SEA-LION | Support |
|---|---|---|---|---|
| ITEM | 0.900 | 0.923 | 0.745 | 21 |
| QTY | 0.932 | 0.875 | 0.757 | 35 |
| UNIT | 0.930 | 0.878 | 0.755 | 21 |
| VARIANT | 0.583 | 0.789 | 0.850 | 19 |
| ANAPHORIC | 0.500 | 0.400 | 0.400 | 2 |

#### What this shows, and what it does not

Sahabat-AI beats the shipped model on F1, 0.8556 against 0.8365, outside
the ±0.02 resolution limit, so a real difference. It also produces fewer false
positives on non-order messages: 1 of 41 against 5. Its precision is high
across every class (ITEM 1.000, QTY 0.966) and its recall is lower (0.816
against 0.888): conservative where the tagger is eager.

The two baselines differ from each other more than either differs from the
tagger. Same parameter count, same language focus, and one produced 10
hallucinations while the other produced 72; one produced zero schema failures
while the other produced 9. A generative baseline's behaviour is not
predictable in advance from its size or its language coverage.

Sahabat-AI wins on F1. Speed alone is meaningless when a server is available. 11 MB is worth nothing in isolation.
The result is the combination: 0.8365 F1 at 11 MB at 22 ms, offline, on one
CPU thread. The generative baseline buys 0.019 F1 for roughly 6,000× the
latency and ~700× the storage, and no configuration of that approach retains
its accuracy while fitting on the device the seller owns.

F12 - Hallucination is structural, not statistical. Both baselines
produced spans whose text does not appear in the input message: 10 and 72
respectively. The tagger's rate is zero by construction, since every output
token is a pointer into the input. Better prompting can reduce hallucination
frequency; it cannot make it impossible.

One observed case is characteristic. On `Pesan hari ni risol 20 adakah Bu`,
which contains `ITEM` and `QTY` but no unit, a baseline emitted
`UNIT: buting`. That is not garbling, it is pattern completion, filling the
shape of a typical order with a plausible value. It would pass any
well-formedness check and would reach the resolver as a quantity nobody
stated.

#### Experimental settings and their limits

Run once, without iteration:

| | |
|---|---|
| Prompt | Single few-shot prompt, not tuned |
| Examples | 8, retrieved by TF-IDF character n-gram similarity |
| Decoding | Greedy, `max_new_tokens=64` |
| Quantization | 4-bit NF4, `compute_dtype` bfloat16 |
| Hardware | 2× Tesla T4, layers split across devices (pipeline parallelism, not full parallel) |

bfloat16 was required because SEA-LION failed to load under fp16. Turing
provides no native bfloat16, so the compute path is emulated and slower than
newer hardware would give. The seconds-per-message figures are therefore an
upper bound for this hardware, not a general measure of generative model
speed. The same constraint is what ended the 8B teacher attempt in [C7](#8-caveats).

The two baselines' latencies are not comparable to each other. Sahabat-AI
does not emit EOS reliably and runs to `max_new_tokens` on every call,
SEA-LION terminates early. Each figure is comparable to the tagger, not to the
other baseline.

The baseline numbers are a lower bound. The prompt was not optimised, the
few-shot count and retrieval strategy were not varied, and no constrained
decoding or grammar-based sampling was used, the latter would eliminate all 9
schema failures outright. A full run took roughly three hours for Sahabat-AI,
so iterative tuning was not possible within the competition window. See
[C15](#8-caveats).

### 3.4 Evaluation data

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
| F3 | Quantization is accuracy-neutral | B-series spread 0.8014-0.8106 across 9 variants |
| F4 | Quantization is the only source of speedup | 1.6-2.6× from quantization; 1.0× from model choice |
| F5 | Weight-shared architectures need pre-processing + explicit op types | B1 default 1.1× vs v3 3.7×; B2 unaffected. See [6](#6-quantization-recipe) |
| F6 | Synthetic-to-real transfer holds: 0.902 on real messages from generated training data | [3.1](#31-o1-indobert-lite-p2-synthetic-training) |
| F7 | Templated training produces a saturated, uninformative confidence signal | [3.1](#31-o1-indobert-lite-p2-synthetic-training) |
| F8 | Nearly half of real orders contain no product name (implicit item) | 14 of 31 annotated orders |
| F9 | Generator vocabulary expansion produced no measurable effect | Three configurations scored 0.895 / 0.878 / 0.883, a 0.017 spread, inside the ±0.02 resolution limit of a 31-message set. See [C11](#8-caveats) |
| F10 | Depth determines speed where parameter count does not | 4 layers at 2.4 ms vs 12 layers at 21.6 ms, same task. See [3.2](#32-o2-distillation-into-a-reduced-layer-student) |
| F11 | A 124M teacher does not outperform an 11.7M model on this task | 0.9016 vs 0.9022 under matched hyperparameters. F1 replicated on the target domain |
| F12 | Hallucination is structural, not statistical | 10 and 72 invented spans against 0 by construction. See §3.3 |
| F13 | Generative baseline behaviour is unpredictable from size | Same 8B scale, same language focus: 0.856 vs 0.762 F1, 10 vs 72 hallucinations |
---

## 5. Decisions derived

| Decision | Basis |
|---|---|
| Student = reduced-layer non-shared BERT, ~4 layers | F2 |
| Retain quantization in the pipeline | F3, F4 |
| Claim size and offline capability, never speed | F2, F4 |
| Quantize Gather only where it pays: yes on B1 and O1 | F5, [3.1](#31-o1-indobert-lite-p2-synthetic-training) |
| Resolver must handle "quantity + variant, no item" as a normal case | F8 |
| Do not surface tagging confidence in the UI | F7 |
| Ship O1, not the distilled student | 3.2, 0.027 F1 for imperceptible latency gain |
| Contribution described as small and offline, not distilled | 3.2 |
| Ship the tagger despite Sahabat-AI's higher F1 | §3.3, 0.019 F1 for 6,000× latency and ~700× size |

---

## 6. Quantization recipe

Supporting detail for F5.

Symptom: Default `quantize_dynamic` on an ALBERT export produced 1.1× compression against an expected 4×.

Investigation: ONNX initializer data types, `1`=FLOAT, `2`=UINT8, `3`=INT8:

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

Cause: ALBERT shares one transformer layer's weights across all 12 layers, so the exporter emits them as anonymous shared graph constants (`onnx::MatMul_*`). ONNX Runtime's default dynamic quantizer skips shared initializers, converting only the embedding table. Fix is three parts, all required. Passing `MatMulConstBOnly` alone does not work, the pre-processing step is what makes the shared constants reachable.

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
| v3 | MatMul and `Gather` (the embedding lookup) |

Verification: Check initializer dtypes rather than trusting file size, `quantize_dynamic` writes a pre-processed copy alongside the quantized model, and summing a directory double-counts.

Which to use is task-dependent. On B1, v3 gave 3.7× for no accuracy cost. On O1, v3 was smallest with no measurable accuracy difference across variants, so v3 shipped. On B2 (BERT, no shared weights) default quantization already recovers full compression and v3 is unnecessary.

---

## 7. Conventions

| | |
|---|---|
| Metric | `seqeval` span-level strict |
| Benchmark training | 8 epochs · lr 5e-5 · batch 16 · max_len 128 |
| O1 training | 5 epochs · lr 5e-5 · batch 32 · max_len 96 |
| O2 teacher training | 8 epochs · lr 5e-5 · batch 32 · warmup 0.1 · weight decay 0.01 |
| O2 student training | 8 epochs · lr 1e-4 · batch 32 · warmup 0.1 · α 0.5 · T 3.0 |
| Latency protocol | 1 thread · 20 warmup · 100 runs · frozen input · batch 1 · median + p95 |
| Hardware | Kaggle CPU (evaluation, latency) · T4 (training) |
| Benchmark data | IndoNLU NERP · 6,720 / 840 / 840 · 11 IOB labels |
| Order data | generated training · 31 real held-out messages |
| Seed | 42 across all order-domain runs |

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
| **C11** | Three generator configurations were evaluated against the same held-out set: baseline, plus elicited vocabulary at two negative-example ratios for the `di-` prefix (10:1 and 4.9:1). F1 was 0.895 / 0.878 / 0.883 respectively. All differences are below the resolution limit (one span ≈ 0.011 F1), so no configuration is measurably better and the baseline was retained. The elicited vocabulary contributed anaphoric surface forms and untagged sentence furniture only, span vocabulary remains 99.2% real-corpus. |
| **C12** | B-series latency was measured single-thread for v3 only. Other variants use a superseded multi-thread mean-of-50 protocol and are not comparable to each other or to O1. F4's 1.6-2.6× range derives from those figures and should be re-derived if they are re-measured. |
| **C13** | The student's ANAPHORIC collapse (0.286 F1, 2 true positives against 10 false) was not investigated. Reducing the generator's anaphoric variety would have changed two variables and invalidated the O1 comparison. |
| **C14** | Generative baselines were run once without prompt tuning, constrained decoding, or few-shot variation. Their numbers are a lower bound. A fair comparison would optimise the prompt; a single full run cost ~3 hours on the available hardware. |

---

## 9. Open items

| Item | Resolves |
|---|---|
| Re-measure latency on demonstration hardware | C2 |
| Record B1 per-class breakdown | C3 |
| Align metric before citing IndoNLU baselines | C1 |
| Annotate non-order messages in the evaluation split | C9 |
| Re-measure B-series latency across all variants under the standard protocol | C12 |

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

**Order (O2, distillation)**

```bash
cd training/order
jupyter notebook nyatet_order_O2_distill.ipynb
```

Trains the teacher and the student in one pass. Uses the same generator and
the same held-out set as O1, so no separate data preparation is needed.

Both order notebooks require `eval_annotated.json` in the same directory.
