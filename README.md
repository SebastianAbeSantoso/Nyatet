**English** · [Bahasa Indonesia](README.id.md)

# Nyatet

A span-tagging model small enough to ship with the app and run on the shop's own device. It replaces a large-model API for pulling structure out of informal Indonesian text.

> **Status:** The order-parsing pipeline runs end to end. Distillation was evaluated and not shipped. The RAG comparison is completed, details below. Progress is listed under [Status](#status).

![Model](https://img.shields.io/badge/model-IndoBERT--lite--p2-blue)
![Size](https://img.shields.io/badge/ONNX%20int8-11.00%20MB-brightgreen)
![Latency](https://img.shields.io/badge/latency-21.6%20ms%20(1%20thread)-green)
![F1](https://img.shields.io/badge/F1-0.84%20(n%3D81)-brightgreen)
![Offline](https://img.shields.io/badge/inference-100%25%20offline-brightgreen)

---

## Quick start

```bash
docker compose up --build
```

Then open `http://localhost:8000`. The container serves the UI there.

Try `Pesan resoles 50 biji`. The quantity resolves, the SKU stays open with three candidates, and the response asks which variant. That is the two-score design in one request.

Full response shape: [Interface](#interface).

---

## The problem

A home food producer in Banjarmasin takes orders in person and over WhatsApp. The messages are abbreviated, mixed between Indonesian and Banjarese, and often don't name the product at all. A regular ordering their usual doesn't need to.

Real messages, from the dataset in this repo:

```
Esok tinggali 20 biji ya, jam 08.15 diambil
Pagi ini tinggali risol goreng 10 biji, risol mentah 10 biji
Bisakh pesan risoles yg mentah 20, jadikan 2 kotak
320 buting risol lh.
Om esk pagi ibu ambil ky biasa jam 6 pagi 45 buting bisalh
```

Existing automation tools (Dazo, WATI, Qiscus) handle this by constraining the input: catalogs, order forms, guided dialogue. That works for a first-time buyer browsing an online store. It fails for the weekly regular, who won't navigate a menu to order the one thing they always order, so the seller keeps transcribing by hand.

Nyatet goes the other way: keep the message as it is, and parse it.

---

## Why this generalizes

The order-parsing task is the demonstration, not the contribution. Nothing in `serving/` names a span type. The label inventory is read from the model's own `config.json` at load time, so the same code serves the IndoNLU NERP model (person, place, industry, event, food) and the order model (item, quantity, unit, variant, anaphoric).

Any task where the answer is *spans inside the input* fits the same pipeline: receipts, forms, delivery notes, or a different trade. What the demonstration establishes is that a bounded output space makes a small offline model sufficient for that class of task.

---

## Architecture

```
       raw message
             │
             ▼
   ┌────────────────────────────┐
   │ tagger                     │   BIO spans, the only model
   └────────────────────────────┘
             │
             ▼
   ┌────────────────────────────┐
   │ normalizer                 │   deterministic rules, no model
   └────────────────────────────┘
             │
             ▼
   ┌────────────────────────────┐
   │ resolver                   │   catalog match + confidence
   └────────────────────────────┘
             │
             ▼
   order lines + tagging_confidence + resolution_confidence
```

---

## Design decisions

### Tagging, not generating JSON

The model emits BIO spans over the original text rather than generating structured output. Every output token points at a position inside the input, so the model cannot hallucinate an item that is not in the message, and the bounded output space is what makes a small encoder enough.

The "tell an LLM to output JSON" approach needs a big model precisely because generation is unconstrained. Once the output space is narrowed, a small model becomes sufficient. Catalog resolution then becomes a separate stage that can be swapped without touching the model.

### Five span types

| Type | Real examples |
|---|---|
| `ITEM` | `risol`, `risoles`, `resol`, `bronies` |
| `QTY` | `20`, `10`, `320`, `1` |
| `UNIT` | `biji`, `buting`, `kotak`, `pcs`, `loyang` |
| `VARIANT` | `mentah`, `digoreng`, `frozen`, `sdh masak` |
| `ANAPHORIC` | `ky biasa`, `kya kmrn` |

`VARIANT` is a separate type rather than folded into `ITEM`. In the annotated data it appears in 12 of 29 orders, and it frequently arrives detached from the product:

```
25 nya di goreng 25 yg mentah
→ QTY "25", VARIANT "di goreng", QTY "25", VARIANT "mentah"
```

`ANAPHORIC` covers references the system cannot resolve at all. `ky biasa` ("as usual") might mean the usual quantity, the usual variant, the usual pickup time, or all three, and the seller knows from history while nobody else does. So the output is a flag, not a guess.

### Nearly half of real orders never name the product

14 of 29 annotated orders contain no `ITEM` span:

```
10 biji ga adakah
Pagi 12 digoreng
Yg mentah 10
```

The product is implicit for two different reasons, and only one of them is safely resolvable. Sometimes it was never stated, because the seller makes essentially one thing. Sometimes it was stated in an earlier message and the customer is completing an order across turns: `Bagoreng`, `Enggeh mentah`.

The resolver treats both the same way: default to the primary product, mark the line `item_inferred`, and reduce confidence. That is correct for the first case and coincidentally correct for the second, since this seller has one primary product. With a broader catalog the second case would need conversation history, which the MVP does not carry.

### Quantity arithmetic is not the model's job

Nested box quantities are the interesting case, and they are frequent:

```
Risol isi 15 1 kotak     → 15 per box, 1 box
Buat 2 kotak isi 10      → 10 per box, 2 boxes
```

Both tag as two `QTY` spans and one `UNIT`. A positional rule in the normalizer disambiguates: the `QTY` adjacent to the `UNIT` is the number of units, the other is contents-per-unit. Pack size is a catalog column, since a `kotak` of risol and a `loyang` of brownies hold different amounts. Deterministic, unit-testable, and no model capacity spent on it.

### Two separate confidence scores

Parsing and resolution fail independently, so they are reported independently.

| Score | Source | Failure it catches |
|---|---|---|
| Tagging confidence | Span probability | Didn't notice an entity was there |
| Resolution confidence | Top-1 vs top-2 margin | Noticed it, but can't tell which catalog row |

The motivating case is in the real data, repeatedly. The customer omits the variant, and the seller has to ask:

```
buyer:  Pesan resoles 50 biji
seller: Mentah kh ka?
```

`resoles` tags confidently, it is obviously the product. But it resolves ambiguously, because mentah and goreng are equally close catalog rows. A single conflated score would report this as bad parsing when the parsing was correct and only the variant was missing. Those seller questions are annotated in the dataset (`is_clarifying_question`, with `triggered_by_idx`), which gives ground truth for when resolution *should* fail.

---

## Data

**60 real WhatsApp conversations, 682 messages, one seller.** Used with the owner's permission. PII such as names, identification and location landmarks are replaced with placeholders.

| | |
|---|---|
| Order-bearing buyer messages | 102 |
| Seller clarifying questions (linked to trigger) | 47 |
| Split | 45 conversations train / 15 evaluation, split by conversation |
| Annotated | 81 messages, 40 with spans, 41 non-order negatives |

Splitting by conversation rather than by message matters: the same customer's phrasing must not appear on both sides.

Spans mark entity mentions, not order content. The model sees one message with no history, so `mentah` in `Enggeh mentah` and in `risol mentah 20 biji` are indistinguishable to it. Which mentions become order lines is decided by the resolver, not by the annotation.

Training data is synthetic, evaluation data is real. The generator (`training/order/generate_data.py`) is built from the observed distributions of the training-side conversations: real unit vocabulary, real variant morphology (`butinglh`, `mentahnya`), and the structural frequencies above. Only the real held-out set produces a reportable number; the synthetic split is diagnostic.

Three standing-order conversations (a reseller sending weekly day-to-quantity schedules like `selasa - jumat 34, sabtu 38`) are excluded from evaluation and used as negative examples. The MVP detects and flags these rather than parsing them.

---

## Measured results

### Order tagger: 81 real held-out messages, split by conversation

| | |
|---|---|
| F1 | 0.837 |
| Per class | QTY 0.932 · UNIT 0.930 · ITEM 0.900 · VARIANT 0.583 |
| False positives on non-orders | 5 of 41 messages (12%) |
| Size (ONNX int8) | 11.00 MB |
| Latency (1 thread) | 21.6 ms median, 22.6 ms p95 |

At n=81 one span is ~0.010 F1, so differences under ~0.02 are not measurable.

### Depth vs parameters

| | Params | Layers | F1 | Size | Median |
|---|---|---|---|---|---|
| base-p2 teacher | 124 M | 12 | 0.8378 | 118.80 MB | 24.0 ms |
| lite-p2, shipped | 11.7 M | 12 | 0.8365 | 11.00 MB | 21.6 ms |
| distilled student | ~20 M | 4 | 0.6917 | 25.45 MB | 2.4 ms |

10.6× the parameters at constant depth changes nothing, including the false-positive rate. Cutting 12 layers to 4 gives 9× the speed for 0.145 F1 and triples the false-positive rate on ordinary chatter. Depth buys the ability to output nothing, and that only shows up when you test on messages that should produce no output.

Full methodology, per-class breakdowns, and quantization detail: [docs/RESULTS.md](docs/RESULTS.md)

---

### Against generative baselines

Two 8B instruction-tuned LLMs, few-shot prompted for the same five span types, same 81 messages, same scoring.

| | F1 | Size | sec/msg | Schema fails | Hallucinated spans |
|---|---|---|---|---|---|
| Nyatet tagger | 0.837 | 11 MB | 0.022 | 0 | 0 |
| Sahabat-AI 8B | 0.856 | ~8 GB | 136.7 | 0 | 10 |
| SEA-LION v4 8B | 0.762 | ~8 GB | 13.0 | 9 | 72 |

Sahabat-AI beats the shipped model on F1 by 0.019. No single number here is the result, speed alone is meaningless with a server, and 11 MB is worth nothing in isolation. The result is the combination: 0.837 at 11 MB at 22 ms, offline, on one CPU thread. The baseline buys 0.019 F1 for ~6,000× the latency and ~700× the storage.

Both baselines emitted spans whose text does not appear in the input — 10 and 72. The tagger's rate is zero by construction. Baselines were run once without prompt tuning; their numbers are a lower bound. Detail: [docs/RESULTS.md](docs/RESULTS.md).

---

## Repo structure

```
training/
  order/              target task: data generator + fine-tune + export
  benchmark/          IndoNLU NERP validation (runs on our own machine / Colab / Kaggle, never inside the served container)
serving/              the MVP
  app.py
  inference/
    tagger.py         the only file that touches a model
    normalizer.py     deterministic, no model
    resolver.py       deterministic, no model
  models/tagger/      exported ONNX
  requirements.txt
frontend/
tests/                pytest for every deterministic component
docs/                 PRD, RESULTS, CONTRIBUTING
docker-compose.yml    the only thing the judges need to run
```

The AI / backend / frontend split is readable straight from the directory layout: the model is touched in exactly one file, and every stage after it is deterministic.

---

## Interface

```
POST /parse
{ "message": "Pesan resoles 50 biji" }

{
  "message": "Pesan resoles 50 biji",
  "spans": [
    { "type": "ITEM", "text": "resoles", "confidence": 1 },
    { "type": "QTY",  "text": "50",      "confidence": 1 },
    { "type": "UNIT", "text": "biji",    "confidence": 1 }
  ],
  "order_lines": [{
    "item_span": "resoles",
    "variant_span": null,
    "matched_name": null,
    "sku": null,
    "resolution_confidence": 0.333,
    "candidates": ["Risol mentah", "Risol goreng", "Risol frozen"],
    "quantity": 50,
    "per_container": null,
    "unit": "biji",
    "total_pieces": 50,
    "item_inferred": false,
    "needs_clarification": "variant"
  }],
  "flags": [{
    "kind": "variant",
    "span_text": "resoles",
    "note": "Varian tidak disebutkan — perlu dikonfirmasi (mentah / goreng / frozen)."
  }]
}
```

`GET /health` returns `{"status": "ok", "model_loaded": true, "catalog_loaded": true, "labels": [...], "catalog_items": 8}`.

---

## Known limitations

- The evaluation set is small (n=81, 98 spans) and comes from a single seller in one city. It is real, but narrow, and nothing here shows the model generalizes to other businesses or regional varieties.
- Cake and pastry orders are undersampled: 4 conversations, 1 in the evaluation split.
- Training data is generated. The gap between synthetic training and real evaluation is the main open risk, and is reported rather than hidden.
- Tagging confidence is saturated and carries no information. Every predicted span returns ≥0.99999, including on deliberately ambiguous input. Templated training data contains no genuine ambiguity, so the model never learned to express uncertainty. Resolution confidence is unaffected, since it is a retrieval margin rather than a model output, so the two-score design holds, but one of the two scores is currently a constant and is not surfaced in the UI.
- 12% of non-order messages (5 of 41) get a spurious span, concentrated on numbers in time and price contexts (`jam 7.30`, `55 ribu`) and on `yg` triggering `ANAPHORIC`. All three patterns are fixable in the generator; not fixed, because changing it would invalidate the distillation comparison.
- `VARIANT` is the weakest class (0.583). The `di-` prefix hypothesis was tested at two ratios and neither improved it. Cause unidentified.
- Order-line grouping uses a positional heuristic. Attaching quantities correctly in multi-item messages is unsolved and is the likeliest error source.
- Cross-turn orders resolve to the primary product by coincidence, since this seller has one. A broader catalog would need conversation history, which the MVP does not carry.
- Standing orders are detected and flagged, not parsed. A weekly-schedule customer is the highest-value customer in this dataset and the MVP declines to handle them.
- Unit conversion only handles pack units the catalog declares.
- Latency was measured on Kaggle CPU, not the machine the demo will run on.
- Generative baselines were run once without prompt tuning or constrained decoding, their numbers are a lower bound.

---

## Status

**Done**
- Pipeline validated end to end: train → ONNX → int8 → serve
- Architecture decided from measurement (see docs/RESULTS.md)
- Benchmark results, B1 and B2
- Real conversation dataset: collected, pseudonymized, role-labelled, split
- Span annotation, 81-message evaluation set including non-order negatives
- Generator rebuilt from real-data distributions
- Order-domain training and evaluation (O1, O1b)
- Teacher-student distillation into a reduced-layer student (O2, not shipped)
- Normalizer, resolver, standing-order detection, and frontend wiring
- RAG comparison baseline

---

## Documents

| File | Contents |
|---|---|
| [docs/PRD.md](docs/PRD.md) | Full spec: architecture, data plan, evaluation, timeline, risks |
| [docs/RESULTS.md](docs/RESULTS.md) | Measured results: methodology, metrics, model comparisons |
| [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) | Commit convention and repo layout |
