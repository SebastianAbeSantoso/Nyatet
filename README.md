**English** · [Bahasa Indonesia](README.id.md)

# Nyatet

An 11 MB span-tagging model, small enough to ship with the app and run on the shop's own device. It replaces a large-model API for pulling structure out of informal Indonesian text.

> **Status:** The order-parsing pipeline runs end to end. Distillation and the RAG comparison are not yet done. Progress is listed under [Status](#status).

![Model](https://img.shields.io/badge/model-IndoBERT--lite--p2-blue)
![Size](https://img.shields.io/badge/ONNX%20int8-11.00%20MB-brightgreen)
![Latency](https://img.shields.io/badge/latency-20.6%20ms%20(1%20thread)-green)
![F1](https://img.shields.io/badge/F1-0.90%20(n%3D31)-brightgreen)
![Offline](https://img.shields.io/badge/inference-100%25%20offline-brightgreen)

---

## The problem

A home food producer in Banjarmasin takes orders over WhatsApp. Customers text the way they always text, abbreviated, code-switched between Indonesian and Banjarese, with the product usually left implicit because she only makes a few things. Every order gets read and written down by hand.

Real messages, from the dataset in this repo:

```
Esok tinggali 20 biji ya, jam 08.15 diambil
Pagi ini tinggali risol goreng 10 biji, risol mentah 10 biji
Bisakh pesan risoles yg mentah 20, jadikan 2 kotak
320 buting risol lh.
Om esk pagi ibu ambil ky biasa jam 6 pagi 45 buting bisalh
```

The WhatsApp automation platforms that already exist (Dazo, WATI, Qiscus) solve this by constraining the input: catalog buttons, forms, guided flows. That makes sense for a consumer browsing a store. But a regular customer who has ordered the same thing every week for two years is not going to click through a catalog, and there is no point trying to make them.

So Nyatet goes the other way: let them type the way they always type, then parse the message.

---

## Architecture

```
       raw message
             │
             ▼
   ┌────────────────────────────┐
   │ tagger                     │  BIO spans, the only model  
   └────────────────────────────┘   
             │
             ▼
   ┌────────────────────────────┐
   │ normalizer                 │  deterministic rules, no model
   └────────────────────────────┘  
             │
             ▼
   ┌────────────────────────────┐
   │ resolver                   │  catalog match + confidence.
   └────────────────────────────┘  
             │
             ▼
   order lines + tagging_confidence + resolution_confidence
```

---

## Design decisions

### Tagging, not generating JSON

The model emits BIO spans over the original text rather than generating structured output.

- Every output token points at a position inside the input, so the model cannot hallucinate an item that is not in the message.
- The output space is bounded, so a small encoder is enough. No billion-parameter API.
- Catalog resolution becomes a separate stage that can be swapped without touching the model.

The "tell an LLM to output JSON" approach needs a big model precisely because generation is unconstrained. Once the output space is narrowed, a small model becomes sufficient.

### Five span types

| Type | Real examples |
|---|---|
| `ITEM` | `risol`, `risoles`, `resol`, `bronies` |
| `QTY` | `20`, `10`, `320`, `1` |
| `UNIT` | `biji`, `buting`, `kotak`, `pcs`, `loyang` |
| `VARIANT` | `mentah`, `digoreng`, `frozen`, `sdh masak` |
| `ANAPHORIC` | `ky biasa`, `kya kmrn` |

`VARIANT` is a separate type rather than folded into `ITEM`. In the annotated data it appears in 14 of 31 orders, and it frequently arrives **detached** from the product:

```
25 nya di goreng 25 yg mentah
→ QTY "25", VARIANT "di goreng", QTY "25", VARIANT "mentah"
```

`ANAPHORIC` covers references the system cannot resolve at all. `ky biasa` ("as usual") might mean the usual quantity, the usual variant, the usual pickup time, or all three, the seller knows from history, nobody else does. So the output is a flag, not a guess.

### Nearly half of real orders never name the product

14 of 31 annotated orders contain no `ITEM` span:

```
10 biji ga adakah
Pagi 12 digoreng
Yg mentah 10
```

The product is implicit because the seller makes essentially one thing. This is not noise, it is how single-product businesses actually order, and no synthetic generator would have produced it. The resolver therefore has to treat "quantity and variant, no item" as a normal case and default to the primary product.

### Quantity arithmetic is not the model's job

Nested box quantities are the interesting case, and they are frequent:

```
Risol isi 15 1 kotak     → 15 per box, 1 box
Buat 2 kotak isi 10      → 10 per box, 2 boxes
```

Both tag as two `QTY` spans and one `UNIT`. A positional rule in the normalizer disambiguates: the `QTY` adjacent to the `UNIT` is the number of units, the other is contents-per-unit. Deterministic, unit-testable, and no model capacity spent on it.

Pack size itself is a catalog column, not language knowledge, the catalog knows a box holds 10 or 15, the model does not need to.

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

**60 real WhatsApp conversations, 682 messages, one seller.** Used with the owner's permission. PIIs like names, identification and location landmarks are replaced with placeholders.

| | |
|---|---|
| Order-bearing buyer messages | 102 |
| Seller clarifying questions (linked to trigger) | 47 |
| Split | 45 conversations train / 15 evaluation, split **by conversation** |
| Annotated with spans | 31 evaluation orders |

Splitting by conversation rather than by message matters: the same customer's phrasing must not appear on both sides.

**Training data is synthetic, evaluation data is real.** The generator (`training/order/generate_data.py`) is built from the observed distributions of the training-side conversations — real unit vocabulary, real variant morphology (`butinglh`, `mentahnya`), and the structural frequencies above. Only the real held-out set produces a reportable number; the synthetic split is diagnostic.

Three standing-order conversations (a reseller sending weekly day-to-quantity schedules like `selasa - jumat 34, sabtu 38`) are excluded from evaluation and used as negative examples. The MVP detects and flags these rather than parsing them.

---

## Measured results

### Order tagger (O1): 31 real held-out messages, split by conversation.
|   |   |
|---|---|
| F1 | 0.902 |
| Per class | QTY 0.986 · UNIT 0.952 · ITEM 0.970 · VARIANT 0.611 |
| Size (ONNX int8) | 11.00 MB |
| Latency (1 thread) | 20.6 ms median, 22.7 ms p95 |

At n=31 one span is ~0.011 F1, differences under ~0.02 are not measurable.

Separately, on IndoNLU NERP, a 124M model scored 0.007 F1 above the 11.7M
model at 10.8× the deployed size. Full comparison and quantization: [docs/RESULTS.md](docs/RESULTS.md)

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

```bash
docker compose up --build
```

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

- The evaluation set is small (n=31 orders) and comes from a single seller in one city. It is real, but narrow, nothing here shows the model generalizes to other businesses or regional varieties.
- Cake and pastry orders are undersampled: 4 conversations, 1 in the evaluation split.
- Training data is generated. The gap between synthetic training and real evaluation is the main open risk, and is reported rather than hidden.
- Tagging confidence is saturated and carries no information. Every predicted span returns ≥0.99999, including on deliberately ambiguous input. Templated training data contains no genuine ambiguity, so the model never learned to express uncertainty. Resolution confidence is unaffected — it is a retrieval margin, not a model output — so the two-score design holds, but one of the two scores is currently a constant and is not surfaced in the UI.
- Order-line grouping uses a positional heuristic. Attaching quantities correctly in multi-item messages is unsolved and is the likeliest error source.
- Standing orders are detected and flagged, not parsed. A weekly-schedule customer is the highest-value customer in this dataset and the MVP declines to handle them.
- Unit conversion only handles pack units the catalog declares.
- Latency was measured on Kaggle CPU, not the machine the demo will run on.

---

## Status

**Done**
- Pipeline validated end to end: train → ONNX → int8 → serve
- Architecture decided from measurement (see docs/RESULTS.md)
- Benchmark results, B1 and B2
- Real conversation dataset: collected, pseudonymized, role-labelled, split
- Span annotation on the 31-message evaluation set
- Generator rebuilt from real-data distributions
- Order-domain training and evaluation (O1)
- Normalizer, resolver, and frontend wiring
- Standing-order detection

**In progress**
- Teacher-student distillation into a reduced-layer student

**Planned**
- RAG comparison baseline

**Deferred to final round**
- `PACK_SIZE`, `PAYMENT_NOTE`, `LOCATION` span types

---

## Documents

| File | Contents |
|---|---|
| [docs/PRD.md](docs/PRD.md) | Full spec: architecture, data plan, evaluation, timeline, risks |
| [docs/RESULTS.md](docs/RESULTS.md) | Measured results: methodology, metrics, model comparisons |
| [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) | Commit convention, repo layout rules, where results go |
