**English** · [Bahasa Indonesia](README.id.md)

# Nyatet

A distilled span-tagging model, small enough to ship with the app and run on the shop's own device. It replaces a large-model API for pulling structure out of informal Indonesian text.

> **Status:** implementation hasn't started. This document is an architecture spec and a set of early benchmark results, not documentation of a running system. Everything still outstanding is listed under [Not built yet](#not-built-yet).

![Model](https://img.shields.io/badge/model-IndoBERT--lite--p2-blue)
![Size](https://img.shields.io/badge/ONNX-42.6%20MB-green)
![Latency](https://img.shields.io/badge/latency-18.3%20ms-green)
![Runtime](https://img.shields.io/badge/runtime-onnxruntime-lightgrey)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Status](https://img.shields.io/badge/status-pre--implementation-orange)

---

## The problem

An order admin at a grocery wholesaler has 40 to 80 regular trade customers who reorder every week by text message. At 22:00 sixty messages come in, each one written the way that customer has always written it, and they get copied out by hand into the order book before the 05:00 loading run.

The messages look roughly like this:

```
bu indomi goreng 1 dus sm gula 5kg,
minyak 2 jrigen yg biasa
itu aja, besok pagi bisa?
```

The WhatsApp automation platforms that already exist (Dazo, WATI, Qiscus) solve this by constraining the input: catalog buttons, forms, guided flows. That makes sense for a consumer who's actually browsing a store. But a trade customer who's spent the last three years buying the same twelve items from the same supplier isn't going to click through a catalog, and there's no point trying to make them.

So Nyatet goes the other way: let them type the way they always type, then parse the message.

---

## Planned architecture

```
       raw message
             │
             ▼
   ┌────────────────────────────┐
   │ tagger  (ONNX, 42.6 MB)    │   BIO spans: ITEM / QTY / UNIT / ANAPHORIC
   └────────────────────────────┘
             │
             ▼
   ┌────────────────────────────┐
   │ normalizer                 │   "dua" -> 2, "seperempat" -> 0.25
   └────────────────────────────┘     (lookup table + regex, not a model)
             │
             ▼
   ┌────────────────────────────┐
   │ resolver                   │   span -> catalog SKU + pack multiplier
   └────────────────────────────┘
             │
             ▼
   order lines + tagging_confidence + resolution_confidence
```

Only the first stage touches the model. The other two are designed as plain deterministic Python with no model dependency, so most of the pipeline can be tested without a checkpoint at all.

---

## Design decisions

These are locked in, and the implementation will be built on them.

### Why tagging instead of generating JSON

The model emits BIO spans over the original text rather than making up JSON.

- Every output token points at a position inside the input, so the model can't hallucinate an item that isn't in the message.
- The output space is bounded, so a small encoder is enough. No need for a billion-parameter API.
- Catalog resolution becomes a separate stage that can be swapped out without touching the model.

The "just tell an LLM to output JSON" approach needs a big model precisely because generation is unconstrained. Once you narrow the output space, a small model starts to make sense.

### Four span types

| Type | Example |
|---|---|
| `ITEM` | `indomi goreng` |
| `QTY` | `1`, `dua`, `seperempat` |
| `UNIT` | `dus`, `kg`, `renceng` |
| `ANAPHORIC` | `yg kyk kmrn`, `yg biasa`, `itu aja` |

### Quantity arithmetic isn't something the model learns

`dua` becomes 2, `seperempat` / `¼` / `1/4` becomes 0.25. That's a lookup table and a regex, not something that needs training.

Pack size isn't language knowledge either, it's a **catalog column**. A `dus` of instant noodles and a `dus` of soap clearly hold different amounts, and the only thing that knows the difference is the catalog. The model tags the span, the normalizer converts the number, the catalog supplies the multiplier.

### Two separate confidence scores

Parsing and resolution can fail independently, so they'll be reported independently too.

| Score | Source | Failure it catches |
|---|---|---|
| Tagging confidence | Span probability | Didn't notice an entity was there |
| Resolution confidence | Top-1 vs top-2 margin | Noticed it, but can't tell which SKU |

The case that motivates splitting them: a token like `indomi` should tag confidently, because the sentence makes it obvious it's a product name. But resolving it is genuinely ambiguous, since *Goreng*, *Soto* and *Ayam Bawang* are all about equally close. If you fold those two things into a single number, this case reads as bad parsing when the parsing was correct and the only thing missing was catalog detail.

---

## Measured results

Architecture validation on IndoNLU NERP, run separately on Colab/Kaggle. This measures whether the span-tagging approach is viable, not the ordering task itself. Training and export code are identical for both models, only the checkpoint name differs.

| | IndoBERT-lite-p2 (11.7M) | IndoBERT-base-p2 (124M) |
|---|---|---|
| F1, fp32 | 0.8040 | 0.8077 |
| F1, int8 (full) | 0.8014 | 0.8088 |
| Size, fp32 | 42.6 MB | 472.7 MB |
| **Latency, int8 (full)** | **18.3 ms** | **18.1 ms** |

Three things come out of these numbers:

**Scale barely helps on this task.** 10.6× the parameters buys 0.004 F1.

**Parameter count determines file size, not speed.** Both models land at ~18 ms once quantized. ALBERT does reduce parameters by sharing weights across layers, but it still runs all 12 of them. The entire speedup came from quantization, not from picking a small model.

**Quantization doesn't hurt accuracy.** The spread across five variants runs from 0.8014 to 0.8106, which is inside run-to-run variance.

> **Note on the metric.** These are `seqeval` span-level strict scores. IndoNLU reports sequence labeling tasks with word-level metrics, so these figures **can't be compared directly** against their published table without aligning the metric first.

### A note on quantization

The default `quantize_dynamic` on IndoBERT-lite only shrank the model by ~10%, nowhere near the expected 4×, and it took a while to work out why.

Inspecting the ONNX initializer data types turned up the cause: ALBERT's cross-layer parameter sharing makes the exporter emit the transformer weights as anonymous graph constants (`onnx::MatMul_*`), each one referenced by twelve consumers. ONNX Runtime's default dynamic quantizer skips straight past them, so the only thing that actually got quantized was the embedding table.

The fix:

```python
quantize_dynamic(
    model_input, model_output,
    op_types_to_quantize=["MatMul"],
    extra_options={"MatMulConstBOnly": False},
)
```

Full quantization restored, at no cost to accuracy.

---

## Planned repo structure

```
training/
  order/            target task: data generator + fine-tune + export
  benchmark/        IndoNLU NERP validation
                    (runs on our own machine / Colab / Kaggle,
                     never inside the served container)
serving/            FastAPI + inference, ONNX only, no torch
frontend/           one HTML page, one text box, no build step
tests/              pytest for every deterministic component
docs/               PRD, rulebook digest, handoff notes
docker-compose.yml  the only thing the judges need to run
```

`training/` and `serving/` deliberately have separate dependency trees. `training/*/requirements.txt` carries torch, transformers, datasets. `serving/requirements.txt` is just `onnxruntime` + `transformers` (for the tokenizer only) + FastAPI. The Docker image won't install torch at all, which keeps start-up fast and the image small.

The AI / backend / frontend split is also meant to be readable straight from the directory layout: the model gets touched in exactly one file, and every stage after it is deterministic.

---

## Risks we've already accounted for

Written down here so nobody has to ask:

- Order-line grouping is planned around a positional heuristic. Attaching quantities correctly in multi-item messages is an open problem and most likely where the errors will come from.
- Unit conversion will only handle pack units the catalog actually declares.
- The elicited evaluation set will probably stay small (*n* ≈ 150) and authored, rather than taken from real operations.
- Benchmark-domain evaluation doesn't exercise the normalizer or the resolver, so the numbers above don't represent end-to-end accuracy.
- Event spans are the weakest class on the benchmark (EVT 0.388 F1).
- Latency figures are still means over 50 runs. They need re-measuring as median and p95.

---

## Not built yet

Roughly in the order we plan to build it:

- [ ] `training/order/generate_data.py`, phenomenon-weighted synthetic training data generator
- [ ] `training/order/train_tagger.py`, tagger fine-tuning
- [ ] Elicited evaluation set
- [ ] Trained Order model
- [ ] `export_onnx.py`, export + quantization into `serving/models/tagger/`
- [ ] `normalizer.py` and `resolver.py`, all the deterministic stages
- [ ] Test suite for the deterministic components
- [ ] FastAPI service and the `/parse` endpoint
- [ ] `docker-compose.yml`
- [ ] Single-page frontend
- [ ] Teacher–student distillation (the initial plan is to fine-tune directly first)
- [ ] RAG comparison baseline
- [ ] `PACK_SIZE`, `PAYMENT_NOTE` and `LOCATION` span types, scheduled as a 10-hour increment in the final round

## Documents

| File | Contents |
|---|---|
| `docs/PRD.md` | Full spec: architecture, data plan, evaluation, timeline, risks |
| `docs/rulebook-digest.md` | Competition rules, extracted and reorganized |
| `docs/handoff.md` | Planning context and decisions that are locked in |
