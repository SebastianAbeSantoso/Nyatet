# Nyatet — Product Requirements Document

**COMPFEST 18 · AI Innovation Challenge · Smart Commerce**

**Status, 8 August 2026.** The order-parsing pipeline runs end to end and is
shipped. Distillation and the RAG comparison are not yet started.

---

## 1. Thesis

Nyatet is a way to run capable Indonesian NLP where there is no connectivity
and no budget for per-call inference. A span-tagging model small enough to
ship inside the repository and run offline on ordinary hardware replaces a
hosted large-model API for pulling structure out of informal Indonesian text.

We ship one interaction flow: order parsing for a home food producer. The
method behind it is not specific to that task, which is why we also measure it
on a public benchmark. The contribution is the task formulation plus offline
deployment; the demonstration is order parsing; the evidence is the same
architecture measured across two domains.

One thing to keep straight in everything we write externally: the claim is
size and offline capability, not speed. A 124M model and an 11.7M model land
at identical latency once quantized (#6.1). Speed comes from quantization,
size comes from architecture. Conflating them would be overclaiming.

### 1.1 What ships and what does not

Order parsing is the only interactive flow. The public benchmark domain is
evaluation only and never appears in the interface. Variety shows up in the
results table and for about twenty seconds of the video. A second interactive
flow would breach the MVP scope cap, so there is not one.

---

## 2. Demonstration domain

**Who.** A home food producer in Banjarmasin who takes orders over WhatsApp.
She makes risol in three preparations (mentah, goreng, frozen), sells them by
the piece or by the box, and also takes occasional cake orders.

**When it hurts.** Customers text the way they always text: abbreviated,
code-switched between Indonesian and Banjarese, with the product usually left
implicit because she only makes a few things. Every order gets read and
written down by hand.

Real messages from the dataset in this repository:

```
Esok tinggali 20 biji ya, jam 08.15 diambil
Pagi ini tinggali risol goreng 10 biji, risol mentah 10 biji
Bisakh pesan risoles yg mentah 20, jadikan 2 kotak
320 buting risol lh.
Om esk pagi ibu ambil ky biasa jam 6 pagi 45 buting bisalh
```

Why existing tools do not reach this user. Indonesian conversational
commerce platforms (Dazo, WATI, Qiscus, ManyChat) automate ordering by
constraining the input: catalog buttons, forms, guided flows. That works for a
consumer browsing a webstore. It does not work for a regular customer who has
ordered the same thing every week for two years and will not click through a
catalog to do it again.

The one customer you cannot put behind a form is the one who reorders every
week. So Nyatet parses the free text instead of preventing it.

---

## 3. Why the model has to be small and local

### 3.1 Language asymmetries

| Phenomenon | Example | Why a general model struggles |
|---|---|---|
| Implicit product | `10 biji ga adakah` | 14 of 31 annotated orders name no product at all |
| Banjar counting units | `buting`, `biji`, `kotak` | `buting` is a Banjar classifier absent from general corpora |
| Fused particles | `butinglh`, `mentahnya`, `20bj` | Morphology attaches to the span itself |
| Detached variants | `25 nya di goreng 25 yg mentah` | Preparation state arrives separately from the product |
| Anaphora | `ky biasa`, `kaya kmrn` | Requires noticing what is *not* there |

### 3.2 Why on-device

Three drivers, in the order we would argue them.

Personal data. Order messages carry names, phone numbers, addresses. UU
PDP No. 27/2022 has been binding since 17 October 2024, with fines up to 2% of
annual revenue. Processing on-device removes the transfer and the third-party
processing entirely. Worth being careful: the supervisory body's structure is
still unsettled, so what we can claim is that we have eliminated a category of
risk, not that we have cleared a specific enforcement hurdle.

Connectivity. 3,029 Indonesian villages are still internet blank spots as
of 2026. The headline figure of roughly 95% mobile broadband coverage
overstates how much of that is usable.

Unit economics. Per-call API pricing does not survive Rupiah-margin trade.

None of these are uniquely Indonesian conditions. Thin margins and patchy
connectivity are the majority condition globally, and the method transfers
wherever hosted inference is unaffordable or unreachable.

---

## 4. Architecture

```
raw text
  → span tagger          BIO tagging, the only model
  → normalizer           deterministic rules, no model
  → catalog resolver     fuzzy match against a local CSV
  → order lines + two independent confidence scores
```

Every stage is domain-agnostic. The server reads its label inventory from the
model's own `config.json`, so retargeting to a new domain swaps the model file
and the catalog, not the code.

### 4.1 Tagging, not generation

The model emits BIO spans over the original text instead of generating
structured output. Three consequences:

- Every output token points into the input, so the model cannot emit an item
  that is not in the message.
- The output space is bounded, which is what makes the task fit a small
  encoder instead of a multi-billion-parameter API call.
- Resolution becomes a separate stage we can swap out.

The conventional approach, prompting a large model to generate JSON, needs
scale precisely because generation is unconstrained. Bounding the output space
is what makes a small model sufficient, and that is the central technical
claim of the project.

### 4.2 Span schema

Five types, 11 BIO labels.

| Type | Real examples |
|---|---|
| `ITEM` | `risol`, `risoles`, `resol`, `bronies` |
| `QTY` | `20`, `10`, `320`, `1` |
| `UNIT` | `biji`, `buting`, `kotak`, `pcs`, `loyang` |
| `VARIANT` | `mentah`, `digoreng`, `frozen`, `sdh masak` |
| `ANAPHORIC` | `ky biasa`, `kaya kmrn` |

`VARIANT` is a separate type rather than folded into `ITEM`. It appears in 15
of 31 annotated orders and frequently arrives **detached** from the product:

```
25 nya di goreng 25 yg mentah
→ QTY "25", VARIANT "di goreng", QTY "25", VARIANT "mentah"
```

Folding it into `ITEM` would make one label mean two different things and
would lose the quantity-to-variant pairing.

`ANAPHORIC` covers references the system cannot resolve at all. `ky biasa`
("as usual") might mean the usual quantity, the usual variant, the usual
pickup time, or all three. The seller knows from history; nobody else does.
The output is a flag, not a guess.

`PACK_SIZE`, `PAYMENT_NOTE` and `LOCATION` are deferred to the final-round
increment. Each is attested in the corpus but adding them now would widen the
schema without a measurable payoff at *n*=31.

### 4.3 Deterministic normalization

Quantity arithmetic is not the model's job.

Word numerals and fractions resolve by lookup and regex. Fused particles are
stripped (`butinglh` → `buting`). Nested box quantities are the interesting
case and they are frequent:

```
Risol isi 15 1 kotak     → 15 per box, 1 box
Buat 2 kotak isi 10      → 10 per box, 2 boxes
```

Both tag as two `QTY` spans and one `UNIT`. A positional rule disambiguates:
the `QTY` adjacent to the `UNIT` is the number of units, the other is
contents-per-unit. Pack size itself is a catalog column rather than language
knowledge, since a `kotak` of risol and a `loyang` of brownies hold different
amounts and only the catalog knows that.

Deterministic components are unit-testable and cannot fail unpredictably in
the middle of an uncut recording. 48 tests cover this stage.

### 4.4 Two independent confidence scores

Parsing and resolution fail independently, so we report them separately.

| Score | Source | Failure it catches |
|---|---|---|
| Tagging confidence | Span probability | Did not recognize an entity was there |
| Resolution confidence | Top-1 vs top-2 retrieval margin | Recognized it, cannot tell which catalog row |

The motivating case is in the real data, repeatedly. The customer omits the
variant and the seller has to ask:

```
buyer:  Pesan resoles 50 biji
seller: Mentah kh ka?
```

`resoles` tags confidently; it is obviously the product. It resolves
ambiguously, because *mentah*, *goreng* and *frozen* are equally close catalog
rows. A single conflated score would report that as bad parsing when the
parsing was correct and only the variant was missing. The interface state is
*entity recognized, needs disambiguation*, with candidates shown.

Those seller questions are annotated in the dataset (`is_clarifying_question`
with `triggered_by_idx`), which gives ground truth for when resolution
*should* fail.

**Known defect.** Tagging confidence is currently saturated and carries no
information. see #6.2. Resolution confidence is unaffected. The two-score
architecture holds; one of the two scores is a constant and is not surfaced in
the interface.

### 4.5 Out-of-scope input returns a result, not a failure

The MVP handles standalone order messages. Two classes of input it cannot
resolve are detected and flagged rather than guessed at:

- Anaphoric references, emitted as a span class, flagged for review.
- Standing orders,weekly day-to-quantity schedules
  (`selasa - jumat 34, sabtu 38`) from a reseller. Detected by a deterministic
  regex. A recurring schedule is not one order.

This matters for the proof-of-work video, which must show a non-working flow.

### 4.6 Model chain

| Stage | Artifact | Ships |
|---|---|---|
| Data generation | Phenomenon-weighted generator, built from real-data distributions | No |
| Student | `indobert-lite-base-p2`, fine-tuned | Yes |
| Export | ONNX int8, pre-processed, MatMul + Gather | Yes |
| Resolver | Fuzzy match over catalog CSV | Yes |

Distillation (O2) is not started yet. The intended teacher is
`indobert-base-p2`, already trained as B2. A Sahabat-AI 8B teacher was
attempted and abandoned, the architecture loads and LoRA attaches, but fp16
compute produces non-finite gradients from the freshly-initialized
classification head and fp32 compute exceeds memory during loading, and Turing
provides no native bf16. Documented in `RESULTS.md` C7.

Until O2 lands, the shipped model is directly fine-tuned rather than distilled,
and the project's claims are written accordingly.

### 4.7 Student architecture, decided on measurement

Benchmark results established that parameter count determines file size and
not inference speed. IndoBERT-lite and IndoBERT-base both execute 12
transformer layers at 768 hidden; ALBERT gets its parameter reduction by
sharing one layer's weights across all twelve, not by doing less work.
Measured latency was 31.0 ms against 31.5 ms, no difference.

So the distillation student is specified as a reduced-layer, non-shared BERT,
roughly 4 layers with reduced hidden width. Depth removes compute; parameter
sharing does not. Non-shared weights quantize cleanly without the
pre-processing workaround ALBERT needs.

**Size policy.** External claims state the property ("runs offline on low-cost
hardware") and never the parameter count. Once we say a number, someone will
compare it to a number from a different architecture and the comparison will
not mean anything.

---

## 5. Product scope

Constrained by the MVP scope cap: one interaction flow, synchronous
request/response, `docker compose`, localhost.

One text area, one button, one result panel. Paste a single order message, get
back tagged spans, resolved order lines with piece counts, two confidence
scores per line, and any clarification flags.

Deliberately excluded: dashboards, history, authentication, analytics,
background jobs, second flows.

The served container installs `onnxruntime` and FastAPI but never torch. That
is why `docker compose up` starts in seconds, and it is verifiable from the
two requirements files.

---

## 6. Data and evaluation

### 6.1 Benchmark domain, complete

IndoNLU NERP, 6,720 / 840 / 840 sentences, 11 IOB labels. Freely available,
loaded programmatically. Identical training and export code across both
models; only the checkpoint name differs.

| | lite (11.7M) | base (124M) |
|---|---|---|
| F1 | 0.8014 | 0.8088 |
| Deployed size | 11.53 MB | 118.9 MB |
| Median latency, 1 thread | 31.0 ms | 31.5 ms |

**10.6× the parameters buys 0.007 F1 and no latency improvement.** This is the
project's thesis, measured on a public benchmark rather than asserted.

Selected over the Shopee Code League address dataset, which is no longer
publicly obtainable, that competition is closed and access is
invitation-gated. IndoNLU is the better choice regardless, because it
publishes performance-versus-model-size trade-offs for sequence labeling,
which is this project's thesis already expressed as a peer-reviewed baseline.

### 6.2 Target domain, complete

**Evaluation data is real; training data is generated.**

| | |
|---|---|
| Source | 60 real WhatsApp conversations, 682 messages, one seller, used with permission |
| Pseudonymized | names, business name, location landmarks replaced with placeholders |
| Split | 45 conversations train / 15 evaluation, **by conversation** |
| Annotated | 31 messages (29 orders + 2 anaphoric), 89 spans |
| Training data | ~10,800 generated rows (8,000 orders + 35% negatives) |

Splitting by conversation rather than by message prevents a customer's
phrasing appearing on both sides. Three standing-order conversations are
excluded from evaluation and retained as negatives.

**O1 results**, `indobert-lite-p2`, int8 v3, *n*=31:

| | |
|---|---|
| F1 | 0.902 |
| Per class | QTY 0.986 · UNIT 0.952 · ITEM 0.970 · VARIANT 0.611 |
| Size | 11.00 MB |
| Latency, 1 thread | 21.6 ms median, 22.6 ms p95 |
| Synthetic held-out split | 1.0000, diagnostic only |

The synthetic split scoring 1.0000 is not a good sign. It means the generator's
held-out data contains nothing its training data did not already teach, and it
is the same fact that produces the confidence saturation below. It is the
clearest possible argument for evaluating on real data.

**Resolution limit.** At *n*=31 one span is ~0.011 F1 and run-to-run seed
variance is ~±0.015. Differences under ~0.02 are not measurable, and we report
them as such rather than claiming improvements we cannot detect.

**Confidence saturation.** Every predicted span returns tagging confidence
≥0.999995, including on deliberately ambiguous input. Templated training data
contains no genuine ambiguity, so the model never learned that uncertainty
exists. Consequence in #4.4.

### 6.3 The generator

Built from the observed distributions of the training-side conversations: real
unit vocabulary, real variant morphology, and the structural frequencies of
the corpus (48% of orders have no `ITEM`, 41% carry a `VARIANT`, 34% have no
`UNIT`, 21% have two or more `QTY`). Committed and documented as methodology,
with a character-offset assertion over every generated span.

**Elicitation was attempted and did not contribute evaluation data.** A
five-prompt form returned 19 responses, but the prompts were written for the
earlier sembako framing and none of the products or units transfer. What did
transfer was language-level: anaphoric surface forms, Banjar connectives
(`lawan`, `lwn`), address terms, and `se-` fusion. Three generator
configurations were then evaluated against the same held-out set and scored
0.895 / 0.878 / 0.883, inside the resolution limit, so no configuration is
measurably better and the baseline was retained. Span vocabulary remains 99.2%
real-corpus.

### 6.4 RAG baseline, not started yet

The organisers' 22 July clarification allows RAG, agentic workflows and tool
calling as alternatives to fine-tuning. We are building a comparison arm
deliberately, about two days of work, against the target task.

It buys three things: a comparison against a permitted architecture rather
than a strawman, the offline demonstration in #7, and a documented technology
decision where we implemented the permitted alternative, measured it, and
declined it on evidence.

---

## 7. Demonstration and deliverables

Offline comparison. Identical input, both systems, network disabled,
recorded. The RAG baseline fails and the local model responds. Since the
proof-of-work video cannot be cut, nobody can allege the demonstration was
spliced.

Clarification case. `Pesan resoles 50 biji` returns quantity resolved at
50 pieces, the SKU held open with three candidates, and a flag asking for the
variant, the same question the seller asks in the transcript. This is the
two-score design justifying itself in one response.

Transfer segment. About twenty seconds showing the same architecture on
the benchmark domain. Evidence for the claim in #1, not a second product.

| Deliverable | Requirement |
|---|---|
| GitHub repository | Public, `README.md` + working `docker compose` |
| Commit convention | Conventional Commits from the first commit, graded |
| Proof-of-work video | ≤7 min, unlisted, double screen, visible timestamp, **no cuts**, must show working and non-working flows |
| Promo video | ≤5 min, public, MP4, ≥720p, must depict the design process |
| Proposal | PDF, ≤20 pages excluding cover, bibliography, appendices |

---

## 8. Timeline

| Dates | Status |
|---|---|
| 29 Jul – 4 Aug | ~~Benchmark validation, B1 and B2~~ **done** |
| 5 Aug – 7 Aug | ~~Data collection, annotation, split~~ · ~~O1 training and export~~ · ~~normalizer, resolver, frontend~~ **done** |
| 8 Aug – 14 Aug | O2 distillation · RAG baseline · **proposal drafting** |
| 15 Aug – 17 Aug | Final measurements, integration, demo-hardware latency |
| **18 Aug – 25 Aug** | **Feature freeze, deliverables only** |

Track A finished ahead of schedule. The critical path is now deliverables:
promo video (15%) and proposal (15%) together outweigh technical
implementation (25%), and both are behind.

The temptation to keep building past 18 August is the most predictable way
this project loses points.

---

## 9. Decision points

| Date | Test | If we miss it |
|---|---|---|
| ~~4 Aug~~ | ~~Evaluation data collected~~ | **Met**, 31 real annotated messages |
| **14 Aug** | Distilled student converged and beats O1 | Ship O1, reframe the claim from "distilled" to "small fine-tuned model", report the comparison |
| **18 Aug** | Feature freeze | Unconditional |

On the 14 August. The PRD, README and pitch currently describe
distillation as the contribution. If O2 does not land, that language has to
change rather than quietly stand. "We fine-tuned a small model and measured it
carefully" is a weaker claim on the originality criterion, but it is the one
we would be able to defend.

---

## 10. Risks

| Risk | Severity | Status / mitigation |
|---|---|---|
| Deliverables compressed into the last week | **High** | **The critical-path risk.** 18 Aug freeze; proposal drafting starts now |
| Distillation not completed | High | It is the stated contribution; 14 Aug gate, with a defined fallback claim |
| Evaluation set too small to detect anything | High | Stated everywhere as ±0.02; no improvement claimed inside that band |
| Single-seller data does not generalize | Medium | Stated as a limitation; no generalization claimed |
| VARIANT class weak | Medium | Cause unidentified after two ablations; reported rather than hidden |
| ~~Pipeline does not converge or export~~ | ~~High~~ | **Closed** |
| ~~No evaluation data~~ | ~~High~~ | **Closed** real corpus obtained |
| ~~Student capacity insufficient~~ | ~~Medium~~ | **Closed** 11.7M within 0.007 F1 of 124M |

### 10.1 Components we know are improvable

Listed on purpose, since the MVP-readiness criterion asks for it.

- Order-line grouping uses a positional heuristic. Multi-item quantity
  attachment is unsolved and is where we expect most errors.
- `VARIANT` is the weakest class (0.611) in every run. The `di-` prefix
  hypothesis was tested at two ratios and neither improved it. Cause
  unidentified.
- Tagging confidence is saturated and uninformative.
- Evaluation covers only messages containing spans. Nothing measures whether
  the model correctly outputs nothing on ordinary chatter, the most likely
  live-demo failure.
- `ANAPHORIC` has 2 held-out instances. Not measurable.
- Cake orders are undersampled: 4 conversations, 1 in evaluation.
- Unit conversion covers only the pack units the catalog declares.
- Latency measured on Kaggle CPU, not demonstration hardware.

### 10.2 Designed but deferred

A distilled small language model invoked **only** on messages the grouping
heuristic flags as low-confidence would handle multi-item attachment without
touching the guarantees on the normal path. It is the natural extension of the
identified weak point and the right size for the 10-hour final round. Not in
scope now: it adds a second model to a container whose smallness is part of
the claim.

---

## 11. Rules compliance

| Requirement | Status |
|---|---|
| Model customization required | Fine-tuned on a task-specific 11-label schema; distillation outstanding |
| No live external integration | No messaging-platform API, input is a text area |
| Synchronous processing only | One inference call per request |
| `docker compose`, localhost | Model committed to the repository, CPU only, no network, no API key |
| Public or synthetic data | IndoNLU (public), generated training data, real evaluation data used with permission and pseudonymized |
| Work period 17 Jun – 25 Aug | New repository, no prior work incorporated |
| Conventional Commits | Enforced from the first commit |
| No institutional identification | Checked across videos, README, proposal and anything else published |

---

## 12. Questions we expect, and how we answer them

**"Conversational commerce platforms already automate ordering."**
They prevent free text by putting a form in front of it. We parse it instead.
The customer who cannot be put behind a form is precisely the one who reorders
every week, so that is the customer we built for.

**"Why not just call a hosted model API?"**
Demonstrated offline with the network disabled, on camera. Beyond that, order
messages are personal data under UU PDP, and per-call pricing does not survive
Rupiah-margin trade.

**"Is the small model actually good enough?"**
Measured: 0.8014 F1 at 11.7M parameters against 0.8088 at 124M on IndoNLU
NERP. 10.6× the parameters for 0.007 F1.

**"Why not a small generative model instead of a tagger?"**
Bounded output is what makes an 11 MB model sufficient, and pointer-into-input
is a structural hallucination guarantee that generation cannot give. At this
size, generating schema-valid JSON is a much harder learning problem than
picking one of 11 labels per token.

**"This is one narrow task."**
The task is the demonstration. The contribution is a method for offline
Indonesian extraction, measured across two domains rather than asserted.

**"Your evaluation set is 31 messages."**
Correct, and we say so everywhere including in the badge. It is real
operational data from a real business rather than a larger set of authored
approximations, and we state the resolution limit rather than claiming
improvements inside it.

---

## 13. Scoring reference

| Criterion | Weight |
|---|---|
| Technology implementation & architecture maturity | 25% |
| Originality & social impact | 20% |
| MVP readiness for final | 15% |
| Promo video | 15% |
| Proposal quality & development process | 15% |
| Theme relevance | 10% |
| Business value & governance (bonus) | 3.5% |
| AIC Talks attendance (bonus) | 1.5% |
| **Total** | **105%** |

---

## 14. Open decisions

| # | Decision | When |
|---|---|---|
| 1 | Whether O2 ships, and the consequent wording of the contribution claim | 14 Aug |
| 2 | Whether to annotate non-order evaluation messages, closing the biggest live-demo risk | Before 17 Aug |
| 3 | Catalog source for the demo: partner seller or constructed | Before video production |

---

## 15. References

- IndoNLU benchmark, huggingface.co/datasets/indonlp/indonlu · aclanthology.org/2020.aacl-main.85.pdf
- IndoBERT, huggingface.co/indobenchmark
- NusaX, huggingface.co/datasets/indonlp/NusaX-senti
- Sahabat-AI, huggingface.co/GoToCompany/llama3-8b-cpt-sahabatai-v1-instruct
- UU PDP No. 27/2022, in force 17 Oct 2024
- Measurements, methodology and caveats — `docs/RESULTS.md`
