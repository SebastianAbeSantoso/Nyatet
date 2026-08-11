# Nyatet: Product Requirements Document

**COMPFEST 18 · AI Innovation Challenge · Smart Commerce**

**Status, 10 August 2026.** The order-parsing pipeline runs end to end and is
shipped. Distillation was evaluated and not shipped. The RAG comparison is
outstanding. Deliverables (proposal and both videos) are the critical path.

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
architecture measured across two domains and three model configurations.

One thing to keep straight in everything we write externally: **the claim is
size and offline capability, not speed.** A 124M model and an 11.7M model land
at identical latency once quantized (§6.1). Speed comes from *depth*, not from
parameter count (§6.4). Conflating those would be overclaiming.

### 1.1 What ships and what does not

Order parsing is the only interactive flow. The public benchmark domain is
evaluation only and never appears in the interface. Variety shows up in the
results table and for about twenty seconds of the video. A second interactive
flow would breach the MVP scope cap, so there is not one.

The distilled student is also not shipped. It is retained as measured evidence
for an architectural finding, not as a deployment candidate (§6.4).

---

## 2. Demonstration domain

**Who.** A home food producer in Banjarmasin who takes orders in person and
over WhatsApp. She makes risol in three preparations (mentah, goreng, frozen),
sells them by the piece or by the box, and also takes occasional cake orders.

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

**Why existing tools do not reach this user.** Indonesian conversational
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
| Implicit product | `10 biji ga adakah` | 14 of 29 annotated orders name no product at all |
| Banjar counting units | `buting`, `biji`, `kotak` | `buting` is a Banjar classifier absent from general corpora |
| Fused particles | `butinglh`, `mentahnya`, `20bj` | Morphology attaches to the span itself |
| Detached variants | `25 nya di goreng 25 yg mentah` | Preparation state arrives separately from the product |
| Anaphora | `ky biasa`, `kaya kmrn` | Requires noticing what is *not* there |

### 3.2 Why on-device

Three drivers, in the order we would argue them.

**Personal data.** Order messages carry names, phone numbers, addresses. UU
PDP No. 27/2022 has been binding since 17 October 2024, with fines up to 2% of
annual revenue. Processing on-device removes the transfer and the third-party
processing entirely. Worth being careful: the supervisory body's structure is
still unsettled, so what we can claim is that we have eliminated a category of
risk, not that we have cleared a specific enforcement hurdle.

**Connectivity.** 3,029 Indonesian villages are still internet blank spots as
of 2026. The headline figure of roughly 95% mobile broadband coverage
overstates how much of that is usable.

**Unit economics.** Per-call API pricing does not survive Rupiah-margin trade.

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

`VARIANT` is a separate type rather than folded into `ITEM`. It appears in 12
of 29 annotated orders and frequently arrives **detached** from the product:

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
schema without a measurable payoff at this evaluation size.

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
information, see §6.5. Resolution confidence is unaffected. The two-score
architecture holds; one of the two scores is a constant and is not surfaced in
the interface.

### 4.5 Out-of-scope input returns a result, not a failure

The MVP handles standalone order messages. Two classes of input it cannot
resolve are detected and flagged rather than guessed at:

- **Anaphoric references**, emitted as a span class, flagged for review.
- **Standing orders**: weekly day-to-quantity schedules
  (`selasa - jumat 34, sabtu 38`) from a reseller. Detected by a deterministic
  regex. A recurring schedule is not one order.

This matters for the proof-of-work video, which must show a non-working flow.

### 4.6 Model chain

| Stage | Artifact | Ships |
|---|---|---|
| Data generation | Phenomenon-weighted generator, built from real-data distributions | No |
| Model | `indobert-lite-base-p2`, fine-tuned | Yes |
| Export | ONNX int8, pre-processed, MatMul + Gather | Yes |
| Resolver | Fuzzy match over catalog CSV | Yes |

**Distillation was run and measured (§6.4), and the result changed the plan.**
`indobert-base-p2` was fine-tuned as teacher and distilled into a 4-layer
student. The student lost 0.027 F1 for a latency advantage that is
imperceptible at this workload, so the shipped model is the directly
fine-tuned one.

A Sahabat-AI 8B teacher was attempted earlier and abandoned. The architecture
loads and LoRA attaches, but fp16 compute produces non-finite gradients from
the freshly-initialized classification head and fp32 compute exceeds memory
during loading, and Turing provides no native bf16. Documented in
`RESULTS.md` C7.

### 4.7 Student architecture, decided on measurement

Benchmark results established that parameter count determines file size and
not inference speed. IndoBERT-lite and IndoBERT-base both execute 12
transformer layers at 768 hidden; ALBERT gets its parameter reduction by
sharing one layer's weights across all twelve, not by doing less work.
Measured latency was 31.0 ms against 31.5 ms, no difference.

The distillation student was therefore specified as a reduced-layer,
non-shared BERT: 4 layers at 384 hidden. Depth removes compute; parameter
sharing does not. §6.4 confirms this directly: the same task, 9× faster.

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
background jobs, second flows, and a model picker.

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
publicly obtainable, since that competition is closed and access is
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
| Annotated | 81 messages: 40 with spans, 41 non-order negatives, 98 spans total |
| Training data | ~10,800 generated rows (8,000 orders + 35% negatives) |

Splitting by conversation rather than by message prevents a customer's
phrasing appearing on both sides. Three standing-order conversations are
excluded from evaluation and retained as negatives.

**Annotation policy: spans mark entity mentions, not order content.** The
model sees one message with no history, so `mentah` in `Enggeh mentah` and in
`risol mentah 20 biji` are indistinguishable to it. Which mentions become
order lines is decided by the resolver, not by the annotation.

**O1 results**, `indobert-lite-p2`, int8 v3, evaluation set *n*=81:

| | |
|---|---|
| F1 | 0.837 |
| Per class | QTY 0.932 · UNIT 0.930 · ITEM 0.900 · VARIANT 0.583 |
| False positives on non-orders | 5 of 41 messages (12%), 10 spurious tokens |
| Size | 11.00 MB |
| Latency, 1 thread | 21.6 ms median, 22.6 ms p95 |
| Synthetic held-out split | 1.0000, diagnostic only |

`seqeval` scores only the messages that carry reference spans, so the 41
negatives contribute nothing to the F1 above and are measured separately as
the false-positive row.

The synthetic split scoring 1.0000 is not a good sign. It means the generator's
held-out data contains nothing its training data did not already teach, and it
is the same fact that produces the confidence saturation below. It is the
clearest possible argument for evaluating on real data.

**The 12% false-positive rate has an identified cause.** Numbers in time and
price contexts (`jam 7.30`, `55 ribu`) get tagged `QTY`, and the token `yg`
triggers `ANAPHORIC`. Both are generator gaps: times appear only as
`jam 7 pagi diambil`, and most anaphoric forms begin `yg`/`ky`/`kaya`. Not
fixed, since changing the generator would invalidate the §6.4 comparison.

**Resolution limit.** At *n*=81 one span is ~0.010 F1 and run-to-run seed
variance is ~±0.015. Differences under ~0.02 are not measurable, and we report
them as such rather than claiming improvements we cannot detect.

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

### 6.4 Distillation, complete but not shipped

`indobert-base-p2` fine-tuned on the same order data as teacher, distilled
into a 4-layer non-shared BERT at 384 hidden. Same generator, same seed, one
variable changed.

| | Params | Layers | F1 | Size | Median |
|---|---|---|---|---|---|
| Teacher · base-p2 | 124 M | 12 | 0.9016 | 118.80 MB | 24.0 ms |
| **Fine-tuned · lite-p2, shipped** | **11.7 M** | **12** | **0.9022** | **11.00 MB** | **21.6 ms** |
| Student · distilled | ~20 M | **4** | 0.8756 | 25.45 MB | **2.4 ms** |

All three rows were measured on the earlier 31-message evaluation set, before
it was expanded to 81. They are comparable to each other but not to the 0.837
reported in §6.2 as the shipped model's final result.

Two findings, in opposite directions:

**Depth determines speed, where parameter count does not.** Cutting 12 layers
to 4 gives 2.4 ms against 21.6 ms, 9× faster. This is §6.1's finding
confirmed from the other side, on the target task rather than the benchmark.

**A 124M teacher does not outperform an 11.7M model on this task.** 0.9016
against 0.9022 under matched hyperparameters. §6.1 replicated on the target
domain.

Note the student is *larger* than the shipped model (25.45 MB vs 11.00) while
being 9× faster. Fewer layers, but no cross-layer weight sharing, so its
parameters sit in storage rather than in compute. The same point from a third
angle.

**Decision: the fine-tuned model ships.** The student's 0.027 deficit is
outside the resolution limit, so it is a real difference. At sixty messages a
night the gap between 21.6 ms and 2.4 ms is imperceptible; 0.027 F1 is not.

The student's weakness is concentrated in `ANAPHORIC` (0.286, 2 true positives
against 10 false). Every content class held, and `VARIANT` was actually better
on the student. We did not investigate, because fixing it would require
changing the generator and that would invalidate the comparison.

### 6.5 Findings that limit claims

**Confidence saturation.** Every predicted span returns tagging confidence
≥0.999995, including on deliberately ambiguous input. Templated training data
contains no genuine ambiguity, so the model never learned that uncertainty
exists. Consequence in §4.4.

**Implicit products have two different causes.** Sometimes the product was
never stated, because the seller makes one thing. Sometimes it was stated in an
earlier message and the customer is completing an order across turns
(`Bagoreng`, `Enggeh mentah`). The resolver treats both the same way and is
correct for the first, coincidentally correct for the second. A broader
catalog would need conversation history, which the MVP does not carry.

### 6.6 RAG baseline, outstanding

The organisers' 22 July clarification allows RAG, agentic workflows and tool
calling as alternatives to fine-tuning. We are building a comparison arm
deliberately, about two days of work, against the target task.

It buys three things: a comparison against a permitted architecture rather
than a strawman, the offline demonstration in §7, and a documented technology
decision where we implemented the permitted alternative, measured it, and
declined it on evidence.

---

## 7. Demonstration and deliverables

**Offline comparison.** Identical input, both systems, network disabled,
recorded. The RAG baseline fails and the local model responds. Since the
proof-of-work video cannot be cut, nobody can allege the demonstration was
spliced.

**Clarification case.** `Pesan resoles 50 biji` returns quantity resolved at
50 pieces, the SKU held open with three candidates, and a flag asking for the
variant: the same question the seller asks in the transcript. This is the
two-score design justifying itself in one response.

**Failure case.** The rules require showing a non-working flow. One of the
measured false positives, a time or price expression tagged as a quantity, is
honest, on-brand, and already documented.

**Transfer segment.** About twenty seconds showing the same architecture on
the benchmark domain. Evidence for the claim in §1, not a second product.

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
| 8 Aug – 10 Aug | ~~O2 distillation~~ · ~~expanded evaluation set (O1b)~~ · ~~document sync~~ **done** |
| 11 Aug – 14 Aug | RAG baseline · **video planning and recording** |
| 15 Aug – 17 Aug | Final measurements, demo-hardware latency, integration check |
| **18 Aug – 25 Aug** | **Feature freeze, deliverables only** |

Model work finished ahead of schedule. The critical path is now deliverables:
promo video (15%) and proposal (15%) together outweigh technical
implementation (25%), and the videos have not started.

The temptation to keep building past 18 August is the most predictable way
this project loses points.

---

## 9. Decision points

| Date | Test | Outcome |
|---|---|---|
| ~~4 Aug~~ | ~~Evaluation data collected~~ | **Met**: 81 real annotated messages |
| ~~14 Aug~~ | ~~Distilled student beats the fine-tuned model~~ | **Not met**, resolved early on 10 Aug: 0.8756 vs 0.9022 (§6.4). Fine-tuned model ships; contribution wording changed from "distilled" to "small fine-tuned model, compared against a distilled alternative" |
| **18 Aug** | Feature freeze | Unconditional |

---

## 10. Risks

| Risk | Severity | Status / mitigation |
|---|---|---|
| Videos not started, 8 days to freeze | **High** | **The critical-path risk.** Shot sequence planned §7; no-cuts format needs rehearsal |
| Evaluation set too small to detect anything | High | Stated everywhere as ±0.02; no improvement claimed inside that band |
| Single-seller data does not generalize | Medium | Stated as a limitation; no generalization claimed |
| VARIANT class weak | Medium | Cause unidentified after two ablations; reported rather than hidden |
| ~~Pipeline does not converge or export~~ | ~~High~~ | **Closed** |
| ~~No evaluation data~~ | ~~High~~ | **Closed** |
| ~~Student capacity insufficient~~ | ~~Medium~~ | **Closed**, 11.7M within 0.007 F1 of 124M |
| ~~Distillation not completed~~ | ~~High~~ | **Closed**, completed, measured, not shipped |
| ~~Silent behaviour on non-orders unmeasured~~ | ~~High~~ | **Closed**, 12% false-positive rate, cause identified |

### 10.1 Components we know are improvable

Listed on purpose, since the MVP-readiness criterion asks for it.

- 12% of non-order messages get a spurious span, concentrated on numbers in
  time and price contexts and on `yg` triggering `ANAPHORIC`. Cause identified,
  fix deferred to protect the §6.4 comparison.
- `VARIANT` is the weakest class (0.583 at *n*=81, 0.611 on the earlier
  31-message set) in every run. The `di-` prefix hypothesis was tested at two
  ratios and neither improved it. Cause unidentified.
- Order-line grouping uses a positional heuristic. Multi-item quantity
  attachment is unsolved and is where we expect most errors.
- Cross-turn orders resolve to the primary product by coincidence, since this
  seller has one. A broader catalog would need conversation history.
- Tagging confidence is saturated and uninformative.
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
| Model customization required | Fine-tuned on a task-specific 11-label schema; distillation also implemented and measured |
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
Measured twice. On IndoNLU NERP: 0.8014 at 11.7M against 0.8088 at 124M. On
the target task: 0.9022 against 0.9016. 10.6× the parameters, no gain.

**"Why not a small generative model instead of a tagger?"**
Bounded output is what makes an 11 MB model sufficient, and pointer-into-input
is a structural hallucination guarantee that generation cannot give. At this
size, generating schema-valid JSON is a much harder learning problem than
picking one of 11 labels per token.

**"You said distillation was the contribution."**
It was the plan. We ran it, measured it, and the student lost 0.027 F1 for a
latency advantage that is imperceptible at this workload. We changed the
claim, not the data. The student is reported in full as evidence for the
depth-versus-parameters finding.

**"This is one narrow task."**
The task is the demonstration. The contribution is a method for offline
Indonesian extraction, measured across two domains and three model
configurations rather than asserted.

**"Your evaluation set is 81 messages."**
Correct, and we say so everywhere including in the badge. It is real
operational data from a real business rather than a larger set of authored
approximations, and we state the resolution limit rather than claiming
improvements inside it. 41 of those messages are non-orders, so the set also
measures whether the model correctly stays silent, which the earlier
31-message version could not.

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
| 1 | Whether the RAG baseline gets built, or the video ships without the offline comparison | 14 Aug |
| 2 | Catalog source for the demo: partner seller or constructed | Before video production |
| 3 | Whether to re-measure the §6.4 distillation comparison on the 81-message set | Before 17 Aug, only if time permits |

---

## 15. References

- IndoNLU benchmark: huggingface.co/datasets/indonlp/indonlu · aclanthology.org/2020.aacl-main.85.pdf
- IndoBERT: huggingface.co/indobenchmark
- Hinton, Vinyals & Dean (2015), *Distilling the Knowledge in a Neural Network*
- Sahabat-AI: huggingface.co/GoToCompany/llama3-8b-cpt-sahabatai-v1-instruct
- UU PDP No. 27/2022, in force 17 Oct 2024
- Measurements, methodology and caveats: `docs/RESULTS.md`
