# Nyatet — Product Requirements Document

**COMPFEST 18 · AI Innovation Challenge · Smart Commerce**

**Status:** Week-1 pipeline validation is done (p7.2). Elicitation hasn't started.

---

## 1. Thesis

Nyatet is a way to run capable Indonesian NLP where there's no connectivity and no budget for per-call inference. A distilled span-tagging model, small enough to ship and run offline on ordinary hardware, replaces a hosted large-model API for pulling structure out of informal Indonesian text.

We ship one interaction flow: trade order parsing. The method behind it isn't specific to that task, which is why we also measure it on a public benchmark. So the contribution is the task formulation plus distillation plus offline deployment, the demonstration is order parsing, and the evidence is the same architecture measured across more than one domain.

One thing to keep straight in everything we write externally: **the claim is size and offline capability, not speed.** Week-1 measurement showed a 124M model and an 11.7M model landing at identical latency once quantized (p7.3). The speedup came from quantization; the size came from architecture. Both are useful, but they're separate claims and conflating them would be overclaiming, so we state them separately throughout.

### 1.1 What ships and what doesn't

Trade order parsing is the only interactive flow, evaluated in p7.1. The public benchmark domains are evaluation only (p7.2) and never appear in the interface.

Variety shows up in the results table and for about twenty seconds of the video. A second interactive flow would breach the MVP scope cap, so there isn't one.

---

## 2. Demonstration domain

**Who.** Order admin at a sembako grosir or FMCG distributor with 40 to 80 regular trade customers who reorder weekly by text message.

**When it hurts.** 22:00. Sixty incoming messages, each written the way that customer has always written it, transcribed by hand into an order book before the 05:00 loading run. Mistakes here turn into wrong deliveries at 06:00.

**Why existing tools don't reach this user.** Indonesian conversational-commerce platforms (Dazo, WATI, Qiscus, ManyChat) automate ordering by constraining the input: catalog buttons, forms, guided flows. That works for a consumer browsing a webstore. It doesn't work for a trade customer who's bought the same twelve items from the same supplier for three years and won't click through a catalog to do it again.

The one customer you can't put behind a form is the one who reorders every week, so Nyatet parses the free text instead of preventing it.

---

## 3. Why the model has to be small and local

### 3.1 Language asymmetries

| Phenomenon | Example | Why a general model struggles |
|---|---|---|
| Brand-as-category | `indomi` = any instant noodle | Resolution depends on *this seller's* catalog |
| Trade units | `dus`, `renceng`, `kodi`, `jerigen` | Pack size varies per product, no global convention |
| Code-switching | Banjar/Javanese fragments mid-sentence | Underrepresented in multilingual pretraining |
| Ellipsis and anaphora | `yg biasa`, `itu aja` | Requires noticing what *isn't* there |

### 3.2 Why on-device

Three drivers, in the order we'd argue them.

**Personal data.** Order messages carry names, phone numbers, addresses. UU PDP No. 27/2022 has been binding since 17 October 2024, with fines up to 2% of annual revenue. Processing on-device removes the transfer and the third-party processing entirely.

Worth being careful here: the supervisory body's structure is still unsettled, so what we can claim is that we've eliminated a category of risk, not that we've cleared a specific enforcement hurdle.

**Connectivity.** 3,029 Indonesian villages are still internet blank spots as of 2026. The headline figure of roughly 95% mobile broadband coverage overstates how much of that is usable.

**Unit economics.** Per-call API pricing doesn't survive Rupiah-margin trade at sixty messages a night.

None of these are uniquely Indonesian conditions. Thin margins and patchy connectivity are the majority condition globally, and the method transfers wherever hosted inference is unaffordable or unreachable.

---

## 4. Architecture

```
raw text
  → span tagger          BIO tagging, distilled encoder, the only model
  → normalizer           deterministic rules, no model
  → reference resolver   fuzzy match against a local file
  → structured output + two independent confidence scores
```

Every stage is domain-agnostic. Retargeting means changing three inputs and never the pipeline:

| | Trade orders | Benchmark domains |
|---|---|---|
| Span types | `ITEM`, `QTY`, `UNIT`, `ANAPHORIC` | Dataset-defined |
| Normalizer rules | Numerals, fractions, unit aliases | Not applicable |
| Reference file | Seller's catalog CSV | Not applicable |

### 4.1 Tagging, not generation

The model emits BIO spans over the original text instead of generating structured output. Three consequences:

- Every output token points into the input, so the model can't emit an item that isn't in the message.
- The output space is bounded, which is what makes the task fit a small encoder instead of a multi-billion-parameter API call.
- Resolution becomes a separate stage we can swap out.

The conventional approach, prompting a large model to generate JSON, needs scale precisely because generation is unconstrained. Bounding the output space is what makes a small model sufficient, and that's the central technical claim of the project.

### 4.2 Span schema

| Type | Example |
|---|---|
| `ITEM` | `indomi goreng` |
| `QTY` | `1`, `dua`, `seperempat` |
| `UNIT` | `dus`, `kg`, `renceng` |
| `ANAPHORIC` | `yg kyk kmrn`, `yg biasa`, `itu aja` |

`PACK_SIZE`, `PAYMENT_NOTE` and `LOCATION` are deferred to the final-round increment. Each needs elicitation data we don't have yet.

### 4.3 Deterministic normalization

Quantity arithmetic isn't the model's job.

Word numerals and fractions resolve by lookup and regex (`dua` to 2, `seperempat` / `¼` / `1/4` to 0.25). Pack size is a catalog column rather than language knowledge, since a `dus` of instant noodles and a `dus` of detergent hold different amounts and only the catalog knows that. So the model tags the span, the normalizer converts the number, and the catalog supplies the multiplier.

A useful side effect: deterministic components are unit-testable and can't fail unpredictably in the middle of an uncut recording.

### 4.4 Two independent confidence scores

Parsing and resolution fail independently, so we report them separately.

| Score | Source | Failure it catches |
|---|---|---|
| Tagging confidence | Span probability | Didn't recognize an entity was there |
| Resolution confidence | Top-1 vs top-2 retrieval margin | Recognized it, can't tell which SKU |

`indomi` tags at about 0.99 and resolves at about 0.13, because *Goreng*, *Soto* and *Ayam Bawang* are near-tied. A single conflated score would report that as bad parsing when the parsing was fine. The interface state we want is "entity recognized, needs disambiguation," with the candidates shown.

### 4.5 Anaphora as output

The MVP handles standalone reorder messages only. Multi-turn conversational orders are out of scope, and no parser resolves them without order history anyway.

Because `ANAPHORIC` is an emitted span class, out-of-scope input comes back as a defined result rather than a failure:

> Reference to prior order detected. Cannot resolve without history, flagged for review.

This matters for the video, where we have to show a non-working flow.

### 4.6 Model chain

| Stage | Artifact | Ships |
|---|---|---|
| Data generation | Phenomenon-weighted template generator, seeded from elicited messages | No |
| Teacher | Sahabat-AI `llama3-8b-cpt-sahabatai-v1-instruct`, fine-tuned | No |
| Student | Reduced-layer BERT, see p4.7 | Yes |
| Export | ONNX, int8 dynamic quantization with `MatMulConstBOnly=False` | Yes |
| Resolver | Fuzzy match over catalog CSV | Yes |

There's precedent for this: LazarusNLP has demonstrated cross-architecture distillation into `indobert-lite` for Indonesian NLI. Our methodological anchor is Distilling Step-by-Step (Hsieh et al., 2023), where a 770M model beats few-shot 540B PaLM on a target task.

The distillation stage is the contribution and it isn't optional. A directly fine-tuned student is a routine result. Teacher-to-student distillation adds nothing the user can see, which is exactly why it's tempting to cut under time pressure, and it's also the thing that separates a method from an exercise. See the 10 August decision point in p10.

### 4.7 Student architecture, decided on measurement

Week-1 results (p7.2) established that parameter count determines file size and not inference speed. IndoBERT-lite and IndoBERT-base both execute 12 transformer layers at 768 hidden; ALBERT gets its parameter reduction by sharing one layer's weights across all twelve, not by doing less work. Measured latency was 18.3 ms against 18.1 ms, which is no difference at all.

So the distillation student is specified as a reduced-layer, non-shared BERT, roughly 4 layers with reduced hidden width. It's depth that removes compute, not parameter sharing. Non-shared weights have a second advantage: they quantize cleanly without the workaround we needed in p7.3.

**Size policy.** External claims state the property ("runs offline on low-cost hardware") and never the parameter count. Once we say a number, someone will compare it to a number from a different architecture and the comparison won't mean anything.

---

## 5. Product scope

Constrained by the MVP scope cap: one interaction flow, synchronous request/response, `docker compose`, localhost.

The interface is one text area, one button, one result panel. Paste a single trade reorder message, get back tagged spans, resolved order lines, and two confidence scores per line.

Deliberately excluded: dashboards, history, authentication, analytics, background jobs, second flows.

---

## 6. Data

### 6.1 Target domain, elicited evaluation set

Target is 150 messages from about 30 respondents, for evaluation only and never for training.

The number comes from what we need to report:

| Purpose | Messages |
|---|---|
| Establish phenomenon taxonomy | ~50 |
| Report aggregate F1 credibly | 120–150 |
| Report per-phenomenon breakdown | ~30 × 5 classes = 150 |

**Method.** Respondents fill in a five-prompt form, one prompt per phenomenon class, writing each message the way they'd actually send it. Five messages each, about five minutes. Thirty respondents gets us 150 messages with coverage balanced by construction. The instrument is in Appendix A.

Two constraints govern whether this set is valid. No team member contributes entries, because this is held-out. And respondents are told to write fast and not proofread, since careful composition is the main reason authored data comes out cleaner than the real thing.

**Why elicited rather than harvested.** Harvested messages contain real customer personal data, which is inconsistent with a privacy-first product and would put consent and redaction work on the critical path. Elicited data is synthetic by construction.

We should be upfront that authored messages are still cleaner than operational ones. Results get reported as *n*=150 with per-class counts stated.

### 6.2 Phenomenon taxonomy

| Class | Example | Handling |
|---|---|---|
| Explicit | `indomi goreng 1 dus` | Baseline |
| Anaphoric | `yg kyk kmrn` | `ANAPHORIC` tag, then flag |
| Underspecified attribute | `1 dus yg gede` | Catalog + default |
| Quantity elided | `minyak sama gula aja` | Default quantity, marked as defaulted |
| Discourse marker | `itu aja` | Recognized as non-content |
| Numeral variant | `dua dus`, `¼` | Normalizer |

Real messages mix these. The generator produces multi-phenomenon messages including greetings, sign-offs and trailing questions as untagged noise, because a model trained only on bare order lines will try to tag them.

### 6.3 Target domain, training data

Phenomenon-weighted template generation seeded from the elicited set, with the fine-tuned teacher paraphrasing and verifying. SKU vocabulary is grounded in a public Tokopedia listings dataset (29,519 real product titles). Code-switching lexicon comes from NusaX, which covers Banjarese. The generator gets committed and documented as methodology, with a character-offset assertion over every generated span.

### 6.4 Benchmark domain

IndoNLU NERP: 6,720 train / 840 validation / 840 test sentences, 11 IOB labels covering person, place, industry, event, and food & beverage, drawn from Indonesian news text. Freely available and loaded programmatically.

We picked it over the Shopee Code League address dataset, which turned out not to be publicly obtainable any more; that competition is closed and access is invitation-gated. IndoNLU is the better choice regardless, because it publishes performance-versus-model-size trade-offs for sequence labeling, which is this project's thesis already expressed as a peer-reviewed baseline.

---

## 7. Evaluation

### 7.1 Target domain, primary results

**Pending elicited data.** We can't report any target-domain result until the p6.1 set exists. The synthetic held-out split is diagnostic only and isn't reportable, because it measures the generator's internal consistency rather than model quality.

| System | Accuracy | p95 latency | Size |
|---|---|---|---|
| Teacher (Sahabat-AI, fine-tuned) | | | |
| Student (distilled) | | | |
| Student (ONNX int8), **shipped** | | | |
| RAG baseline (p7.4) | | | |

Accuracy gets reported in three parts: tagging F1 per span type, resolution accuracy@1, and end-to-end order-line accuracy.

### 7.2 Benchmark domain, completed results

IndoNLU NERP. Identical training and export code across both models, only the checkpoint name differs.

| | IndoBERT-lite-p2 (11.7M) | IndoBERT-base-p2 (124M) |
|---|---|---|
| F1, fp32 | 0.8040 | 0.8077 |
| F1, int8 (default) | 0.8076 | 0.8077 |
| F1, int8 (full) | 0.8014 | 0.8088 |
| Size, fp32 | 42.6 MB | 472.7 MB |
| Size, int8 (default) | 38.2 MB | 118.9 MB |
| Latency, fp32 | 29.8 ms | 47.2 ms |
| **Latency, int8 (full)** | **18.3 ms** | **18.1 ms** |

**1. Scale buys almost nothing on this task.** A 10.6× parameter increase yields 0.004 F1. This is the project's thesis, measured on a public benchmark, which is the strongest single result we have.

**2. Parameter count determines size, not speed.** Both models land at roughly 18 ms once quantized. ALBERT's parameter sharing reduces storage while still executing all 12 layers. Every bit of measured speedup came from quantization (47.2 to 18.1, and 29.8 to 18.3) and none of it from model selection. p4.7 revises the student architecture on the back of this.

**3. Accuracy is quantization-invariant.** The full spread across five variants is 0.8014 to 0.8106, which is inside run-to-run variance. No variant is meaningfully better than another.

**Per-class weakness.** On base-p2: EVT 0.388 F1 (n=130), IND 0.737, PLC 0.820, FNB 0.853, PPL 0.920. Event spans are the weakest class in both models.

**Metric caveat.** These are `seqeval` span-level strict scores. IndoNLU reports sequence-labeling tasks with word-level metrics, so our figures are **not directly comparable** to their published table without aligning the metric first. Any comparison in the proposal has to say so.

Two things still outstanding on this table. We never recorded `int8_v3` file sizes for either model, which needs fixing before publication. And the latency figures are means over 50 runs, which should be re-measured as median and p95 over 100 runs on a frozen input string and named hardware.

### 7.3 Quantization finding

Default `quantize_dynamic` on IndoBERT-lite gave only a 10% size reduction against an expected 4×, which cost us a while to work out.

Inspecting ONNX initializer data types traced it: ALBERT's cross-layer parameter sharing makes the exporter emit transformer weights as anonymous shared graph constants (`onnx::MatMul_*`), each referenced by twelve consumers. ONNX Runtime's default dynamic quantizer skips those, so all it actually quantized was the embedding table.

Setting `extra_options={"MatMulConstBOnly": False}` with `op_types_to_quantize=["MatMul"]` recovered full quantization of both embeddings and transformer weights at no accuracy cost.

We report this as a methodology result rather than a war story: the problem was measured, diagnosed at the architecture level, corrected, and the correction verified against accuracy.

### 7.4 RAG baseline

The organisers' 22 July clarification allows RAG, agentic workflows and tool calling as alternatives to fine-tuning. We're building a comparison arm deliberately, about two days of work, against the target task.

It buys three things: a comparison against a permitted architecture rather than a strawman, the offline demonstration in p8, and a documented technology decision, where we implemented the permitted alternative, measured it, and declined it on evidence.

---

## 8. Demonstration and deliverables

**Offline comparison.** Identical input, both systems, network disabled, recorded. The RAG baseline fails and the distilled model responds locally. Since the proof-of-work video can't be cut, nobody can allege the demonstration was spliced.

**Transfer segment.** About twenty seconds showing the same architecture on the benchmark domain with the p7.2 numbers on screen. It's evidence for the claim in p1, not a second product.

| Deliverable | Requirement |
|---|---|
| GitHub repository | Public, `README.md` + working `docker compose` |
| Commit convention | Conventional Commits from the first commit, graded |
| Proof-of-work video | ≤7 min, unlisted, double screen, visible timestamp, **no cuts**, must show working and non-working flows |
| Promo video | ≤5 min, public, MP4, ≥720p, must depict the design process |
| Proposal | PDF, ≤20 pages excluding cover, bibliography, appendices |

---

## 9. Timeline

| Dates | Track A, pipeline | Track B, data |
|---|---|---|
| 29 Jul – 4 Aug | ~~Benchmark domain: train → export → quantize → measure~~ **done** · RAG baseline | **Publish elicitation form, collect 150 messages** |
| 5 Aug – 11 Aug | Teacher fine-tune, generator, student distillation | Label elicited set, assemble catalog CSV |
| 12 Aug – 16 Aug | Quantize, export, integrate | Frontend, resolver, `docker compose` |
| 17 Aug | Comparison runs, measurement | — |
| **18 Aug – 25 Aug** | **Feature freeze, deliverables only** | Proposal, both videos |

Week-1 pipeline work finished ahead of schedule, so Track A is ahead and Track B hasn't started. Track B is the only thing on the critical path right now.

### 9.1 Why we ran the benchmark domain first

The pipeline is identical across domains, so the high-risk questions (does training converge, does export work, what's the real latency) were answerable on free public data before we had any dependency on elicited data. All three are answered now and pipeline risk is closed. This is worth a line in the proposal, since sequencing decisions are part of what p14 grades as development process.

### 9.2 Deliverable reserve

Promo video (15%) and proposal (15%) together outweigh technical implementation (25%). The 18 August freeze isn't negotiable, and the temptation to keep building past it is the most predictable way this project loses points.

---

## 10. Decision points

| Date | Test | If we miss it |
|---|---|---|
| **4 Aug** | 150 elicited messages collected | Cut the target to ~60, report per-class results as indicative rather than measured |
| **10 Aug** | Distilled student converged | Ship the directly fine-tuned student, reframe around the p7.2 and p7.3 findings |
| **18 Aug** | Feature freeze | Unconditional |

**Degraded fallback.** If elicitation fails completely, we submit the same product trained on synthetic data only, evaluated against whatever elicited messages exist, with the sample size stated plainly. That's a weaker submission but it's still on theme and still demoable. There's no second product to fall back on, so this is the floor.

---

## 11. Risks

| Risk | Severity | Status / mitigation |
|---|---|---|
| No elicited data by 4 Aug | **High** | **The only critical-path risk.** Five-minute form, no business recruitment, degraded fallback in p10 |
| Elicited data unrepresentative | High | Phenomenon-primed elicitation, external respondents, honest reporting |
| Distillation skipped under time pressure | High | It's the contribution; protected by the 10 Aug decision point |
| Deliverable compression | High | 18 Aug freeze |
| Resolution ambiguity dominates errors | Medium | Two-score design surfaces it as an interface state |
| ~~Pipeline doesn't converge or export~~ | ~~High~~ | **Closed**, p7.2 |
| ~~Student capacity insufficient~~ | ~~Medium~~ | **Closed**, 11.7M within 0.004 F1 of 124M |

### 11.1 Components we know are improvable

Listing these on purpose, since the MVP-readiness criterion asks for it.

- Order-line grouping uses a positional heuristic. Multi-item quantity attachment is unsolved and is where we expect most errors to come from.
- Unit conversion only covers the item's declared pack unit.
- The elicited evaluation set is small (*n*≈150) and authored rather than operational.
- Benchmark-domain evaluation doesn't exercise the normalizer or resolver stages, so those numbers say nothing about end-to-end accuracy.
- Event-type spans are the weakest benchmark class (EVT 0.388 F1) and we didn't attempt a mitigation.
- Latency figures need re-measuring as median and p95 on named hardware.

---

## 12. Rules compliance

| Requirement | Status |
|---|---|
| Model customization required | Teacher fine-tuning plus distillation, two training stages |
| No live external integration | No messaging-platform API, input is a text area |
| Synchronous processing only | One inference call per request |
| `docker compose`, localhost | Model committed to the repository, CPU only, no network, no API key |
| Public or synthetic data | IndoNLU (public), Tokopedia listings (public), elicited set (authored) |
| Work period 17 Jun – 25 Aug | New repository, no prior work incorporated |
| Conventional Commits | Enforced from the first commit |
| No institutional identification | Checked across videos, README, proposal and anything else published |

---

## 13. Questions we expect, and how we answer them

**"Conversational commerce platforms already automate ordering."**
They prevent free text by putting a form in front of it. We parse the free text instead. The customer who can't be put behind a form is precisely the one who reorders every week, so that's the customer we built for.

**"Why not just call a hosted model API?"**
We demonstrate it offline with the network disabled, on camera. Beyond that, order messages are personal data under UU PDP, and per-call pricing doesn't survive Rupiah-margin trade at sixty messages a night.

**"Is the small model actually good enough?"**
We measured it: 0.8014 F1 at 11.7M parameters against 0.8088 at 124M on IndoNLU NERP. That's 10.6× the parameters for 0.004 F1 (p7.2).

**"This is one narrow task."**
The task is the demonstration. The contribution is a method for offline Indonesian extraction, and we measured it across domains rather than asserting that it transfers.

**"Distillation is engineering, not innovation."**
The contribution is the task formulation: bounded-output tagging instead of generation, and separating parsing confidence from resolution confidence. That's supported by a measured architecture comparison (p7.2), a diagnosed quantization result (p7.3), and a comparison against the permitted alternative (p7.4).

---

## 14. Scoring reference

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

## 15. Open decisions

| # | Decision | When |
|---|---|---|
| 1 | **Team size.** The parallel plan assumes 4–5 people. At 3, we cut the elicitation target to ~60 and drop the second benchmark domain. | Immediately |
| 2 | Product name | Before video production |
| 3 | Catalog source for the demo, partner distributor or constructed | Affects how credible the demo looks, so early |
| 4 | Student depth and hidden width for distillation | Start of week 2 |
| 5 | Whether to add a second benchmark domain | 17 Aug, against the schedule |

---

## 16. References

- IndoNLU benchmark — huggingface.co/datasets/indonlp/indonlu · aclanthology.org/2020.aacl-main.85.pdf
- Tokopedia product listings (29,519 rows) — Kaggle
- NusaX — huggingface.co/datasets/indonlp/NusaX-senti
- Sahabat-AI — huggingface.co/GoToCompany/llama3-8b-cpt-sahabatai-v1-instruct
- IndoBERT — huggingface.co/indobenchmark
- Hsieh et al., Distilling Step-by-Step, 2023 — arxiv.org/abs/2305.02301
- UU PDP No. 27/2022 — in force 17 Oct 2024

---

## Appendix A — Elicitation instrument

Deploy as a Google Form. Five prompts, one per phenomenon class, target around 30 respondents.

### A.1 Introduction

> **Bantu kami mengumpulkan contoh pesan pemesanan**
>
> Kami sedang membangun sistem yang membaca pesan pemesanan barang dan mengubahnya jadi daftar pesanan yang rapi. Untuk mengujinya, kami butuh contoh pesan yang ditulis **seperti orang beneran menulis**, bukan yang rapi dan formal.
>
> Bayangkan kamu pemilik warung yang memesan barang ke supplier langganan, supplier yang sudah kamu pakai bertahun-tahun.
>
> **Tulis cepat. Jangan dirapikan, jangan dikoreksi.** Singkatan, typo, dan bahasa daerah justru yang kami butuhkan. Sekitar 5 menit.
>
> Jawaban tidak dikaitkan dengan identitas apa pun dan hanya dipakai untuk menguji akurasi sistem.

### A.2 Prompts

| # | Prompt | Targets |
|---|---|---|
| 1 | Kamu butuh 1 dus mie instan dan 5 kg gula. Tulis pesanmu ke supplier. | `explicit` |
| 2 | Kamu mau pesan **persis seperti pesananmu minggu lalu**. Tulis pesanmu. | `anaphoric` |
| 3 | Kamu butuh minyak goreng dan gula, jumlahnya seperti biasa, tidak perlu kamu sebut. | `quantity_elided` |
| 4 | Kamu butuh 2 dus mie instan, tapi yang **ukuran besar**. | `underspecified_attr` |
| 5 | Tambahkan satu kalimat penutup untuk mengakhiri pesananmu. | `discourse_marker` |

Each prompt is a required free-text field with no character minimum.

### A.3 Optional metadata

- Daerah asal (province only)
- Pernah punya usaha warung/toko? (Ya / Tidak / Keluarga saya punya)

No names, no phone numbers, no contact details. Nothing we collect requires consent management under UU PDP, which is worth a line in the governance section.

### A.4 Labelling protocol

- Character-offset spans in the same `{text, spans}` format as the training data
- Two labellers per message, disagreements resolved by discussion
- Inter-annotator agreement reported
- Prompt number kept as the phenomenon label, which is what enables the per-class breakdown in p7.1
- Responses that don't exhibit the targeted phenomenon get relabelled rather than discarded, since a respondent ignoring the prompt is itself realistic input

### A.5 Distribution

One campus or community group should be enough for 30 respondents. No business recruitment, no interviews, no consent paperwork. If the first posting underdelivers, we add a second channel before the 4 August decision point.
