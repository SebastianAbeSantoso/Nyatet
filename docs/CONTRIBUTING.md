# Contributing

Conventions for working in this repository. Read before your first commit, commit history is not rewritten after it has been pushed.

---

## Contents

1. [Commit convention](#1-commit-convention)
2. [Where things go](#2-where-things-go)
3. [Recording results](#3-recording-results)
4. [Before pushing](#4-before-pushing)
5. [Notebooks](#5-notebooks)
6. [Dependencies](#6-dependencies)

---

## 1. Commit convention

Every commit follows [Conventional Commits](https://www.conventionalcommits.org/): `type: description`

| Type | Use for |
|---|---|
| `feat:` | New feature or capability |
| `fix:` | Bug or error correction |
| `refactor:` | Structural change with no behaviour change |
| `docs:` | Documentation, README, proposal |
| `build:` | Build system, compilation, bundling, CI/CD, or packaging changes |
| `chore:` | Dependencies, config, tooling, and repository maintenance not affecting the build system |



**Write the description so it means something to someone without context.**

Good:

```
feat: add NERP benchmark pipeline (B1, indobert-lite-p2)
fix: enable MatMulConstBOnly for ALBERT shared-weight quantization
refactor: split resolver confidence from tagging confidence
docs: record B1/B2 comparison in RESULTS.md
build: package ONNX model in Docker image
chore: update .gitignore for ONNX export cache
```

Not acceptable:

```
update
fix bug
wip
final
asdf
```

Include measured numbers when a commit produces them. The history becomes part of the development-process record.

```
feat: export int8 tagger — NERP F1 0.8014, 11.53MB, 31ms
```

Commit in steps, not lumps. Four commits that show work unfolding are better than one commit that says "add everything". If a change touches unrelated things, split it.

---

## 2. Where things go

| Path | Contents | Never |
|---|---|---|
| `training/order/` | Target task: generator, fine-tune, export | Runs in the container |
| `training/benchmark/` | Benchmark validation runs | Runs in the container |
| `serving/` | The MVP: FastAPI + ONNX inference | Imports torch |
| `serving/inference/tagger.py` | The only file that touches a model | — |
| `serving/inference/normalizer.py` | Deterministic quantity handling | Uses a model |
| `serving/inference/resolver.py` | Deterministic catalog matching | Uses a model |
| `frontend/` | One page, no build step | Gains a second flow |
| `tests/` | pytest for deterministic components | Requires a trained model |
| `docs/` | PRD, RESULTS, rulebook digest | Holds duplicate numbers |

Two rules hold the architecture together:

Training never runs in the served container. `training/` produces an ONNX file; `serving/` loads it. Nothing else crosses that line.

Only `tagger.py` touches a model. Normalizer and resolver stay plain Python. This is what makes the AI/backend separation readable from the directory tree.

---

## 3. Recording results

All measured numbers go in `docs/RESULTS.md`. Nowhere else.

READMEs may carry the two or three headline figures and link out. No other file restates a measurement.

Each run gets an ID (`B1`, `B2`, `O1`…) registered in the RESULTS index. Cite by ID rather than repeating numbers:

When you add a run:

1. Add a row to the run index
2. Add a section with accuracy, size, latency, per-class
3. Add findings or caveats if it produced any
4. Update the "last updated" line

Findings are numbered (`F1`…) and caveats are numbered (`C1`…) so they can be cited from the PRD without restating them.

---

## 4. Before pushing

- [ ] Tests pass — `python -m pytest tests/ -v`
- [ ] New measurements recorded in `docs/RESULTS.md`
- [ ] Commit message follows the convention above
- [ ] No institutional, university, or team-member identification anywhere, code, comments, notebook headers, commit messages, filenames, or model cards

---

## 5. Notebooks

Commit notebooks with outputs intact. Normally this is bad practice, here the rendered training curves, classification reports and measured numbers are the evidence. Stripping outputs discards proof of work.

First cell is a markdown header:

```markdown
# B1 — indobert-lite-base-p2 · results in docs/RESULTS.md
```

Use underscores in filenames, not hyphens: `benchmark_lite.ipynb`.

---

## 6. Dependencies

| File | Contains | Used by |
|---|---|---|
| `training/requirements.txt` | torch, transformers, datasets, optimum, onnx | Both training domains |
| `serving/requirements.txt` | onnxruntime, transformers (tokenizer only), FastAPI | The served container |

`serving/requirements.txt` must never gain torch, datasets, or optimum. The container loads an already-exported ONNX file, it does not train, export, or quantize.

Environment note. Kaggle and Colab preinstall `diffusers`, which pulls a `huggingface-hub` version incompatible with optimum's ONNX export. Remove it before installing:

```bash
pip uninstall -y diffusers
pip install -r training/requirements.txt
```

`training/environment-freeze.txt` is the full 881-package `pip freeze` from the environment that produced B1 and B2. It is a record for reproducing that exact environment, not a file to install from.

Model artifacts (`.onnx`) are committed, not gitignored. At ~12 MB the model ships inside the repository, so `docker compose up` needs no download step.