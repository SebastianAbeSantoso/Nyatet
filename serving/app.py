"""
Nyatet serving API.

One synchronous endpoint. No background jobs, no queues, no database,
no auth — per the MVP scope cap.

Pipeline: tagger (the only model) -> normalizer -> resolver, each stage
in its own module, the last two with no model dependency.
"""

import os
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from inference.resolver import CatalogResolver, group_spans
from inference.tagger import SpanTagger

MODEL_DIR = os.environ.get("MODEL_DIR", str(Path(__file__).parent / "models" / "tagger"))
CATALOG_PATH = os.environ.get("CATALOG_PATH", str(Path(__file__).parent / "data" / "catalog.csv"))
NUM_THREADS = int(os.environ.get("NUM_THREADS", "1"))

app = FastAPI(title="Nyatet", version="1.0.0",
              description="Offline order parsing for informal Indonesian text")

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

_tagger: SpanTagger | None = None
_resolver: CatalogResolver | None = None

# Standing orders are weekly day->quantity schedules from resellers
# (`selasa - jumat 34, sabtu 38`). They are detected deterministically
# and flagged, not parsed: a recurring schedule is not one order.
_DAYS = r"(senin|selasa|rabu|kamis|jumat|sabtu|minggu)"
_STANDING = re.compile(rf"{_DAYS}.*\d.*{_DAYS}.*\d", re.I | re.S)


@app.on_event("startup")
def load():
    global _tagger, _resolver
    try:
        _tagger = SpanTagger(MODEL_DIR, num_threads=NUM_THREADS)
        _resolver = CatalogResolver(CATALOG_PATH)
        print(f"loaded model from {MODEL_DIR}: {len(_tagger.labels)} labels")
        print(f"loaded catalog from {CATALOG_PATH}: {len(_resolver.items)} items, "
              f"primary product '{_resolver.primary_product}'")
    except (FileNotFoundError, ValueError) as e:
        print(f"WARNING: {e}")


class ParseRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)


class Span(BaseModel):
    type: str
    text: str
    confidence: float | None


class Line(BaseModel):
    item_span: str | None
    variant_span: str | None
    matched_name: str | None
    sku: str | None
    resolution_confidence: float
    candidates: list[str]
    quantity: float | None
    per_container: float | None
    unit: str | None
    total_pieces: float | None
    item_inferred: bool
    needs_clarification: str | None


class Flag(BaseModel):
    kind: str
    span_text: str | None
    note: str


class ParseResponse(BaseModel):
    message: str
    spans: list[Span]
    order_lines: list[Line]
    flags: list[Flag]


CLARIFY_NOTE = {
    "variant": "Varian tidak disebutkan — perlu dikonfirmasi (mentah / goreng / frozen).",
    "ambiguous_option": "Beberapa pilihan cocok — perlu dikonfirmasi.",
    "quantity": "Jumlah tidak dapat dipastikan dari pesan ini.",
    "unknown_product": "Produk tidak dikenali dalam katalog.",
}


@app.post("/parse", response_model=ParseResponse)
def parse(request: ParseRequest) -> ParseResponse:
    if _tagger is None or _resolver is None:
        raise HTTPException(503, f"model or catalog not loaded (MODEL_DIR={MODEL_DIR})")

    flags: list[Flag] = []

    if _STANDING.search(request.message):
        flags.append(Flag(kind="standing_order", span_text=None,
                          note="Terdeteksi jadwal pesanan berulang. Tidak dapat "
                               "diproses sebagai satu pesanan — ditandai untuk ditinjau."))
        return ParseResponse(message=request.message, spans=[], order_lines=[], flags=flags)

    tagged = _tagger.tag(request.message)

    for s in tagged:
        if s["type"] == "ANAPHORIC":
            flags.append(Flag(kind="anaphoric", span_text=s["text"],
                              note="Merujuk ke pesanan sebelumnya. Tidak dapat "
                                   "diselesaikan tanpa riwayat — ditandai untuk ditinjau."))

    lines = [_resolver.resolve(l) for l in group_spans(tagged)]

    for l in lines:
        if l.needs_clarification:
            flags.append(Flag(kind=l.needs_clarification, span_text=l.item_span,
                              note=CLARIFY_NOTE.get(l.needs_clarification, "Perlu dikonfirmasi.")))

    return ParseResponse(
        message=request.message,
        spans=[Span(type=s["type"], text=s["text"], confidence=s.get("confidence")) for s in tagged],
        order_lines=[Line(**{k: getattr(l, k) for k in Line.model_fields}) for l in lines],
        flags=flags,
    )


@app.get("/health")
def health():
    return {
        "status": "ok" if (_tagger and _resolver) else "not_ready",
        "model_loaded": _tagger is not None,
        "catalog_loaded": _resolver is not None,
        "labels": _tagger.labels if _tagger else [],
        "catalog_items": len(_resolver.items) if _resolver else 0,
    }
