import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from inference.tagger import SpanTagger

MODEL_DIR = os.environ.get("MODEL_DIR", str(Path(__file__).parent / "models" / "tagger"))
NUM_THREADS = int(os.environ.get("NUM_THREADS", "1"))

app = FastAPI(
    title="Nyatet",
    description="Offline span tagging for informal Indonesian text",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   
    allow_methods=["*"],
    allow_headers=["*"],
)

_tagger: SpanTagger | None = None


@app.on_event("startup")
def load_model():
    """Loaded once at process start — not per request, not in a background job."""
    global _tagger
    try:
        _tagger = SpanTagger(MODEL_DIR, num_threads=NUM_THREADS)
        print(f"loaded model from {MODEL_DIR}: {len(_tagger.labels)} labels")
    except FileNotFoundError as e:
        print(f"WARNING: {e}")
        _tagger = None


class ParseRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)


class Span(BaseModel):
    type: str
    text: str
    start_token: int
    end_token: int
    confidence: float | None


class ParseResponse(BaseModel):
    message: str
    spans: list[Span]


@app.post("/parse", response_model=ParseResponse)
def parse(request: ParseRequest) -> ParseResponse:
    if _tagger is None:
        raise HTTPException(status_code=503, detail=f"No model loaded from {MODEL_DIR}")
    return ParseResponse(message=request.message, spans=_tagger.tag(request.message))


@app.get("/health")
def health():
    return {
        "status": "ok" if _tagger else "no_model",
        "model_loaded": _tagger is not None,
        "model_dir": MODEL_DIR,
        "labels": _tagger.labels if _tagger else [],
    }
