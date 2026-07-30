import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "serving"))

from inference.spans import collapse_bio


def test_multi_token_span():
    spans = collapse_bio(
        ["indomi", "goreng", "1", "dus"],
        ["B-ITEM", "I-ITEM", "B-QTY", "B-UNIT"],
    )
    assert spans[0]["type"] == "ITEM"
    assert spans[0]["text"] == "indomi goreng"
    assert spans[0]["start_token"] == 0 and spans[0]["end_token"] == 1
    assert [s["type"] for s in spans] == ["ITEM", "QTY", "UNIT"]


def test_adjacent_same_type_does_not_merge():
    """Two B- tags of the same type are two entities, not one."""
    spans = collapse_bio(["gula", "rinso"], ["B-ITEM", "B-ITEM"])
    assert len(spans) == 2
    assert [s["text"] for s in spans] == ["gula", "rinso"]


def test_wordpiece_tokens_rejoined():
    spans = collapse_bio(["am", "##os", "kend", "##a"], ["B-PPL", "I-PPL", "I-PPL", "I-PPL"])
    assert spans[0]["text"] == "amos kenda"


def test_o_tags_produce_nothing():
    assert collapse_bio(["itu", "aja"], ["O", "O"]) == []


def test_confidence_is_span_mean():
    spans = collapse_bio(["a", "b"], ["B-X", "I-X"], [1.0, 0.5])
    assert spans[0]["confidence"] == 0.75


def test_confidence_omitted_when_no_scores():
    assert collapse_bio(["a"], ["B-X"])[0]["confidence"] is None


def test_schema_agnostic():
    """Same function, unrelated label inventories — this is the architecture claim."""
    nerp = collapse_bio(["manado"], ["B-PLC"])
    order = collapse_bio(["gula"], ["B-ITEM"])
    assert nerp[0]["type"] == "PLC"
    assert order[0]["type"] == "ITEM"


def test_tags_without_bio_prefix():
    """Tolerate schemas that don't use B-/I- prefixes."""
    spans = collapse_bio(["jakarta"], ["PLC"])
    assert spans[0]["type"] == "PLC"
