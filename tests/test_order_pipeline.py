import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from inference.normalizer import (
    normalize_quantity, normalize_unit, normalize_variant,
    resolve_quantities, total_pieces,
)
from inference.resolver import CatalogResolver, group_spans

CATALOG = Path(__file__).resolve().parents[1] / "data" / "catalog.csv"


def spans(*pairs):
    return [{"type": t, "text": x} for t, x in pairs]


@pytest.fixture(scope="module")
def resolver():
    return CatalogResolver(CATALOG)

@pytest.mark.parametrize("raw,expected", [
    ("biji", "biji"), ("bj", "biji"), ("bji", "biji"),
    ("buting", "buting"), ("butinglh", "buting"),   
    ("pcs", "pcs"), ("pc", "pcs"),
    ("kotak", "kotak"), ("loyang", "loyang"),
    ("dus", None), ("kg", None),                  
])
def test_unit_aliases(raw, expected):
    assert normalize_unit(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("20", 20.0), ("1", 1.0), ("320", 320.0),
    ("dua", 2.0), ("lima", 5.0),
    ("setengah", 0.5), ("1/4", 0.25), ("¼", 0.25),
    ("sebuting", 1.0),                              
    ("banyak", None),                              
])
def test_quantity_forms(raw, expected):
    assert normalize_quantity(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("mentah", "mentah"), ("mentahnya", "mentah"), ("yg mentah", "mentah"),
    ("goreng", "goreng"), ("digoreng", "goreng"), ("di goreng", "goreng"),
    ("begoreng", "goreng"), ("sdh masak", "goreng"),
    ("frozen", "frozen"), ("Frozen", "frozen"),
    ("xyz", None),
])
def test_variant_surface_forms(raw, expected):
    assert normalize_variant(raw) == expected


def test_nested_quantity_adjacency_rule():
    q = resolve_quantities(["15", "1"], "kotak")
    assert q["count"] == 1 and q["per_container"] == 15
    assert not q["ambiguous"]


def test_empty_quantity_is_ambiguous_not_a_crash():
    q = resolve_quantities([], "biji")
    assert q["count"] is None and q["ambiguous"]


def test_piece_units_take_no_multiplier():
    assert total_pieces(20, None, "biji", catalog_pack_size=1) == 20


def test_inline_pack_size_overrides_catalog():
    assert total_pieces(1, 15, "kotak", catalog_pack_size=10) == 15

def test_two_products_two_lines():
    lines = group_spans(spans(
        ("ITEM", "risol"), ("VARIANT", "goreng"), ("QTY", "10"), ("UNIT", "biji"),
        ("ITEM", "risol"), ("VARIANT", "mentah"), ("QTY", "10"), ("UNIT", "biji")))
    assert len(lines) == 2
    assert [l.variant_span for l in lines] == ["goreng", "mentah"]


def test_detached_variant_splits_into_two_lines():
    lines = group_spans(spans(
        ("QTY", "25"), ("VARIANT", "di goreng"),
        ("QTY", "25"), ("VARIANT", "yg mentah")))
    assert len(lines) == 2
    assert all(l.qty_spans == ["25"] for l in lines)


def test_nested_quantities_stay_on_one_line():
    lines = group_spans(spans(
        ("ITEM", "Risol"), ("QTY", "15"), ("QTY", "1"), ("UNIT", "kotak")))
    assert len(lines) == 1
    assert lines[0].qty_spans == ["15", "1"]


def test_anaphoric_spans_are_not_order_lines():
    lines = group_spans(spans(("ANAPHORIC", "ky biasa"), ("QTY", "45"), ("UNIT", "buting")))
    assert len(lines) == 1
    assert lines[0].qty_spans == ["45"]

def test_full_specification_resolves_cleanly(resolver):
    line = resolver.resolve(group_spans(spans(
        ("ITEM", "risol"), ("VARIANT", "mentah"), ("QTY", "20"), ("UNIT", "biji")))[0])
    assert line.sku == "RSL-MTH"
    assert line.resolution_confidence == 1.0
    assert line.needs_clarification is None
    assert line.total_pieces == 20


def test_missing_variant_collapses_resolution_confidence(resolver):
    line = resolver.resolve(group_spans(spans(
        ("ITEM", "resoles"), ("QTY", "50"), ("UNIT", "biji")))[0])
    assert line.needs_clarification == "variant"
    assert line.resolution_confidence < 0.5
    assert len(line.candidates) == 3
    assert line.total_pieces == 50


def test_implicit_item_defaults_to_primary_product(resolver):
    line = resolver.resolve(group_spans(spans(
        ("VARIANT", "Yg mentah"), ("QTY", "10")))[0])
    assert line.sku == "RSL-MTH"
    assert line.item_inferred
    assert line.resolution_confidence < 1.0     


def test_product_spelling_variants_all_match(resolver):
    for spelling in ("risol", "risoles", "resoles", "resol", "risolnya"):
        line = resolver.resolve(group_spans(spans(
            ("ITEM", spelling), ("VARIANT", "mentah"), ("QTY", "5"), ("UNIT", "biji")))[0])
        assert line.sku == "RSL-MTH", spelling


def test_inline_pack_size_selects_the_right_container_sku(resolver):
    line = resolver.resolve(group_spans(spans(
        ("ITEM", "Risol"), ("QTY", "15"), ("QTY", "1"), ("UNIT", "kotak")))[0])
    assert line.sku == "RSL-KTK-15"
    assert line.total_pieces == 15


def test_unspecified_size_flags_rather_than_guesses(resolver):
    line = resolver.resolve(group_spans(spans(
        ("ITEM", "bronies"), ("QTY", "3"), ("UNIT", "loyang")))[0])
    assert line.sku is None
    assert line.needs_clarification == "ambiguous_option"
    assert len(line.candidates) == 2


def test_unknown_product_is_flagged(resolver):
    line = resolver.resolve(group_spans(spans(
        ("ITEM", "sepatu"), ("QTY", "2"), ("UNIT", "biji")))[0])
    assert line.needs_clarification == "unknown_product"


def test_fused_particle_unit_still_resolves(resolver):
    line = resolver.resolve(group_spans(spans(("QTY", "200"), ("UNIT", "butinglh")))[0])
    assert line.unit == "buting"
    assert line.total_pieces == 200
