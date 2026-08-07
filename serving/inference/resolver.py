import csv
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

from .normalizer import (
    CONTAINER_UNITS,
    normalize_quantity,
    normalize_unit,
    normalize_variant,
    resolve_quantities,
    total_pieces,
)


@dataclass
class CatalogItem:
    sku_id: str
    name: str
    product: str
    variant: str | None
    unit: str
    pack_size: float
    is_primary: bool


@dataclass
class OrderLine:
    item_span: str | None
    variant_span: str | None
    qty_spans: list[str]
    unit_span: str | None

    sku: str | None = None
    matched_name: str | None = None
    resolution_confidence: float = 0.0
    candidates: list[str] = field(default_factory=list)

    quantity: float | None = None
    per_container: float | None = None
    unit: str | None = None
    total_pieces: float | None = None

    item_inferred: bool = False       
    needs_clarification: str | None = None


class CatalogResolver:
    def __init__(self, catalog_path: str | Path):
        self.items: list[CatalogItem] = []
        with open(catalog_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self.items.append(CatalogItem(
                    sku_id=row["sku_id"],
                    name=row["name"],
                    product=row["product"].strip().lower(),
                    variant=(row["variant"].strip().lower() or None),
                    unit=row["unit"].strip().lower(),
                    pack_size=float(row["pack_size"]),
                    is_primary=row["is_primary"].strip().lower() == "true",
                ))
        if not self.items:
            raise ValueError(f"empty catalog: {catalog_path}")

    @property
    def primary_product(self) -> str:
        for i in self.items:
            if i.is_primary:
                return i.product
        return self.items[0].product

    def _product_similarity(self, span: str, product: str) -> float:
        a, b = span.lower(), product.lower()
        if a.startswith(b) or b.startswith(a):
            return 1.0                      
        return SequenceMatcher(None, a, b).ratio()

    def resolve(self, line: OrderLine) -> OrderLine:
        if line.item_span:
            scored = sorted(
                ((i, self._product_similarity(line.item_span, i.product)) for i in self.items),
                key=lambda p: p[1], reverse=True)
            best_score = scored[0][1]
            if best_score < 0.5:
                line.needs_clarification = "unknown_product"
                line.candidates = [i.name for i, _ in scored[:3]]
                return self._apply_quantities(line, None)
            product = scored[0][0].product
        else:
            product = self.primary_product
            line.item_inferred = True

        pool = [i for i in self.items if i.product == product]

        unit = normalize_unit(line.unit_span) if line.unit_span else None
        if unit:
            by_unit = [i for i in pool if i.unit == unit]
            if by_unit:
                pool = by_unit

        variant = normalize_variant(line.variant_span) if line.variant_span else None
        variant_bearing = [i for i in pool if i.variant]

        if variant:
            exact = [i for i in pool if i.variant == variant]
            if exact:
                pool = exact
        elif len(variant_bearing) > 1:
            line.candidates = [i.name for i in variant_bearing]
            line.resolution_confidence = round(1.0 / len(variant_bearing), 3)
            line.needs_clarification = "variant"
            return self._apply_quantities(line, None)

        if len(pool) == 1:
            match = pool[0]
            line.sku = match.sku_id
            line.matched_name = match.name
            line.resolution_confidence = 1.0 if not line.item_inferred else 0.85
            return self._apply_quantities(line, match)

        q_pre = resolve_quantities(line.qty_spans, line.unit_span)
        if q_pre["per_container"] is not None:
            sized = [i for i in pool if i.pack_size == q_pre["per_container"]]
            if len(sized) == 1:
                match = sized[0]
                line.sku = match.sku_id
                line.matched_name = match.name
                line.resolution_confidence = 1.0 if not line.item_inferred else 0.85
                return self._apply_quantities(line, match)

        line.candidates = [i.name for i in pool]
        line.resolution_confidence = round(1.0 / len(pool), 3)
        line.needs_clarification = "ambiguous_option"
        return self._apply_quantities(line, None)

    def _apply_quantities(self, line: OrderLine, match: CatalogItem | None) -> OrderLine:
        q = resolve_quantities(line.qty_spans, line.unit_span)
        line.quantity = q["count"]
        line.per_container = q["per_container"]
        line.unit = q["unit"]

        if q["ambiguous"] and line.needs_clarification is None:
            line.needs_clarification = "quantity"

        if line.quantity is None and line.needs_clarification is None:
            line.needs_clarification = "quantity"

        line.total_pieces = total_pieces(
            line.quantity, line.per_container, line.unit,
            catalog_pack_size=match.pack_size if match else None)
        return line


def group_spans(spans: list[dict]) -> list[OrderLine]:
    lines: list[OrderLine] = []
    cur: OrderLine | None = None

    def new_line() -> OrderLine:
        return OrderLine(item_span=None, variant_span=None, qty_spans=[], unit_span=None)

    for s in spans:
        t, text = s["type"], s["text"]

        if t == "ANAPHORIC":
            continue                      
        if t == "ITEM":
            if cur and (cur.item_span or cur.qty_spans):
                lines.append(cur)
            cur = new_line()
            cur.item_span = text
            continue

        if cur is None:
            cur = new_line()

        if t == "VARIANT":
            if cur.variant_span is not None:
                lines.append(cur)
                cur = new_line()
            cur.variant_span = text
        elif t == "QTY":
            complete = cur.qty_spans and (cur.unit_span or cur.variant_span)
            nested = cur.qty_spans and not cur.unit_span and not cur.variant_span
            if complete and not nested:
                lines.append(cur)
                cur = new_line()
            cur.qty_spans.append(text)
        elif t == "UNIT":
            cur.unit_span = text

    if cur and (cur.item_span or cur.qty_spans or cur.variant_span):
        lines.append(cur)
    return lines
