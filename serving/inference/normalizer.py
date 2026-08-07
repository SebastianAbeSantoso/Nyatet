import re

_UNIT_ALIASES = {
    "biji":   ["biji", "bj", "bji", "bijinya", "bijilh"],
    "buting": ["buting", "butinglh", "butingnya", "btg"],
    "pcs":    ["pcs", "pc", "pieces"],
    "kotak":  ["kotak", "kotaknya", "ktk", "box"],
    "loyang": ["loyang", "loyangnya"],
    "toples": ["toples"],
    "bungkus": ["bungkus", "bks"],
}
_UNIT_LOOKUP = {v: k for k, vs in _UNIT_ALIASES.items() for v in vs}

PIECE_UNITS = {"biji", "buting", "pcs"}
CONTAINER_UNITS = {"kotak", "loyang", "toples", "bungkus"}

_WORD_NUMBERS = {
    "satu": 1, "dua": 2, "tiga": 3, "empat": 4, "lima": 5, "enam": 6,
    "tujuh": 7, "delapan": 8, "sembilan": 9, "sepuluh": 10,
    "selusin": 12, "lusin": 12, "duapuluh": 20, "dua puluh": 20,
    "seratus": 100,
}

_FRACTIONS = {
    "setengah": 0.5, "1/2": 0.5, "½": 0.5,
    "seperempat": 0.25, "1/4": 0.25, "¼": 0.25,
}

_DIGIT_FRACTION = re.compile(r"^(\d+)\s*/\s*(\d+)$")
_TRAILING_PARTICLE = re.compile(r"(lh|lah|nya|ai|ja|aja|gin|kh|ya)$", re.I)


def strip_particle(text: str) -> str:
    t = text.strip().lower()
    if t in _UNIT_LOOKUP:            
        return t
    stripped = _TRAILING_PARTICLE.sub("", t).strip()
    return stripped if stripped in _UNIT_LOOKUP else t


def normalize_unit(raw: str) -> str | None:
    return _UNIT_LOOKUP.get(strip_particle(raw))


def normalize_quantity(raw: str) -> float | None:
    t = raw.strip().lower()
    t = _TRAILING_PARTICLE.sub("", t).strip()

    if t in _FRACTIONS:
        return _FRACTIONS[t]

    m = _DIGIT_FRACTION.match(t)
    if m and int(m.group(2)) != 0:
        return int(m.group(1)) / int(m.group(2))

    digits = re.sub(r"[^\d,.]", "", t).replace(",", ".")
    if digits:
        try:
            return float(digits)
        except ValueError:
            pass

    if t in _WORD_NUMBERS:
        return float(_WORD_NUMBERS[t])

    if t.startswith("se") and len(t) > 3:
        rest = t[2:]
        if rest in _UNIT_LOOKUP or normalize_unit(rest):
            return 1.0

    return None


def normalize_variant(raw: str) -> str | None:
    t = raw.strip().lower()
    t = re.sub(r"^(yg|yang)\s+", "", t)
    t = _TRAILING_PARTICLE.sub("", t).strip()
    t = re.sub(r"^(di|be|ba|sdh|sudah)\s*", "", t).strip()

    if "mentah" in t:
        return "mentah"
    if "frozen" in t or "freezer" in t:
        return "frozen"
    if "goreng" in t or "masak" in t:
        return "goreng"
    return None

def resolve_quantities(qty_spans: list[str], unit_span: str | None) -> dict:
    unit = normalize_unit(unit_span) if unit_span else None
    values = [normalize_quantity(q) for q in qty_spans]

    if not values:
        return {"count": None, "per_container": None, "unit": unit, "ambiguous": True}

    if any(v is None for v in values):
        return {"count": None, "per_container": None, "unit": unit, "ambiguous": True}

    if len(values) == 1:
        return {"count": values[0], "per_container": None, "unit": unit, "ambiguous": False}

    if len(values) == 2 and unit in CONTAINER_UNITS:
        return {"count": values[1], "per_container": values[0],
                "unit": unit, "ambiguous": False}

    return {"count": values[0], "per_container": None, "unit": unit, "ambiguous": True}


def total_pieces(count: float | None, per_container: float | None,
                unit: str | None, catalog_pack_size: float | None = None) -> float | None:
    if count is None:
        return None
    if unit in PIECE_UNITS or unit is None:
        return count
    fill = per_container if per_container is not None else catalog_pack_size
    return count * fill if fill is not None else None
