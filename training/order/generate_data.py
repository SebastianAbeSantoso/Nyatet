import csv
import json
import random
import re

ITEMS = {
    "risol":      10,
    "risoles":     3,
    "resoles":     1,
    "resol":       1,
    "risolnya":    1,   
}
CAKE_ITEMS = {"bronies": 1, "broniesnya": 1, "bolu": 1, "kue": 1}

UNITS = {"biji": 7, "buting": 6, "kotak": 5, "pcs": 2, "bj": 2, "bji": 1, "pc": 1}
CAKE_UNITS = {"loyang": 2, "biji": 1}

VARIANTS = {
    "mentah": 6, "digoreng": 2, "goreng": 1, "mentahnya": 1,
    "di goreng": 1, "begoreng": 2, "sdh begoreng": 1, "frozen": 2,
    "yg mentah": 1, "sdh masak": 1, "yg masak": 1,
}

ANAPHORIC = ["ky biasa", "kya biasa", "kaya biasa", "ky kmrn", "kaya kmrn",
            "spt biasa", "yg biasa", "kya smlm"]

OPENERS = ["", "", "ka ", "bu ", "kak ", "Assalamualaikum ", "pagi bu ",
        "Ka, ", "Om ", "mba ", "Pesan ", "Mau ", "Bisa pesan ",
        "Bisakh pesan ", "Kawakah pesan ", "ulun pesan ", "lun pesan ",
        "Besok pesan ", "Esok tinggali ", "Pagi ini tinggali "]
TAILS = ["", "", " ya", " lah", " lh", " ka", " bu", " aja", " gin bu",
        " ada kh", " adakh", " bisa lah", " bisalh", " kw lah",
        " ya bu", " aja kak", " lagi lah"]
TIMES = ["", "", " jam 7 pagi diambil", " jam 5 sore", " jam 08.15 diambil",
        " bsk di ambil jm 10", " diambil jam 11", " untuk besok",
        " buat besok", " hari ini", " jam 4 sore ulun ambil",
        " sblm jam 7 pagi", " jam brp ready"]

PARTICLES = ["lh", "lah", "nya", "ai"]   


def _pick(weighted: dict) -> str:
    return random.choices(list(weighted), weights=list(weighted.values()))[0]


def _maybe_fuse(word: str, p: float = 0.07) -> str:
    return word + random.choice(PARTICLES) if random.random() < p else word


class Builder:
    def __init__(self):
        self.parts, self.spans, self.pos = [], [], 0

    def add(self, text: str, span_type: str | None = None):
        if span_type and text.strip():
            self.spans.append({
                "type": span_type,
                "text": text,
                "start": self.pos,
                "end": self.pos + len(text),
            })
        self.parts.append(text)
        self.pos += len(text)

    def build(self):
        return "".join(self.parts), self.spans

def p_qty_unit(b, cake=False):
    b.add(str(_qty()), "QTY")
    b.add(" ")
    b.add(_maybe_fuse(_pick(CAKE_UNITS if cake else UNITS)), "UNIT")


def p_item_qty_unit(b, cake=False):
    b.add(_pick(CAKE_ITEMS if cake else ITEMS), "ITEM")
    b.add(" ")
    b.add(str(_qty()), "QTY")
    b.add(" " if random.random() > 0.15 else "")   
    b.add(_maybe_fuse(_pick(CAKE_UNITS if cake else UNITS)), "UNIT")

def p_item_variant_qty(b, cake=False):
    b.add(_pick(CAKE_ITEMS if cake else ITEMS), "ITEM")
    b.add(" ")
    b.add(_maybe_fuse(_pick(VARIANTS)), "VARIANT")
    b.add(" ")
    b.add(str(_qty()), "QTY")
    if random.random() > 0.3:
        b.add(" ")
        b.add(_pick(CAKE_UNITS if cake else UNITS), "UNIT")

def p_variant_qty(b, cake=False):
    b.add(_maybe_fuse(_pick(VARIANTS)), "VARIANT")
    b.add(" ")
    b.add(str(_qty()), "QTY")
    if random.random() > 0.6:
        b.add(" ")
        b.add(_pick(UNITS), "UNIT")

def p_qty_variant(b, cake=False):
    b.add(str(_qty()), "QTY")
    b.add(" ")
    b.add(_maybe_fuse(_pick(VARIANTS)), "VARIANT")

def p_reversed(b, cake=False):
    b.add(str(_qty()), "QTY")
    b.add(" ")
    b.add(_maybe_fuse(_pick(UNITS)), "UNIT")
    b.add(" ")
    b.add(_pick(ITEMS), "ITEM")

def p_nested_box(b, cake=False):
    if random.random() < 0.5:
        if random.random() > 0.4:
            b.add(_pick(ITEMS), "ITEM")
            b.add(" ")
        b.add("isi ")
        b.add(str(random.choice([10, 12, 15, 20])), "QTY")
        b.add(" ")
        b.add(str(random.choice([1, 2, 3])), "QTY")
        b.add(" ")
        b.add("kotak", "UNIT")
    else:
        b.add(str(random.choice([2, 3])), "QTY")
        b.add(" ")
        b.add("kotak", "UNIT")
        b.add(" isi ")
        b.add(str(random.choice([10, 12, 15])), "QTY")

def p_two_variants(b, cake=False):
    for i in range(2):
        if i:
            b.add(random.choice([", ", " ", ". "]))
        if random.random() > 0.35:
            b.add(_pick(ITEMS), "ITEM")
            b.add(" ")
        b.add(_pick(VARIANTS), "VARIANT")
        b.add(" ")
        b.add(str(_qty()), "QTY")
        if random.random() > 0.35:
            b.add(" ")
            b.add(_pick(UNITS), "UNIT")

def p_split_variant(b, cake=False):
    n = random.choice([10, 20, 25, 30])
    b.add(str(n), "QTY")
    b.add(random.choice([" nya ", " "]))
    b.add(_pick(VARIANTS), "VARIANT")
    b.add(" ")
    b.add(str(n), "QTY")
    b.add(" ")
    b.add(_pick(VARIANTS), "VARIANT")

def p_anaphoric(b, cake=False):
    if random.random() > 0.5:
        b.add(_pick(ITEMS), "ITEM")
        b.add(" ")
    b.add(random.choice(ANAPHORIC), "ANAPHORIC")
    if random.random() > 0.35:
        b.add(" ")
        b.add(str(_qty()), "QTY")
        if random.random() > 0.4:
            b.add(" ")
            b.add(_pick(UNITS), "UNIT")


def _qty() -> int:
    r = random.random()
    if r < 0.55:
        return random.choice([10, 12, 15, 16, 20, 24, 25, 30])
    if r < 0.85:
        return random.choice([1, 2, 3, 5, 8])
    return random.choice([40, 50, 60, 70, 80, 100, 200, 320])

PATTERNS = [
    (p_qty_unit,        0.20),   
    (p_item_qty_unit,   0.16),
    (p_item_variant_qty,0.14),
    (p_variant_qty,     0.10),   
    (p_qty_variant,     0.09),   
    (p_nested_box,      0.09),
    (p_reversed,        0.07),
    (p_two_variants,    0.06),
    (p_split_variant,   0.05),   
    (p_anaphoric,       0.04),
]


def generate_message(cake_rate: float = 0.08) -> dict:
    cake = random.random() < cake_rate
    fn = random.choices([f for f, _ in PATTERNS], weights=[w for _, w in PATTERNS])[0]

    b = Builder()
    b.add(random.choice(OPENERS))
    fn(b, cake)
    b.add(random.choice(TIMES))
    b.add(random.choice(TAILS))

    text, spans = b.build()
    return {"text": text, "spans": spans, "pattern": fn.__name__[2:]}

NON_ORDERS = [
    "Kaaa", "Ka", "bu", "Assalamualaikum", "Inggih", "Okey", "Mksh ka",
    "Jam 5 diambl ya", "Lagi di jalan ka", "Ulun ambil kh bu",
    "Kalo bisa aku dp", "Boleh brp", "Duit nya cash aja ya",
    "Ka bisa di gojek akan aja lah", "Jam brp bisaaa ?",
    "Ini ready lah ka resoles nya", "Masih ada kah risol",
    "Besok jualanlah risol", "Oke malam ini ulun dp",
    "Kena d kabari aja klo nya agak telat m ambil nya",
    "Ka duit nya lun tf j lh", "Sore lun ambil", "Otw m ambil ka",
    "harganya masih sama lah bu 2500?", "Bisa pesan risol kah ?",
    "Klo ready mau pesan lg", "Jam brp ready?", "Sdh kah risol",
]

STANDING = [
    "selasa - jumat 34, sabtu 38", "senin- rabu : 34, jumat 30, sabtu 36",
    "hari ini 30 sampai jumat, sabtu 36", "esok 34, jumat 34, sabtu 38",
    "pesan senin- jumat 36, sabtu 40", "senin 34 sm 50, selasa 30, mentah 10",
    "esok selasa-jumat 34, sabtu 38", "selanjutnya senin-jumat 40, sabtu 46",
]


def generate_dataset(n_orders: int, negative_rate: float = 0.35, seed: int = 7):
    random.seed(seed)
    rows = [generate_message() for _ in range(n_orders)]
    for _ in range(int(n_orders * negative_rate)):
        pool = NON_ORDERS if random.random() > 0.15 else STANDING
        rows.append({"text": random.choice(pool), "spans": [],
                    "pattern": "negative"})
    random.shuffle(rows)
    return rows


def validate(rows) -> None:
    for r in rows:
        occupied = []
        for s in r["spans"]:
            frag = r["text"][s["start"]:s["end"]]
            assert frag == s["text"], f"offset mismatch: {frag!r} != {s['text']!r} in {r['text']!r}"
            assert frag and frag == frag.strip(), f"dirty span {frag!r} in {r['text']!r}"
            occupied.append((s["start"], s["end"]))
        occupied.sort()
        for a, b in zip(occupied, occupied[1:]):
            assert a[1] <= b[0], f"overlapping spans in {r['text']!r}"

def write_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["text", "spans", "pattern"])
        w.writeheader()
        for r in rows:
            w.writerow({"text": r["text"],
                        "spans": json.dumps(r["spans"], ensure_ascii=False),
                        "pattern": r["pattern"]})

if __name__ == "__main__":
    from collections import Counter

    data = generate_dataset(n_orders=8000)
    validate(data)
    write_csv(data, "synthetic_orders.csv")

    orders = [r for r in data if r["pattern"] != "negative"]
    print(f"wrote {len(data)} rows ({len(orders)} orders, {len(data)-len(orders)} negatives)\n")

    print("pattern mix:")
    for k, v in Counter(r["pattern"] for r in data).most_common():
        print(f"  {k:<18} {v:>5}")

    has = lambda r, t: any(s["type"] == t for s in r["spans"])
    print(f"\nstructural check against real data (target in brackets):")
    print(f"  no ITEM span   {sum(1 for r in orders if not has(r,'ITEM'))/len(orders)*100:.0f}%  [48%]")
    print(f"  has VARIANT    {sum(1 for r in orders if has(r,'VARIANT'))/len(orders)*100:.0f}%  [41%]")
    print(f"  no UNIT span   {sum(1 for r in orders if not has(r,'UNIT'))/len(orders)*100:.0f}%  [34%]")
    print(f"  2+ QTY         {sum(1 for r in orders if sum(1 for s in r['spans'] if s['type']=='QTY')>=2)/len(orders)*100:.0f}%  [21%]")

    print("\nsamples:")
    for r in data[:8]:
        if r["spans"]:
            print(f"  {r['text']!r}")
            for s in r["spans"]:
                print(f"      {s['type']:<10} {s['text']!r}")