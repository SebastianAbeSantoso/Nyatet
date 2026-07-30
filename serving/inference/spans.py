def collapse_bio(tokens: list[str], tags: list[str], scores: list[float] | None = None) -> list[dict]:
    spans: list[dict] = []
    current: dict | None = None

    for i, tag in enumerate(tags):
        if tag == "O" or tag is None:
            if current:
                spans.append(current)
                current = None
            continue

        if "-" in tag:
            prefix, ent_type = tag.split("-", 1)
        else:  
            prefix, ent_type = "B", tag

        starts_new = prefix == "B" or current is None or current["type"] != ent_type

        if starts_new:
            if current:
                spans.append(current)
            current = {
                "type": ent_type,
                "_tokens": [tokens[i]],
                "_scores": [scores[i]] if scores else [],
                "start_token": i,
                "end_token": i,
            }
        else:
            current["_tokens"].append(tokens[i])
            if scores:
                current["_scores"].append(scores[i])
            current["end_token"] = i

    if current:
        spans.append(current)

    out = []
    for s in spans:
        toks = s.pop("_tokens")
        scs = s.pop("_scores")
        s["text"] = _detokenize(toks)
        s["confidence"] = round(sum(scs) / len(scs), 4) if scs else None
        out.append(s)
    return out


def _detokenize(tokens: list[str]) -> str:
    text = ""
    for t in tokens:
        if t.startswith("##"):
            text += t[2:]
        else:
            text += (" " if text else "") + t
    return text