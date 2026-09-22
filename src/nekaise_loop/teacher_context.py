"""Exact, retrievable evidence views for the Teacher; never a history filter."""
from collections import Counter
import json

from .artifacts import digest


REFERENCE_KEY = "$shared"
MIN_SHARED_CHARS = 512
EVIDENCE_KEY = "$evidence"


def contains_key(value, key):
    if isinstance(value, dict):
        return key in value or any(contains_key(v, key) for v in value.values())
    return isinstance(value, list) and any(contains_key(v, key) for v in value)


def resolve_pointer(data, pointer):
    """Resolve an RFC 6901 JSON pointer without evaluating path expressions."""
    if not isinstance(pointer, str) or (pointer and not pointer.startswith("/")):
        raise ValueError("Use an empty root pointer or an RFC 6901 /field/index pointer")
    value = data
    for part in pointer.split("/")[1:]:
        if any(part[i:i+2] not in {"~0", "~1"} for i, char in enumerate(part) if char == "~"):
            raise ValueError("Invalid JSON pointer escape")
        key = part.replace("~1", "/").replace("~0", "~")
        if isinstance(value, list):
            if not key.isascii() or not key.isdecimal() or (len(key) > 1 and key[0] == "0"):
                raise ValueError("JSON array pointers require a nonnegative canonical index")
            if int(key) >= len(value):
                raise ValueError("JSON pointer array index is out of range")
            value = value[int(key)]
        elif isinstance(value, dict):
            if key not in value:
                raise ValueError(f"JSON pointer field is absent: {key}")
            value = value[key]
        else:
            raise ValueError("JSON pointer traverses a scalar")
    return value


def evidence_page(data, request):
    """Return exact fields/pages with the identity of the complete pointed value."""
    pointer = request.get("pointer", "")
    value = resolve_pointer(data, pointer)
    result = {"pointer": pointer, "canonical_sha256": digest(value)}
    fields = request.get("fields")
    if fields is not None:
        if not isinstance(value, dict) or not isinstance(fields, list) or not all(isinstance(k, str) for k in fields):
            raise ValueError("fields requires an object and a list of exact field names")
        if any(k not in value for k in fields):
            raise ValueError("Requested fields are absent: " + ", ".join(k for k in fields if k not in value))
        result.update(value={k: value[k] for k in fields}, fields=fields)
    elif isinstance(value, (list, str)):
        offset, limit = int(request.get("offset", 0)), int(request.get("limit", 20))
        if isinstance(value, str):
            offset, limit = int(request.get("start", 0)), int(request.get("length", 4000))
        if offset < 0 or not 1 <= limit <= (20000 if isinstance(value, str) else 200):
            raise ValueError("Use nonnegative offsets, array limit 1..200 or string length 1..20000")
        next_key = "next_start" if isinstance(value, str) else "next_offset"
        result.update(value=value[offset:offset+limit], total=len(value),
                      **{next_key: offset+limit if offset+limit < len(value) else None})
    else:
        result["value"] = value
    return result


def evidence_view(data, purpose, locator, *, pointer=""):
    """Reference only source copies and raw ID arrays, preserving all other fields.

    Revise/evaluate retain exact source spans inline. Reflect and candidate review
    can retrieve source text while reading exact teaching/assessment text eagerly.
    No record, observation, error, instruction or history entry is dropped.
    """
    if purpose not in {"curriculum", "revise", "material_select", "evaluate", "reflect"} or contains_key(data, EVIDENCE_KEY):
        return data

    references = {}
    def reference(value, path):
        key = digest(value)
        # Identical values can use the first exact location. Path differences must
        # not prevent sharing the same source/metadata across many lesson rows.
        return references.setdefault(key, {EVIDENCE_KEY: {**locator, "pointer": path, "canonical_sha256": key,
                              "type": "string" if isinstance(value, str) else "array", "total": len(value)}})

    def render(value, path, *, source=False):
        if isinstance(value, dict):
            result = {}
            for key, child in value.items():
                child_path = path + "/" + key.replace("~", "~0").replace("/", "~1")
                raw_ids = key in {"prompt_token_ids", "generated_token_ids", "raw_argmax_token_ids"} and isinstance(child, list)
                source_text = key == "text" and source and isinstance(child, str)
                # Short values cost less than a reference and need no indirection.
                if (raw_ids or source_text) and len(json.dumps(child, ensure_ascii=False)) > 512:
                    result[key] = reference(child, child_path)
                elif key == "document" and isinstance(child, dict):
                    spans = value.get("sources", [])
                    if isinstance(spans, dict):
                        spans = spans.values()
                    elif not isinstance(spans, list):
                        spans = []
                    duplicate = any(isinstance(s, dict) and s.get("text") == child.get("text") for s in spans)
                    result[key] = render(child, child_path, source=purpose not in {"revise", "evaluate"} or duplicate)
                elif key == "sources" and isinstance(child, (list, dict)):
                    lazy = purpose in {"reflect", "material_select"}
                    if isinstance(child, list):
                        result[key] = [render(v, child_path+f"/{i}", source=lazy) for i, v in enumerate(child)]
                    else:
                        result[key] = {k: render(v, child_path+"/"+k.replace("~", "~0").replace("/", "~1"), source=lazy) for k, v in child.items()}
                else:
                    result[key] = render(child, child_path)
            return result
        if isinstance(value, list):
            return [render(v, path+f"/{i}") for i, v in enumerate(value)]
        return value
    return render(data, pointer)


def shared_view(inputs):
    """Return a smaller, shallow reference table or None; never mutate evidence.

    Only exact repeated JSON values qualify. Shared entries remain original values,
    without nested references, so every reference resolves in one lookup. A literal
    reference marker in the original evidence disables this optional representation.
    """
    counts, encodings, collision = Counter(), {}, False
    def inspect(value):
        nonlocal collision
        if isinstance(value, dict):
            collision |= REFERENCE_KEY in value
            for child in value.values():
                inspect(child)
        elif isinstance(value, list):
            for child in value:
                inspect(child)
        if isinstance(value, (dict, list, str)):
            encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
            if len(encoded) >= MIN_SHARED_CHARS:
                encodings[id(value)] = encoded
                counts[encoded] += 1
    inspect(inputs)
    if collision:
        return None
    shared = {}
    def render(value):
        encoded = encodings.get(id(value))
        if encoded is not None and counts[encoded] > 1:
            key = digest(value)
            shared[key] = value
            return {REFERENCE_KEY: key}
        if isinstance(value, dict):
            return {key: render(child) for key, child in value.items()}
        if isinstance(value, list):
            return [render(child) for child in value]
        return value
    data = render(inputs)
    if not shared:
        return None
    result = {"format": "shared_values_v1", "reference_key": REFERENCE_KEY,
              "shared_values": dict(sorted(shared.items())), "data": data}
    if len(json.dumps(result, ensure_ascii=False)) >= len(json.dumps(inputs, ensure_ascii=False)):
        return None
    return result
