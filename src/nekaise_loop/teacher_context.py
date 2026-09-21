"""Lossless sharing of repeated evidence in the Teacher's rendered context."""
from collections import Counter
import json

from .artifacts import digest


REFERENCE_KEY = "$shared"
MIN_SHARED_CHARS = 2048


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
