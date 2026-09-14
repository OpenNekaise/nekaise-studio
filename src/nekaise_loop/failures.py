"""Expected provider unavailability is recoverable state, not failed training."""
from __future__ import annotations

import re


class TeacherUnavailable(RuntimeError):
    def __init__(self, message, kind="quota", retry_seconds=None):
        super().__init__(message)
        self.kind, self.retry_seconds = kind, retry_seconds


def quota_kind(message):
    text = str(message).lower()
    if any(s in text for s in ("out of usage credits", "usage limit", "usage_limit_reached", "hit your limit", "insufficient_quota", "quota exceeded", "exceeded your current quota", "credit balance", "not enough credits", "weekly limit", "monthly limit")):
        return "quota"
    if any(s in text for s in ("rate limit", "rate_limit", "too many requests", "http 429", '"api_error_status":429')):
        return "rate_limit"
    return None


def retry_seconds(message):
    match = re.search(r"retry[- ]after\s*[:=]?\s*(\d+)\s*(?:s|seconds)?", str(message), re.I)
    return max(30, min(86400, int(match[1]))) if match else None
