"""Single-token label resolution.

The core constraint of the PoC: every answer label must be ONE token, because we read a
single next-token distribution. A label may have several surface forms ("A" and " A",
"Yes" and " Yes" ...). We sum probability over all single-token forms of a label.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence


class LabelError(ValueError):
    """Raised when a label cannot be represented as a single token (or labels collide)."""


@dataclass
class LabelSpace:
    labels: List[str]
    token_ids: List[List[int]]      # one list of token ids per label


def _variants(label: str, case_variants: bool) -> List[str]:
    forms = {label}
    if case_variants:
        forms |= {label.lower(), label.upper(), label.capitalize()}
    out = set()
    for f in forms:
        out.add(f)
        out.add(" " + f)
    return sorted(out)


def _single_token(tokenizer, text: str) -> int | None:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None


def build_label_space(tokenizer, labels: Sequence[str], case_variants: bool = False) -> LabelSpace:
    token_ids: List[List[int]] = []
    owner = {}
    for label in labels:
        ids = []
        for form in _variants(label, case_variants):
            tid = _single_token(tokenizer, form)
            if tid is not None and tid not in ids:
                ids.append(tid)
        if not ids:
            raise LabelError(
                f"Label {label!r} has no single-token form in this tokenizer. "
                "Use shorter labels (letters / digits) or fewer options."
            )
        for tid in ids:
            if tid in owner and owner[tid] != label:
                raise LabelError(f"Token id {tid} is shared by labels {owner[tid]!r} and {label!r}")
            owner[tid] = label
        token_ids.append(ids)
    return LabelSpace(list(labels), token_ids)
