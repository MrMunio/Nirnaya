"""Turn a next-token logit vector into a typed, type-safe answer.

Steps: full-vocab log-softmax -> logsumexp over each label's token ids -> renormalize over
the valid labels only (type-safety by construction) -> pool across option-order
permutations in log space -> optional calibration -> confidence metrics.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from .labels import LabelSpace
from .schema import Answer, ChoiceQuestion, NoulQuestion, Question, ScoreQuestion


def softmax(x: Sequence[float]) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    a = a - a.max()
    e = np.exp(a)
    return e / e.sum()


def log_softmax(x: Sequence[float]) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    m = a.max()
    return a - (m + np.log(np.exp(a - m).sum()))


def label_logprobs(logits, space: LabelSpace) -> Tuple[List[float], float]:
    """logits: 1-D tensor or numpy array over the vocab. Returns (restricted log-probs, valid_mass)."""
    try:
        import torch
        if isinstance(logits, torch.Tensor):
            logp_full = torch.log_softmax(logits.float(), dim=-1)
            per_label = torch.stack([
                torch.logsumexp(logp_full[torch.tensor(ids, device=logp_full.device)], dim=0)
                for ids in space.token_ids
            ])
            valid_log_mass = torch.logsumexp(per_label, dim=0)
            restricted = per_label - valid_log_mass
            return restricted.tolist(), float(valid_log_mass.exp())
    except ImportError:
        pass

    from scipy.special import logsumexp
    arr = np.asarray(logits, dtype=np.float64)
    logp_full = arr - logsumexp(arr)
    per_label = np.array([
        logsumexp(logp_full[ids]) if len(ids) > 1 else logp_full[ids[0]]
        for ids in space.token_ids
    ], dtype=np.float64)
    valid_log_mass = float(logsumexp(per_label))
    restricted = per_label - valid_log_mass
    return restricted.tolist(), float(np.exp(valid_log_mass))


def pool_logps(logps: Sequence[Sequence[float]]) -> np.ndarray:
    """Geometric pooling (mean of log-probs, renormalized) - a product-of-experts across permutations."""
    return log_softmax(np.mean(np.asarray(logps, dtype=np.float64), axis=0))


def build_answer(q: Question, labels: List[str], pooled_logp: np.ndarray, valid_mass: float,
                 agreement: float, calibrator=None) -> Answer:
    raw = softmax(pooled_logp)
    if calibrator is not None and calibrator.has(q.qtype):
        probs = np.asarray(calibrator.apply(q.qtype, pooled_logp.tolist()))
    else:
        probs = raw

    order = np.argsort(-probs)
    confidence = float(probs[order[0]])
    margin = float(probs[order[0]] - probs[order[1]])
    k = len(probs)
    ent = float(-(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum() / np.log(k))

    if isinstance(q, ChoiceQuestion):
        value = labels[int(order[0])]
    elif isinstance(q, ScoreQuestion):
        value = float((probs * np.arange(k)).sum())
    else:  # NoulQuestion: labels == ["yes", "no"]
        value = float(probs[0])

    return Answer(
        qid=q.qid, qtype=q.qtype, value=value, labels=labels,
        probs=probs.tolist(), raw_probs=raw.tolist(), logp=pooled_logp.tolist(),
        confidence=confidence, margin=margin, entropy=ent,
        valid_mass=float(valid_mass), agreement=float(agreement),
    )
