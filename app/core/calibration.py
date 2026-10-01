"""Post-hoc calibration (the training-free stand-in for RLCD).

Temperature scaling per question type; noul additionally learns a bias on the "yes" logit
(Platt-lite). Tiny grid search, numpy only. A light L2 prior keeps the fit near the identity
when data is scarce. Records may have different label counts (e.g. 4- and 5-option choices).
"""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Dict, List, Sequence

import numpy as np

from .readout import softmax


class Calibrator:
    def __init__(self, params: Dict[str, Dict[str, float]] | None = None):
        self.params: Dict[str, Dict[str, float]] = params or {}

    def has(self, qtype: str) -> bool:
        return qtype in self.params

    @staticmethod
    def _scale(X: np.ndarray, T: float, b: float) -> np.ndarray:
        z = X / T
        z = z.copy()
        z[..., 0] += b            # b is only ever non-zero for noul, where label 0 == "yes"
        return z

    @staticmethod
    def _nll_sum(z: np.ndarray, y: np.ndarray) -> float:
        z = z - z.max(axis=1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        return float(-logp[np.arange(len(y)), y].sum())

    def fit(self, qtype: str, logps: Sequence[Sequence[float]], gold: Sequence[int]) -> "Calibrator":
        groups = defaultdict(lambda: ([], []))                 # group by label count K
        for lp, g in zip(logps, gold):
            groups[len(lp)][0].append(lp)
            groups[len(lp)][1].append(g)
        data = [(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=int)) for x, y in groups.values()]
        n_total = len(gold)
        biases = np.linspace(-3, 3, 61) if qtype == "noul" else [0.0]

        best = (np.inf, 1.0, 0.0)
        for log_t in np.linspace(-3, 3, 121):
            T = float(np.exp(log_t))
            for b in biases:
                loss = sum(self._nll_sum(self._scale(X, T, float(b)), y) for X, y in data) / n_total
                loss += 1e-3 * (log_t ** 2 + b ** 2)
                if loss < best[0]:
                    best = (loss, T, float(b))
        self.params[qtype] = {"T": best[1], "b": best[2]}
        return self

    def apply(self, qtype: str, logp: Sequence[float]) -> List[float]:
        p = self.params[qtype]
        z = self._scale(np.asarray(logp, dtype=np.float64)[None, :], p["T"], p["b"])[0]
        return softmax(z).tolist()

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.params, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Calibrator":
        with open(path) as f:
            return cls(json.load(f))
