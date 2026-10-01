"""Evaluation harness: accuracy, calibration (ECE / Brier), selective accuracy, latency."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Any, Dict, List, Sequence

import numpy as np

from .calibration import Calibrator


def load_cases(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)["cases"]


def gold_index(spec: Dict[str, Any]) -> int:
    """Map a gold label to the index used in Answer.probs / Answer.logp."""
    gold = spec["gold"]
    if spec["type"] == "choice":
        return list(spec["options"]).index(gold)
    if spec["type"] == "score":
        return int(gold)
    return 0 if bool(gold) else 1          # noul: index 0 == "yes"


def metrics(probs: Sequence[Sequence[float]], gold: Sequence[int], qtype: str = "") -> Dict[str, float]:
    """Per-example computation, so questions with different label counts can be mixed."""
    n = len(gold)
    conf = np.asarray([max(p) for p in probs], dtype=float)
    correct = np.asarray([float(int(np.argmax(p)) == y) for p, y in zip(probs, gold)])
    brier = float(np.mean([((np.asarray(p) - np.eye(len(p))[y]) ** 2).sum() for p, y in zip(probs, gold)]))

    ece, edges = 0.0, np.linspace(0, 1, 11)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())

    top = np.argsort(-conf)[: max(1, math.ceil(0.5 * n))]
    out = {"n": n, "accuracy": float(correct.mean()), "mean_confidence": float(conf.mean()),
           "ece": float(ece), "brier": brier, "acc_at_50pct_coverage": float(correct[top].mean())}
    if qtype == "score":
        out["mae_expected_level"] = float(np.mean(
            [abs(float((np.asarray(p) * np.arange(len(p))).sum()) - y) for p, y in zip(probs, gold)]))
    return out


def run_eval(engine, cases: List[Dict[str, Any]], calibrate: bool = False, folds: int = 5,
             verbose: bool = True) -> Dict[str, Any]:
    records, latencies, per_question = [], [], []
    for case in cases:
        specs = {qid: {k: v for k, v in q.items() if k != "gold"} for qid, q in case["questions"].items()}
        pred = engine.predict(case["state"], specs)
        latencies.append(pred.timings_ms["total"])
        for qid, ans in pred.answers.items():
            gi = gold_index(case["questions"][qid])
            rec = {"id": f"{case['id']}.{qid}", "qtype": ans.qtype, "gold": gi, "logp": ans.logp,
                   "probs": ans.probs, "valid_mass": ans.valid_mass, "agreement": ans.agreement}
            records.append(rec)
            per_question.append({**{k: rec[k] for k in ("id", "qtype", "gold")},
                                 "pred": int(np.argmax(ans.probs)), "confidence": ans.confidence,
                                 "value": ans.value, "valid_mass": ans.valid_mass, "agreement": ans.agreement})
        if verbose:
            print(f"  {case['id']}: {pred.timings_ms['total']:.0f} ms  "
                  f"({len(specs)} questions, state={pred.tokens['state']} tok)")

    by_type: Dict[str, List[dict]] = defaultdict(list)
    for r in records:
        by_type[r["qtype"]].append(r)

    report: Dict[str, Any] = {"raw": {}, "calibrated_crossfit": {}, "diagnostics": {}}
    for qtype, recs in by_type.items():
        report["raw"][qtype] = metrics([r["probs"] for r in recs], [r["gold"] for r in recs], qtype)
    report["raw"]["ALL"] = metrics([r["probs"] for r in records], [r["gold"] for r in records])

    if calibrate:
        cal_probs: Dict[str, List[List[float]]] = {}
        for qtype, recs in by_type.items():
            k = min(folds, len(recs))
            out = [None] * len(recs)
            for f in range(k):
                tr = [r for i, r in enumerate(recs) if i % k != f]
                cal = Calibrator().fit(qtype, [r["logp"] for r in tr], [r["gold"] for r in tr])
                for i, r in enumerate(recs):
                    if i % k == f:
                        out[i] = cal.apply(qtype, r["logp"])
            cal_probs[qtype] = out
            report["calibrated_crossfit"][qtype] = metrics(out, [r["gold"] for r in recs], qtype)
        flat = [(p, r["gold"]) for t, rs in by_type.items() for p, r in zip(cal_probs[t], rs)]
        report["calibrated_crossfit"]["ALL"] = metrics([p for p, _ in flat], [g for _, g in flat])

    lat = np.asarray(latencies)
    report["diagnostics"] = {
        "mean_valid_mass": float(np.mean([r["valid_mass"] for r in records])),
        "mean_perm_agreement_choice": float(np.mean([r["agreement"] for r in by_type.get("choice", [{"agreement": 1}])])),
        "latency_ms_per_case": {"mean": float(lat.mean()), "p50": float(np.percentile(lat, 50)),
                                "p95": float(np.percentile(lat, 95))},
    }
    report["per_question"] = per_question
    return report


def print_report(report: Dict[str, Any]) -> None:
    def table(title: str, block: Dict[str, Dict[str, float]]) -> None:
        print(f"\n{title}")
        print(f"{'type':<8}{'n':>4}{'acc':>8}{'conf':>8}{'ECE':>8}{'Brier':>8}{'acc@50%':>9}")
        for t, m in block.items():
            print(f"{t:<8}{m['n']:>4}{m['accuracy']:>8.3f}{m['mean_confidence']:>8.3f}{m['ece']:>8.3f}"
                  f"{m['brier']:>8.3f}{m.get('acc_at_50pct_coverage', float('nan')):>9.3f}")

    table("RAW (uncalibrated)", report["raw"])
    if report["calibrated_crossfit"]:
        table("CALIBRATED (k-fold cross-fitted, held-out predictions only)", report["calibrated_crossfit"])
    d = report["diagnostics"]
    print(f"\nvalid-token mass (mean): {d['mean_valid_mass']:.3f}   "
          f"choice permutation agreement: {d['mean_perm_agreement_choice']:.3f}")
    l = d["latency_ms_per_case"]
    print(f"latency per case (state + 3 questions): mean {l['mean']:.0f} ms | p50 {l['p50']:.0f} | p95 {l['p95']:.0f}")
