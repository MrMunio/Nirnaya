"""Typed question / answer schema for Nirnaya.

Three decision primitives (mirroring the "System One" idea):

* choice - pick exactly one of N options (N <= 26, single-token letter labels A-Z)
* score  - place the state on an ordinal rubric (levels 0..K-1, single-token digits)
* noul   - calibrated yes/no; the answer is P(true)
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Union

from .symbols import CHOICE_SYMBOLS_255, MAX_CHOICES

MAX_LEVELS = 10    # digits 0-9, each a single token in Qwen / Llama-style tokenizers


class SchemaError(ValueError):
    """Raised when a question spec is malformed."""


@dataclass(frozen=True)
class ChoiceQuestion:
    qid: str
    instructions: str
    options: Dict[str, str]          # option key -> short description (order preserved)
    qtype: str = "choice"


@dataclass(frozen=True)
class ScoreQuestion:
    qid: str
    instructions: str
    levels: List[str]                # index in the list == numeric level
    qtype: str = "score"


@dataclass(frozen=True)
class NoulQuestion:
    qid: str
    instructions: str
    qtype: str = "noul"


Question = Union[ChoiceQuestion, ScoreQuestion, NoulQuestion]


def parse_question(qid: str, spec: Dict[str, Any]) -> Question:
    """Validate a raw dict spec and turn it into a typed question object."""
    qtype = spec.get("type")
    instructions = str(spec.get("instructions", "")).strip()
    if not instructions:
        raise SchemaError(f"[{qid}] 'instructions' is required")

    if qtype == "choice":
        options = spec.get("criteria") if "criteria" in spec else spec.get("options")
        if isinstance(options, list):
            options = {str(k): "" for k in options}
        if not isinstance(options, dict) or not (2 <= len(options) <= MAX_CHOICES):
            raise SchemaError(f"[{qid}] choice needs 2..{MAX_CHOICES} options or criteria")
        return ChoiceQuestion(qid, instructions, {str(k): str(v or "") for k, v in options.items()})

    if qtype == "score":
        levels = spec.get("criteria") if "criteria" in spec else spec.get("levels")
        if not isinstance(levels, list) or not (2 <= len(levels) <= MAX_LEVELS):
            raise SchemaError(f"[{qid}] score needs 2..{MAX_LEVELS} levels or criteria")
        return ScoreQuestion(qid, instructions, [str(v) for v in levels])

    if qtype == "noul":
        return NoulQuestion(qid, instructions)

    raise SchemaError(f"[{qid}] unknown question type: {qtype!r}")


def parse_questions(specs: Union[Dict[str, Dict[str, Any]], List[Question]]) -> List[Question]:
    if isinstance(specs, list):
        return specs
    return [parse_question(qid, spec) for qid, spec in specs.items()]


@dataclass
class Answer:
    """Result for one question.

    value:  choice -> winning option key | score -> expected level | noul -> P(true)
    labels: label order used for probs/logp (choice: option keys, score: "0".."K-1", noul: ["yes","no"])
    logp:   pooled log-probabilities over labels BEFORE calibration (the calibration input)
    """
    qid: str
    qtype: str
    value: Any
    labels: List[str]
    probs: List[float]
    raw_probs: List[float]
    logp: List[float]
    confidence: float
    margin: float
    entropy: float          # normalized to [0, 1]
    valid_mass: float       # raw probability mass the model put on valid label tokens
    agreement: float        # share of option-order permutations agreeing with the final argmax

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_jev(self, question: Optional[Question] = None, precision: int = 3) -> Dict[str, Any]:
        """Format answer strictly adhering to TypeSafe Jev API response contract."""
        if self.qtype == "choice":
            probs_dict = {lbl: round(float(p), precision) for lbl, p in zip(self.labels, self.probs)}
            return {
                "type": "choice",
                "choice": self.value,
                "confidence": round(float(self.confidence), precision),
                "probabilities": probs_dict,
            }
        elif self.qtype == "score":
            probs_dict = {str(lbl): round(float(p), precision) for lbl, p in zip(self.labels, self.probs)}
            res: Dict[str, Any] = {
                "type": "score",
                "score": round(float(self.value), precision),
                "confidence": round(float(self.confidence), precision),
                "probabilities": probs_dict,
            }
            if question is not None and isinstance(question, ScoreQuestion):
                res["legend"] = {str(i): desc for i, desc in enumerate(question.levels)}
            return res
        else:  # noul
            return {
                "type": "noul",
                "noul": round(float(self.value), precision),
            }


@dataclass
class Prediction:
    answers: Dict[str, Answer]
    timings_ms: Dict[str, float] = field(default_factory=dict)
    tokens: Dict[str, int] = field(default_factory=dict)
    model: str = "nirnaya"
    questions: Dict[str, Question] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "answers": {k: v.to_dict() for k, v in self.answers.items()},
            "timings_ms": self.timings_ms,
            "tokens": self.tokens,
        }

    def to_jev(self) -> Dict[str, Any]:
        """Return the exact response contract expected from TypeSafe Jev /v1/systemone."""
        return {
            "model": self.model,
            "answers": {
                qid: ans.to_jev(self.questions.get(qid))
                for qid, ans in self.answers.items()
            },
            "usage": {
                "input_tokens": sum(self.tokens.values()),
                "output_tokens": len(self.answers),
            },
        }

