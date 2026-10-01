"""Unit tests verifying strict TypeSafe AI Jev contract serialization and question parsing."""
import pytest
from nirnaya_server.app.core.schema import (
    Answer,
    ChoiceQuestion,
    NoulQuestion,
    Prediction,
    ScoreQuestion,
    parse_question,
    parse_questions,
)


def test_parse_choice_with_criteria():
    """Verify TypeSafe Jev 'criteria' dict is parsed correctly for choice."""
    spec = {
        "type": "choice",
        "instructions": "Select issue domain",
        "criteria": {
            "billing": "Invoice and credit card issues",
            "tech": "Application crash or performance bug",
        },
    }
    q = parse_question("q_choice", spec)
    assert isinstance(q, ChoiceQuestion)
    assert q.qid == "q_choice"
    assert q.instructions == "Select issue domain"
    assert set(q.options.keys()) == {"billing", "tech"}


def test_parse_score_with_criteria():
    """Verify TypeSafe Jev 'criteria' list is parsed correctly for continuous rubric score."""
    spec = {
        "type": "score",
        "instructions": "Rate severity from 0 to 3",
        "criteria": ["low", "medium", "high", "critical"],
    }
    q = parse_question("q_score", spec)
    assert isinstance(q, ScoreQuestion)
    assert q.qid == "q_score"
    assert len(q.levels) == 4
    assert q.levels[0] == "low"
    assert q.levels[3] == "critical"


def test_answer_to_jev_choice():
    """Verify choice Answer.to_jev() matches TypeSafe Jev format."""
    ans = Answer(
        qid="cat",
        qtype="choice",
        value="billing",
        labels=["billing", "tech"],
        probs=[0.92, 0.08],
        raw_probs=[0.90, 0.10],
        logp=[-0.08, -2.52],
        confidence=0.92,
        margin=0.84,
        entropy=0.28,
        valid_mass=0.95,
        agreement=1.0,
    )
    jev = ans.to_jev()
    assert jev["type"] == "choice"
    assert jev["choice"] == "billing"
    assert jev["confidence"] == 0.92
    assert "probabilities" in jev
    assert jev["probabilities"]["billing"] == 0.92
    assert jev["probabilities"]["tech"] == 0.08


def test_answer_to_jev_score():
    """Verify score Answer.to_jev() matches TypeSafe Jev format with legend."""
    q = ScoreQuestion(
        qid="urgency",
        instructions="Rate urgency",
        levels=["negligible", "moderate", "severe"],
    )
    ans = Answer(
        qid="urgency",
        qtype="score",
        value=1.8542,
        labels=["0", "1", "2"],
        probs=[0.05, 0.15, 0.80],
        raw_probs=[0.05, 0.15, 0.80],
        logp=[-3.0, -1.9, -0.22],
        confidence=0.80,
        margin=0.65,
        entropy=0.35,
        valid_mass=0.94,
        agreement=1.0,
    )
    jev = ans.to_jev(question=q)
    assert jev["type"] == "score"
    assert jev["score"] == 1.8542
    assert jev["confidence"] == 0.80
    assert jev["probabilities"]["2"] == 0.80
    assert "legend" in jev
    assert jev["legend"] == {"0": "negligible", "1": "moderate", "2": "severe"}


def test_answer_to_jev_noul():
    """Verify noul Answer.to_jev() matches TypeSafe Jev format."""
    ans = Answer(
        qid="escalate",
        qtype="noul",
        value=0.9421,
        labels=["yes", "no"],
        probs=[0.9421, 0.0579],
        raw_probs=[0.9421, 0.0579],
        logp=[-0.06, -2.85],
        confidence=0.9421,
        margin=0.8842,
        entropy=0.21,
        valid_mass=0.89,
        agreement=1.0,
    )
    jev = ans.to_jev()
    assert jev["type"] == "noul"
    assert jev["noul"] == 0.9421


def test_prediction_to_jev_payload():
    """Verify full Prediction payload meets TypeSafe Jev response contract."""
    ans_choice = Answer(
        qid="c1", qtype="choice", value="optA", labels=["optA", "optB"],
        probs=[0.95, 0.05], raw_probs=[0.95, 0.05], logp=[-0.05, -3.0],
        confidence=0.95, margin=0.90, entropy=0.2, valid_mass=0.9, agreement=1.0
    )
    ans_noul = Answer(
        qid="n1", qtype="noul", value=0.88, labels=["yes", "no"],
        probs=[0.88, 0.12], raw_probs=[0.88, 0.12], logp=[-0.12, -2.1],
        confidence=0.88, margin=0.76, entropy=0.3, valid_mass=0.9, agreement=1.0
    )
    pred = Prediction(
        model="qwen3-0.6b-gguf",
        answers={"c1": ans_choice, "n1": ans_noul},
        tokens={"static": 100, "state": 50, "q_c1": 20, "q_n1": 15},
    )
    payload = pred.to_jev()
    assert payload["model"] == "qwen3-0.6b-gguf"
    assert "answers" in payload
    assert payload["answers"]["c1"]["type"] == "choice"
    assert payload["answers"]["c1"]["choice"] == "optA"
    assert payload["answers"]["n1"]["type"] == "noul"
    assert payload["answers"]["n1"]["noul"] == 0.88
    assert "usage" in payload
    assert payload["usage"]["input_tokens"] == 185
    assert payload["usage"]["output_tokens"] == 2
