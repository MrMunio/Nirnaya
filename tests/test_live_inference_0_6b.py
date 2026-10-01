"""Live inference test verifying Qwen3-0.6B-GGUF execution on CPU."""
import os
from pathlib import Path
import pytest

from nirnaya_server.app.services.engine_service import engine_service


def test_live_qwen3_0_6b_inference():
    """Executes a real decision pass on CPU using the downloaded Qwen3-0.6B model."""
    model_path = Path("models/qwen3-0.6b/Qwen3-0.6B-Q8_0.gguf").resolve()
    if not model_path.exists():
        pytest.skip(f"Model file not found at {model_path}")

    state = (
        "Customer message: 'My credit card was charged twice for $99 on invoice #1042. "
        "Please reverse the duplicate charge immediately!'"
    )

    questions = {
        "category": {
            "type": "choice",
            "instructions": "Select the primary category of this ticket",
            "criteria": {
                "billing": "Invoice, charge, refund, payment issues",
                "tech": "Software crash, bug, outage",
                "general": "General questions",
            },
        },
        "urgency": {
            "type": "score",
            "instructions": "Rate the urgency of this ticket from 0 to 4",
            "criteria": ["trivial", "low", "medium", "high", "critical"],
        },
        "is_billing_issue": {
            "type": "noul",
            "instructions": "Is this inquiry directly related to billing or payment charges?",
        },
    }

    # Execute inference with qwen3-0.6b-gguf
    pred = engine_service.predict(state=state, questions=questions, model_name="qwen3-0.6b-gguf")
    assert pred is not None
    assert len(pred.answers) == 3

    jev = pred.to_jev()
    assert jev["model"] == "qwen3-0.6b-gguf"
    assert "answers" in jev

    # 1. Choice Answer Validation
    cat_ans = jev["answers"]["category"]
    assert cat_ans["type"] == "choice"
    assert cat_ans["choice"] in ("billing", "tech", "general")
    assert 0.0 <= cat_ans["confidence"] <= 1.0
    assert "billing" in cat_ans["probabilities"]

    # 2. Score Answer Validation (Rubric Expected Value)
    urg_ans = jev["answers"]["urgency"]
    assert urg_ans["type"] == "score"
    assert 0.0 <= urg_ans["score"] <= 4.0
    assert 0.0 <= urg_ans["confidence"] <= 1.0
    assert "legend" in urg_ans
    assert urg_ans["legend"]["0"] == "trivial"

    # 3. Noul Answer Validation (P(true))
    noul_ans = jev["answers"]["is_billing_issue"]
    assert noul_ans["type"] == "noul"
    assert 0.0 <= noul_ans["noul"] <= 1.0

    print("\n[Live Inference Test Output - Qwen3-0.6B]:")
    print(f"  • Category: {cat_ans['choice']} (confidence: {cat_ans['confidence']:.2f})")
    print(f"  • Urgency Score: {urg_ans['score']:.2f} / 4.0 (confidence: {urg_ans['confidence']:.2f})")
    print(f"  • Is Billing Issue: {noul_ans['noul']:.4f}")
