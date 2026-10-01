"""Unit tests for Streamlit frontend configuration, client, and templates."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from streamlit_app.config import StreamlitSettings
from streamlit_app.api_client import NirnayaClient, HealthStatus
from streamlit_app.templates import SHOWCASE_TEMPLATES


def test_streamlit_config_defaults() -> None:
    settings = StreamlitSettings()
    assert settings.api_timeout_seconds >= 60.0, "Timeout should be at least 60s for CPU inference"
    assert settings.verify_credentials("admin", "nirnaya") is True
    assert settings.verify_credentials("admin", "wrong_password") is False
    assert settings.verify_credentials("unknown_user", "password") is False


def test_streamlit_config_custom_users() -> None:
    settings = StreamlitSettings(
        admin_username="test_admin",
        admin_password="secret_pass",
        extra_users_str="auditor:audit123,dev:dev456",
    )
    assert settings.verify_credentials("test_admin", "secret_pass") is True
    assert settings.verify_credentials("auditor", "audit123") is True
    assert settings.verify_credentials("dev", "dev456") is True
    assert settings.verify_credentials("hacker", "pass") is False


def test_api_client_headers_and_timeout() -> None:
    client = NirnayaClient(
        base_url="http://localhost:8000",
        api_key="test_api_token",
        timeout_seconds=300.0,
        connect_timeout_seconds=15.0,
    )
    headers = client._get_headers()
    assert headers["Authorization"] == "Bearer test_api_token"
    assert headers["X-API-Key"] == "test_api_token"
    assert client.timeout_seconds == 300.0
    assert client.connect_timeout == 15.0


def test_api_client_health_offline() -> None:
    # Point to non-existent port on localhost
    client = NirnayaClient(
        base_url="http://127.0.0.1:59999",
        api_key="test_key",
        timeout_seconds=2.0,
        connect_timeout_seconds=2.0,
    )
    status = client.check_health()
    assert isinstance(status, HealthStatus)
    assert status.is_connected is False
    assert status.is_ready is False


def test_api_client_predict_mock() -> None:
    client = NirnayaClient(base_url="http://localhost:8000", api_key="test_key", timeout_seconds=60.0)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "answers": {
            "urgency": {
                "question_id": "urgency",
                "decision_type": "score",
                "predicted_value": 1.84,
                "confidence": 0.89,
                "distribution": {"0": 0.05, "1": 0.06, "2": 0.89},
            }
        },
        "model": "qwen3-0.6b-gguf",
        "usage": {"input_tokens": 120, "output_tokens": 1},
    }

    with patch("httpx.Client.post", return_value=mock_resp):
        res = client.execute_system_one(
            state="Sample ticket",
            questions={"urgency": {"type": "score", "instructions": "test"}},
        )
        assert "answers" in res
        assert "urgency" in res["answers"]
        assert res["answers"]["urgency"]["predicted_value"] == 1.84
        assert "_client_roundtrip_ms" in res


def test_showcase_templates_structure() -> None:
    assert len(SHOWCASE_TEMPLATES) >= 4

    for name, template in SHOWCASE_TEMPLATES.items():
        assert "description" in template
        assert "state" in template
        assert "questions" in template
        assert isinstance(template["questions"], dict)
        assert len(template["questions"]) >= 1

        for q_id, q_data in template["questions"].items():
            assert "type" in q_data
            assert q_data["type"] in ("choice", "score", "noul")
            assert "instructions" in q_data
            if q_data["type"] == "choice":
                assert "criteria" in q_data or "options" in q_data
            elif q_data["type"] == "score":
                assert "criteria" in q_data or "levels" in q_data


def test_api_client_list_models() -> None:
    client = NirnayaClient(base_url="http://localhost:8000")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "models": [
            {"alias": "qwen3-0.6b-gguf", "name": "Qwen3 0.6B"},
            {"alias": "bonsai-27b", "name": "Bonsai 27B"},
        ],
        "hardware": {"preferred_device": "cpu"},
    }

    with patch("httpx.Client.get", return_value=mock_resp):
        res = client.list_models()
        assert "models" in res
        assert isinstance(res["models"], list)
        aliases = [m["alias"] for m in res["models"]]
        assert "qwen3-0.6b-gguf" in aliases


def test_render_decision_result_typesafe_jev() -> None:
    from streamlit_app.components import render_decision_result

    # 1. Choice test
    q_def_choice = {"type": "choice", "instructions": "Select dept", "criteria": {"billing": "billing desc"}}
    ans_choice = {"type": "choice", "choice": "billing", "confidence": 0.95, "probabilities": {"billing": 0.95, "other": 0.05}}
    render_decision_result("dept", ans_choice, q_def_choice)

    # 2. Score test with None checks
    q_def_score = {"type": "score", "instructions": "Urgency", "criteria": ["low", "med", "high"]}
    ans_score = {"type": "score", "score": 1.84, "confidence": 0.74, "probabilities": {"0": 0.05, "1": 0.06, "2": 0.89}, "legend": {"0": "low", "1": "med", "2": "high"}}
    render_decision_result("urgency", ans_score, q_def_score)

    # 3. Noul test (TypeSafe format with float probability)
    q_def_noul = {"type": "noul", "instructions": "Escalate?"}
    ans_noul = {"type": "noul", "noul": 0.1091}
    render_decision_result("escalate", ans_noul, q_def_noul)


