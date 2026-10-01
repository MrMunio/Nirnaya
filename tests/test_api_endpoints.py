"""API integration tests using FastAPI TestClient."""
import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from nirnaya_server.app.config import settings
from nirnaya_server.app.db import init_db
from nirnaya_server.app.main import app
from nirnaya_server.app.models.api_keys import KeyManager
from nirnaya_server.app.core.schema import Answer, Prediction


@pytest.fixture(autouse=True)
def setup_isolated_env(tmp_path):
    """Sets up an isolated test database with fresh admin and client keys."""
    test_db = tmp_path / "test_api.db"
    orig_db = settings.DB_PATH
    orig_auth = settings.REQUIRE_AUTH
    settings.DB_PATH = str(test_db)
    settings.REQUIRE_AUTH = True
    init_db()

    # Seed Admin Key and Client Key
    admin_key = "nir_live_admin_secret_test_123"
    client_key = "nir_live_client_secret_test_456"
    KeyManager.ensure_admin_key(admin_key, name="Admin Key")
    KeyManager.create_key(name="Client Key", role="client", custom_key=client_key)

    yield {
        "admin_key": admin_key,
        "client_key": client_key,
    }

    settings.DB_PATH = orig_db
    settings.REQUIRE_AUTH = orig_auth


def test_health_and_ready_endpoints():
    client = TestClient(app)
    # Liveness probe
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

    # Readiness probe
    resp = client.get("/ready")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ready"
    assert resp.json()["database"] == "connected"


def test_auth_protection_on_systemone():
    client = TestClient(app)
    sample_payload = {
        "state": "The user reported an issue with their invoice.",
        "questions": {
            "escalate": {"type": "noul", "instructions": "Should escalate?"}
        }
    }

    # 1. Missing authentication header -> 401
    resp = client.post("/v1/systemone", json=sample_payload)
    assert resp.status_code == 401
    assert "Missing authentication" in resp.json()["detail"]

    # 2. Invalid Bearer token -> 401
    resp = client.post(
        "/v1/systemone",
        json=sample_payload,
        headers={"Authorization": "Bearer nir_live_bad_token"}
    )
    assert resp.status_code == 401
    assert "Invalid or revoked API Key" in resp.json()["detail"]


def test_systemone_with_bearer_and_x_api_key(setup_isolated_env):
    client = TestClient(app)
    client_key = setup_isolated_env["client_key"]

    mock_prediction = Prediction(
        model="qwen3-0.6b-gguf",
        answers={
            "escalate": Answer(
                qid="escalate",
                qtype="noul",
                value=0.91,
                labels=["yes", "no"],
                probs=[0.91, 0.09],
                raw_probs=[0.91, 0.09],
                logp=[-0.09, -2.4],
                confidence=0.91,
                margin=0.82,
                entropy=0.25,
                valid_mass=0.95,
                agreement=1.0,
            )
        },
        tokens={"static": 80, "state": 30, "q_escalate": 15},
    )

    with patch("nirnaya_server.app.services.engine_service.engine_service.predict", return_value=mock_prediction):
        # 1. Test using Bearer Token
        resp_bearer = client.post(
            "/v1/systemone",
            json={"state": "Test state", "questions": {"escalate": {"type": "noul", "instructions": "Escalate?"}}},
            headers={"Authorization": f"Bearer {client_key}"},
        )
        assert resp_bearer.status_code == 200
        data = resp_bearer.json()
        assert data["model"] == "qwen3-0.6b-gguf"
        assert data["answers"]["escalate"]["type"] == "noul"
        assert data["answers"]["escalate"]["noul"] == 0.91
        assert data["usage"]["input_tokens"] == 125

        # 2. Test using X-API-Key header
        resp_xkey = client.post(
            "/v1/systemone",
            json={"state": "Test state", "questions": {"escalate": {"type": "noul", "instructions": "Escalate?"}}},
            headers={"X-API-Key": client_key},
        )
        assert resp_xkey.status_code == 200
        assert resp_xkey.json()["answers"]["escalate"]["noul"] == 0.91


def test_admin_endpoints_role_enforcement(setup_isolated_env):
    client = TestClient(app)
    admin_key = setup_isolated_env["admin_key"]
    client_key = setup_isolated_env["client_key"]

    # Client role trying to access admin endpoint -> 403 Forbidden
    resp = client.get("/v1/admin/keys", headers={"Authorization": f"Bearer {client_key}"})
    assert resp.status_code == 403

    # Admin role -> 200 OK
    resp = client.get("/v1/admin/keys", headers={"Authorization": f"Bearer {admin_key}"})
    assert resp.status_code == 200
    keys = resp.json()
    assert len(keys) >= 2

    # Create new key via admin
    resp = client.post(
        "/v1/admin/keys",
        headers={"Authorization": f"Bearer {admin_key}"},
        json={"name": "Worker Node 1", "role": "client"},
    )
    assert resp.status_code == 200
    new_key_data = resp.json()
    assert "raw_key" in new_key_data
    assert new_key_data["name"] == "Worker Node 1"
