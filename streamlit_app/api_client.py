"""Robust HTTP client for interacting with the Nirnaya Decision Server."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import httpx

try:
    from .config import settings
except (ImportError, ValueError):
    from config import settings



@dataclass
class HealthStatus:
    """Backend connection and health status representation."""
    is_connected: bool
    is_ready: bool
    status_code: int
    latency_ms: float
    service_name: str = "nirnaya-server"
    default_model: str = "unknown"
    details: str = ""


class NirnayaClient:
    """Client for Nirnaya REST API with configurable timeout protection."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        connect_timeout_seconds: Optional[float] = None,
    ) -> None:
        self.base_url = (base_url or settings.api_url).rstrip("/")
        self.api_key = api_key or settings.api_key
        self.timeout_seconds = timeout_seconds if timeout_seconds is not None else settings.api_timeout_seconds
        self.connect_timeout = connect_timeout_seconds if connect_timeout_seconds is not None else settings.connect_timeout_seconds

    def _get_headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            headers["X-API-Key"] = self.api_key
        return headers

    def _get_client(self, custom_timeout: Optional[float] = None) -> httpx.Client:
        t = custom_timeout or self.timeout_seconds
        timeout_config = httpx.Timeout(
            timeout=t,
            connect=self.connect_timeout,
            read=t,
            write=30.0,
        )
        return httpx.Client(base_url=self.base_url, timeout=timeout_config, headers=self._get_headers())

    def check_health(self) -> HealthStatus:
        """Checks backend /health and /ready endpoints to assess service status."""
        t0 = time.perf_counter()
        try:
            # Use short timeout (5s) for health check so UI does not stall
            with self._get_client(custom_timeout=5.0) as client:
                resp = client.get("/health")
                latency_ms = (time.perf_counter() - t0) * 1000.0

                if resp.status_code != 200:
                    return HealthStatus(
                        is_connected=False,
                        is_ready=False,
                        status_code=resp.status_code,
                        latency_ms=latency_ms,
                        details=f"Unexpected status code {resp.status_code}",
                    )

                # Try readiness check
                try:
                    ready_resp = client.get("/ready")
                    is_ready = ready_resp.status_code == 200
                    ready_data = ready_resp.json() if is_ready else {}
                    default_model = ready_data.get("default_model", "qwen3-0.6b-gguf")
                except Exception:
                    is_ready = True
                    default_model = "configured"

                return HealthStatus(
                    is_connected=True,
                    is_ready=is_ready,
                    status_code=resp.status_code,
                    latency_ms=latency_ms,
                    service_name=resp.json().get("service", "nirnaya-server"),
                    default_model=default_model,
                    details="Operational",
                )
        except httpx.ConnectError:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return HealthStatus(
                is_connected=False,
                is_ready=False,
                status_code=0,
                latency_ms=latency_ms,
                details="Connection refused. Is Nirnaya server running on this port?",
            )
        except httpx.TimeoutException:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return HealthStatus(
                is_connected=False,
                is_ready=False,
                status_code=408,
                latency_ms=latency_ms,
                details="Connection timed out while probing health.",
            )
        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return HealthStatus(
                is_connected=False,
                is_ready=False,
                status_code=500,
                latency_ms=latency_ms,
                details=str(e),
            )

    def list_models(self) -> Dict[str, Any]:
        """Fetches registered models and hardware status from GET /v1/models."""
        try:
            with self._get_client(custom_timeout=10.0) as client:
                resp = client.get("/v1/models")
                if resp.status_code == 200:
                    return resp.json()
                return {"models": {}, "hardware": {"error": f"Status {resp.status_code}: {resp.text}"}}
        except Exception as e:
            return {"models": {}, "hardware": {"error": str(e)}}

    def execute_system_one(
        self,
        state: Any,
        questions: Dict[str, Any],
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Sends System-One typed decision readout request to POST /v1/systemone.
        
        Uses the extended timeout_seconds configured in .env.streamlit (default 300s).
        """
        payload = {
            "state": state,
            "questions": questions,
        }
        if model:
            payload["model"] = model

        t0 = time.perf_counter()
        try:
            with self._get_client() as client:
                resp = client.post("/v1/systemone", json=payload)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

                if resp.status_code == 200:
                    data = resp.json()
                    data["_client_roundtrip_ms"] = round(elapsed_ms, 2)
                    return data
                elif resp.status_code == 401:
                    raise PermissionError(f"401 Unauthorized: Invalid API key. ({resp.text})")
                else:
                    raise RuntimeError(f"Server returned HTTP {resp.status_code}: {resp.text}")
        except httpx.TimeoutException as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            raise TimeoutError(
                f"Inference timed out after {self.timeout_seconds}s (elapsed: {elapsed_ms:.1f}ms). "
                f"You can increase NIRNAYA_API_TIMEOUT_SECONDS in .env.streamlit if running large models on CPU."
            ) from exc
        except httpx.ConnectError as exc:
            raise ConnectionError(
                f"Failed to connect to Nirnaya server at {self.base_url}. Ensure the server is started."
            ) from exc

    def get_audit_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieves recent audit logs from GET /v1/admin/logs."""
        try:
            with self._get_client(custom_timeout=15.0) as client:
                resp = client.get("/v1/admin/logs", params={"limit": limit})
                if resp.status_code == 200:
                    return resp.json()
                return []
        except Exception:
            return []
