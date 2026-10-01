"""TypeSafe AI Jev Compatible Decision Endpoint: POST /v1/systemone."""
from __future__ import annotations

import time
from typing import Any, Dict, Optional, Union
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from ...auth import get_current_api_key
from ...models.audit import AuditLogger
from ...services.engine_service import engine_service

router = APIRouter()


class SystemOneRequest(BaseModel):
    state: Union[str, Dict[str, Any]] = Field(
        ...,
        description="The context or document state against which typed questions are evaluated.",
    )
    questions: Dict[str, Dict[str, Any]] = Field(
        ...,
        description="Dictionary of typed questions (choice, score, noul) keyed by question_id.",
    )
    model: Optional[str] = Field(
        None,
        description="Target model identifier or alias. If omitted, uses server default from .env.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "model": "qwen3-0.6b-gguf",
                "state": "Customer has an overdue balance of $450 and their card was declined twice.",
                "questions": {
                    "urgency": {
                        "type": "score",
                        "instructions": "Rate how urgent this issue is from 0 (none) to 4 (critical)",
                        "criteria": ["low", "moderate", "high", "critical", "immediate action"],
                    },
                    "category": {
                        "type": "choice",
                        "instructions": "Select the primary ticket category",
                        "criteria": {
                            "billing": "Invoices, payments, credit card failures",
                            "technical": "Software bugs, system outages",
                            "general": "General questions",
                        },
                    },
                    "should_escalate": {
                        "type": "noul",
                        "instructions": "Should this ticket be escalated to a human supervisor?",
                    },
                },
            }
        }
    }


@router.post("/systemone", summary="System-One Decision Readout (Jev Contract)")
async def execute_system_one(
    payload: SystemOneRequest,
    request: Request,
    current_key: Dict[str, Any] = Depends(get_current_api_key),
) -> Dict[str, Any]:
    """Evaluates one state against multiple typed questions with single next-token readout.

    Output format strictly matches TypeSafe AI Jev JSON API contract.
    """
    req_id = AuditLogger.generate_request_id()
    t0 = time.perf_counter()

    try:
        prediction = engine_service.predict(
            state=payload.state,
            questions=payload.questions,
            model_name=payload.model,
        )
    except Exception as exc:
        duration_ms = (time.perf_counter() - t0) * 1000.0
        AuditLogger.log_request(
            request_id=req_id,
            model=payload.model or "default",
            duration_ms=duration_ms,
            input_tokens=0,
            output_tokens=0,
            status_code=500,
            api_key_id=current_key.get("id"),
            error_message=str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution error: {str(exc)}",
        )

    duration_ms = (time.perf_counter() - t0) * 1000.0
    jev_response = prediction.to_jev()

    # Log telemetry to SQLite
    input_tokens = jev_response.get("usage", {}).get("input_tokens", 0)
    output_tokens = jev_response.get("usage", {}).get("output_tokens", len(jev_response.get("answers", {})))

    AuditLogger.log_request(
        request_id=req_id,
        model=prediction.model,
        duration_ms=duration_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        status_code=200,
        api_key_id=current_key.get("id"),
    )

    return jev_response
