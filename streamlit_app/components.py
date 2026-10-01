"""UI visual components for Nirnaya decision outputs and status indicators."""
from __future__ import annotations

from typing import Any, Dict
import pandas as pd
import streamlit as st

try:
    from .api_client import HealthStatus
except (ImportError, ValueError):
    from api_client import HealthStatus



def render_backend_status(status: HealthStatus, api_url: str, timeout_sec: float) -> None:
    """Renders connection status pill and server info in sidebar or header."""
    if status.is_connected and status.is_ready:
        st.markdown(
            f"""
            <div class="status-pill status-online" style="margin-bottom: 8px;">
                <span class="dot-pulse dot-online"></span>
                <span>Connected & Ready ({status.latency_ms:.0f}ms)</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(f"📍 **Backend:** `{api_url}`")
        st.caption(f"📦 **Default Model:** `{status.default_model}`")
        st.caption(f"⏱️ **CPU Safe Timeout:** `{timeout_sec:.0f}s` configured")
    elif status.is_connected and not status.is_ready:
        st.markdown(
            f"""
            <div class="status-pill" style="background: rgba(234, 179, 8, 0.15); border: 1px solid #eab308; color: #fde047; margin-bottom: 8px;">
                <span class="dot-pulse" style="background-color: #eab308; box-shadow: 0 0 8px #eab308;"></span>
                <span>Connecting / Warming ({status.latency_ms:.0f}ms)</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(f"📍 **Backend:** `{api_url}` ({status.details})")
    else:
        st.markdown(
            f"""
            <div class="status-pill status-offline" style="margin-bottom: 8px;">
                <span class="dot-pulse dot-offline"></span>
                <span>Backend Offline</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.error(f"⚠️ Cannot reach Nirnaya at `{api_url}`. {status.details}")
        st.caption("Start the server using: `python -m uvicorn app.main:app --port 8000`")


def render_decision_result(q_id: str, ans: Dict[str, Any], question_def: Dict[str, Any]) -> None:
    """Renders visual decision card for a single typed question result.
    
    Robustly supports both TypeSafe AI Jev contract fields:
      - choice: {"choice": "...", "confidence": ..., "probabilities": {...}}
      - score:  {"score": 1.84, "confidence": ..., "probabilities": {...}, "legend": {...}}
      - noul:   {"noul": 0.1091} (P(true) probability float)
    and Nirnaya native fallback fields (predicted_value, distribution).
    """
    q_type = ans.get("type") or ans.get("decision_type") or question_def.get("type", "choice")
    instructions = question_def.get("instructions", "")
    legend: Dict[str, str] = {}

    if q_type == "choice":
        tag_class = "type-tag-choice"
        type_icon = "🏷️"
        pred_val = ans.get("choice")
        if pred_val is None:
            pred_val = ans.get("predicted_value", "N/A")
        try:
            conf = float(ans.get("confidence", 0.0))
        except (ValueError, TypeError):
            conf = 0.0
        dist = ans.get("probabilities") or ans.get("distribution") or {}

    elif q_type == "score":
        tag_class = "type-tag-score"
        type_icon = "📏"
        raw_score = ans.get("score")
        if raw_score is None:
            raw_score = ans.get("predicted_value", 0.0)
        try:
            pred_val = float(raw_score)
        except (ValueError, TypeError):
            pred_val = 0.0
        try:
            conf = float(ans.get("confidence", 0.0))
        except (ValueError, TypeError):
            conf = 0.0
        dist = ans.get("probabilities") or ans.get("distribution") or {}
        legend = ans.get("legend") or {}
        # Fallback legend from question_def if not in response
        if not legend:
            crit = question_def.get("criteria") or question_def.get("levels") or []
            if isinstance(crit, list):
                legend = {str(i): str(desc) for i, desc in enumerate(crit)}

    elif q_type == "noul":
        tag_class = "type-tag-noul"
        type_icon = "⚖️"
        # In TypeSafe Jev format: {"type": "noul", "noul": <float P(true)>}
        if "noul" in ans and ans["noul"] is not None:
            try:
                p_true = float(ans["noul"])
            except (ValueError, TypeError):
                p_true = 0.5
            is_true = p_true >= 0.5
            pred_val = is_true
            conf = p_true if is_true else (1.0 - p_true)
            dist = {"false": round(1.0 - p_true, 4), "true": round(p_true, 4)}
        else:
            raw_val = ans.get("predicted_value")
            is_true = bool(raw_val)
            pred_val = is_true
            try:
                conf = float(ans.get("confidence", 0.5))
            except (ValueError, TypeError):
                conf = 0.5
            raw_dist = ans.get("probabilities") or ans.get("distribution")
            if raw_dist:
                dist = raw_dist
            else:
                dist = {
                    "false": round(1.0 - conf if is_true else conf, 4),
                    "true": round(conf if is_true else 1.0 - conf, 4),
                }
    else:
        tag_class = ""
        type_icon = "❓"
        pred_val = ans.get("predicted_value", str(ans))
        try:
            conf = float(ans.get("confidence", 0.0))
        except (ValueError, TypeError):
            conf = 0.0
        dist = ans.get("probabilities") or ans.get("distribution") or {}

    st.markdown(
        f"""
        <div class="decision-card">
            <div class="decision-card-header">
                <div>
                    <span class="pill-badge {tag_class}">{type_icon} {q_type.upper()}</span>
                    <strong style="margin-left: 8px; font-size: 1.1rem; color: #f8fafc;">{q_id}</strong>
                </div>
                <div>
                    <span style="font-size: 0.85rem; color: #94a3b8;">Confidence:</span>
                    <strong style="font-size: 1.05rem; color: #38bdf8; margin-left: 4px;">{conf * 100:.1f}%</strong>
                </div>
            </div>
            <div style="font-size: 0.9rem; color: #94a3b8; margin-bottom: 12px;">{instructions}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Main answer presentation
    val_col, chart_col = st.columns([1, 2])

    with val_col:
        st.markdown("**Predicted Value:**")
        if q_type == "noul":
            is_true = bool(pred_val)
            badge_color = "#10b981" if is_true else "#ef4444"
            st.markdown(
                f"""
                <div style="padding: 12px; background: rgba(0,0,0,0.3); border-radius: 8px; border-left: 4px solid {badge_color};">
                    <span style="font-size: 1.6rem; font-weight: 700; color: {badge_color};">
                        {"✅ TRUE" if is_true else "❌ FALSE"}
                    </span>
                    <p style="margin: 4px 0 0; font-size: 0.8rem; color: #94a3b8;">
                        Confidence: <strong>{conf * 100:.1f}%</strong>
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif q_type == "score":
            score_float = float(pred_val) if isinstance(pred_val, (int, float)) else 0.0
            st.markdown(
                f"""
                <div style="padding: 12px; background: rgba(0,0,0,0.3); border-radius: 8px; border-left: 4px solid #10b981;">
                    <span style="font-size: 1.8rem; font-weight: 700; color: #34d399;">
                        {score_float:.2f}
                    </span>
                    <p style="margin: 4px 0 0; font-size: 0.8rem; color: #94a3b8;">
                        1D Center-of-Gravity (Expected Level)
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:  # choice
            st.markdown(
                f"""
                <div style="padding: 12px; background: rgba(0,0,0,0.3); border-radius: 8px; border-left: 4px solid #8b5cf6;">
                    <span style="font-size: 1.5rem; font-weight: 700; color: #c4b5fd;">
                        {pred_val}
                    </span>
                    <p style="margin: 4px 0 0; font-size: 0.8rem; color: #94a3b8;">Permutation Invariant Choice</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with chart_col:
        st.markdown("**Probability Distribution:**")
        if dist:
            # Map keys with legend if available (e.g. "0" -> "0: not urgent")
            display_keys = []
            vals = []
            for k, v in dist.items():
                lbl = str(k)
                if legend and str(k) in legend:
                    lbl = f"{k}: {legend[str(k)]}"
                display_keys.append(lbl)
                vals.append(float(v))

            df = pd.DataFrame({"Outcome": display_keys, "Probability": vals})
            df = df.sort_values(by="Probability", ascending=True)
            st.bar_chart(df.set_index("Outcome"), horizontal=True, color="#818cf8")
        else:
            st.caption("No distribution data returned.")

    st.markdown("<hr style='border: none; border-top: 1px solid rgba(255,255,255,0.06); margin: 1.2rem 0;' />", unsafe_allow_html=True)

