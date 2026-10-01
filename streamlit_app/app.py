"""Nirnaya System-One Decision Studio & Interactive Showcase.

Streamlit frontend service compatible with TypeSafe AI Jev REST API.
"""
from __future__ import annotations

import json
import time
import sys
from pathlib import Path
from typing import Any, Dict
import pandas as pd
import streamlit as st

# Ensure streamlit_app folder and parent directory are on sys.path
_current_dir = Path(__file__).resolve().parent
_parent_dir = _current_dir.parent
_repo_root = _parent_dir.parent
for _p in (_current_dir, _parent_dir, _repo_root):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from .api_client import NirnayaClient
    from .auth import is_authenticated, logout, render_login_page
    from .components import render_backend_status, render_decision_result
    from .config import settings
    from .styles import CUSTOM_CSS
    from .templates import SHOWCASE_TEMPLATES
except (ImportError, ValueError):
    from api_client import NirnayaClient
    from auth import is_authenticated, logout, render_login_page
    from components import render_backend_status, render_decision_result
    from config import settings
    from styles import CUSTOM_CSS
    from templates import SHOWCASE_TEMPLATES



def main() -> None:
    # 1. Page Configuration
    st.set_page_config(
        page_title=settings.app_title,
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # 2. Inject Modern Design System CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # 3. Authentication Check
    if not is_authenticated():
        render_login_page()
        return

    # 4. Session State Initialization
    if "api_url" not in st.session_state:
        st.session_state["api_url"] = settings.api_url
    if "api_key" not in st.session_state:
        st.session_state["api_key"] = settings.api_key
    if "timeout_sec" not in st.session_state:
        st.session_state["timeout_sec"] = settings.api_timeout_seconds
    if "last_result" not in st.session_state:
        st.session_state["last_result"] = None
    if "selected_preset" not in st.session_state:
        st.session_state["selected_preset"] = "Support Ticket & Churn Triage"
    if "current_state" not in st.session_state:
        st.session_state["current_state"] = SHOWCASE_TEMPLATES["Support Ticket & Churn Triage"]["state"]
    if "current_questions" not in st.session_state:
        st.session_state["current_questions"] = SHOWCASE_TEMPLATES["Support Ticket & Churn Triage"]["questions"]

    # Initialize API Client
    client = NirnayaClient(
        base_url=st.session_state["api_url"],
        api_key=st.session_state["api_key"],
        timeout_seconds=st.session_state["timeout_sec"],
        connect_timeout_seconds=settings.connect_timeout_seconds,
    )

    # 5. Sidebar Controls & Real-Time Health Probe
    with st.sidebar:
        st.markdown("### ⚡ Nirnaya Decision Server")

        # Health check
        health_status = client.check_health()
        render_backend_status(health_status, st.session_state["api_url"], st.session_state["timeout_sec"])

        if st.button("🔄 Refresh Connection", use_container_width=True):
            st.rerun()

        st.markdown("---")
        st.markdown("#### ⚙️ Service Connection")

        new_api_url = st.text_input(
            "Backend API URL",
            value=st.session_state["api_url"],
            help="URL of the Nirnaya FastAPI microservice",
        )
        if new_api_url != st.session_state["api_url"]:
            st.session_state["api_url"] = new_api_url
            st.rerun()

        new_api_key = st.text_input(
            "API Key (Bearer)",
            value=st.session_state["api_key"],
            type="password",
            help="Authorization API key for Nirnaya Server",
        )
        if new_api_key != st.session_state["api_key"]:
            st.session_state["api_key"] = new_api_key
            st.rerun()

        # Configurable Timeout Display & Adjuster
        st.markdown("#### ⏱️ Long-Running CPU Timeout")
        custom_timeout = st.number_input(
            "Timeout (seconds)",
            min_value=10,
            max_value=3600,
            value=int(st.session_state["timeout_sec"]),
            step=30,
            help="Configured in .env.streamlit. Large models on CPU require 60-300s to avoid dropping connections.",
        )
        if custom_timeout != st.session_state["timeout_sec"]:
            st.session_state["timeout_sec"] = float(custom_timeout)

        # Model Selector
        st.markdown("#### 📦 Model Engine")
        available_models = ["qwen3-0.6b-gguf", "qwen3.5-4b-gguf", "bonsai-27b", "qwen3-4b"]
        hardware_info = {}

        if health_status.is_connected:
            models_meta = client.list_models()
            if "models" in models_meta and models_meta["models"]:
                raw_models = models_meta["models"]
                if isinstance(raw_models, list):
                    extracted = [m.get("alias") for m in raw_models if isinstance(m, dict) and "alias" in m]
                    if extracted:
                        available_models = extracted
                elif isinstance(raw_models, dict):
                    available_models = list(raw_models.keys())
            hardware_info = models_meta.get("hardware", {})

        # Default to the backend's configured default model if present
        def_idx = 0
        if health_status.default_model in available_models:
            def_idx = available_models.index(health_status.default_model)

        selected_model = st.selectbox(
            "Target Model",
            options=available_models,
            index=def_idx,
            help="Model alias defined in models_config.json",
        )


        if hardware_info:
            device = hardware_info.get("preferred_device", "cpu")
            st.caption(f"Hardware: **{device.upper()}** (Threads: {hardware_info.get('cpu_threads', 'N/A')})")

        st.markdown("---")
        st.markdown(f"👤 Logged in as: **{st.session_state.get('username', 'admin')}**")
        if st.button("🚪 Sign Out", use_container_width=True):
            logout()

    # 6. Main Header Banner
    st.markdown(
        """
        <div class="nirnaya-header-card">
            <div class="nirnaya-title">⚡ Nirnaya System-One Studio</div>
            <div class="nirnaya-subtitle">
                Zero-generation, single next-token probabilistic decision engine strictly compatible with the TypeSafe AI Jev contract.
            </div>
            <div class="nirnaya-pills">
                <span class="pill-badge">⚡ Zero Generative Loops</span>
                <span class="pill-badge">🎯 Single Next-Token Readout</span>
                <span class="pill-badge">🔄 Permutation Invariant Choice</span>
                <span class="pill-badge">📏 1D Center-of-Gravity Continuous Rubric</span>
                <span class="pill-badge">🛡️ TypeSafe AI Jev Parity</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 7. Navigation Tabs
    tab_playground, tab_telemetry, tab_docs = st.tabs([
        "🚀 Decision Playground",
        "📊 Live Audit & Telemetry",
        "📖 System-One Architecture",
    ])

    # =========================================================================
    # TAB 1: Decision Playground
    # =========================================================================
    with tab_playground:
        col_preset, col_actions = st.columns([3, 1])
        with col_preset:
            preset_options = list(SHOWCASE_TEMPLATES.keys())
            chosen_preset = st.selectbox(
                "📂 Load Showcase Preset Scenario",
                options=preset_options,
                index=preset_options.index(st.session_state["selected_preset"])
                if st.session_state["selected_preset"] in preset_options
                else 0,
            )
            if chosen_preset != st.session_state["selected_preset"]:
                st.session_state["selected_preset"] = chosen_preset
                st.session_state["current_state"] = SHOWCASE_TEMPLATES[chosen_preset]["state"]
                st.session_state["current_questions"] = SHOWCASE_TEMPLATES[chosen_preset]["questions"]
                st.rerun()

        with col_actions:
            st.write("")
            st.write("")
            if st.button("🧹 Reset to Default", use_container_width=True):
                st.session_state["current_state"] = SHOWCASE_TEMPLATES[chosen_preset]["state"]
                st.session_state["current_questions"] = SHOWCASE_TEMPLATES[chosen_preset]["questions"]
                st.session_state["last_result"] = None
                st.rerun()

        st.caption(f"ℹ️ {SHOWCASE_TEMPLATES[chosen_preset]['description']}")

        # Input Layout: Left Column = State & Questions, Right Column = Trigger & Output
        left_col, right_col = st.columns([1, 1], gap="medium")

        with left_col:
            st.markdown("#### 1. Input Context / State")
            state_val = st.session_state["current_state"]
            state_str = json.dumps(state_val, indent=2) if isinstance(state_val, dict) else str(state_val)

            edited_state_str = st.text_area(
                "State Payload (JSON or Plain Text)",
                value=state_str,
                height=180,
                help="The document, transaction, or customer interaction being evaluated.",
            )

            # Try to parse state as JSON, otherwise keep as raw string
            try:
                parsed_state = json.loads(edited_state_str)
            except Exception:
                parsed_state = edited_state_str

            st.markdown("#### 2. Typed Questions Schema")
            q_mode = st.radio(
                "Question Editor Mode",
                ["Visual Overview", "Raw JSON Schema"],
                horizontal=True,
            )

            if q_mode == "Raw JSON Schema":
                questions_str = json.dumps(st.session_state["current_questions"], indent=2)
                edited_q_str = st.text_area(
                    "Questions Definition (TypeSafe Jev Schema)",
                    value=questions_str,
                    height=280,
                )
                try:
                    active_questions = json.loads(edited_q_str)
                    st.session_state["current_questions"] = active_questions
                except Exception as err:
                    st.error(f"Invalid JSON: {err}")
                    active_questions = st.session_state["current_questions"]
            else:
                # Visual summary of configured questions
                active_questions = st.session_state["current_questions"]
                for qk, qv in list(active_questions.items()):
                    q_type = qv.get("type", "choice")
                    with st.expander(f"Question: `{qk}` ({q_type.upper()})", expanded=True):
                        st.markdown(f"**Instructions:** {qv.get('instructions', '')}")
                        if q_type == "choice":
                            crit = qv.get("criteria") or qv.get("options", {})
                            st.json(crit, expanded=False)
                        elif q_type == "score":
                            levels = qv.get("criteria") or qv.get("levels", [])
                            st.write("**Rubric Levels:**", levels)
                        elif q_type == "noul":
                            st.caption("Binary decision: evaluates True vs False probabilities.")

        with right_col:
            st.markdown("#### 3. Execution & Results")
            execute_button = st.button(
                f"⚡ Run System-One Decision ({selected_model})",
                type="primary",
                use_container_width=True,
            )

            if execute_button:
                if not health_status.is_connected:
                    st.error("Cannot execute decision: Nirnaya server is offline or unreachable.")
                else:
                    with st.spinner(f"Running System-One readout on {selected_model}... (Safe CPU timeout: {st.session_state['timeout_sec']}s)"):
                        try:
                            result = client.execute_system_one(
                                state=parsed_state,
                                questions=active_questions,
                                model=selected_model,
                            )
                            st.session_state["last_result"] = result
                            st.success("Decision completed successfully!")
                        except Exception as e:
                            st.error(f"Execution Error: {e}")

            # Display Results if Available
            if st.session_state["last_result"]:
                res = st.session_state["last_result"]
                answers = res.get("answers", {})
                usage = res.get("usage", {})
                roundtrip = res.get("_client_roundtrip_ms", "N/A")

                # Telemetry Metric Row
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.markdown(
                        f"""
                        <div class="metric-box">
                            <div class="metric-label">Roundtrip Latency</div>
                            <div class="metric-val" style="color: #38bdf8;">{roundtrip} ms</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with m2:
                    st.markdown(
                        f"""
                        <div class="metric-box">
                            <div class="metric-label">Decisions Evaluated</div>
                            <div class="metric-val" style="color: #34d399;">{len(answers)}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with m3:
                    st.markdown(
                        f"""
                        <div class="metric-box">
                            <div class="metric-label">Model Engine</div>
                            <div class="metric-val" style="font-size: 1.05rem; color: #c4b5fd;">{res.get('model', selected_model)}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
                st.markdown("##### 🎯 Typed System-One Outcomes")

                for q_id, ans in answers.items():
                    q_def = active_questions.get(q_id, {})
                    render_decision_result(q_id, ans, q_def)

                # Raw Jev Contract Inspection
                with st.expander("📄 View TypeSafe AI Jev JSON Payload", expanded=False):
                    st.caption("Exact output structure compliant with TypeSafe AI Jev cloud contract:")
                    st.json(res)
                    st.download_button(
                        label="💾 Download Jev JSON",
                        data=json.dumps(res, indent=2),
                        file_name="nirnaya_jev_decision.json",
                        mime="application/json",
                    )

    # =========================================================================
    # TAB 2: Live Audit & Telemetry
    # =========================================================================
    with tab_telemetry:
        st.markdown("### 📊 Server Audit Logs & Performance Telemetry")
        st.caption("Persisted in SQLite database (`nirnaya.db`) with Write-Ahead Logging (WAL).")

        logs_col1, logs_col2 = st.columns([3, 1])
        with logs_col2:
            log_limit = st.slider("Max Rows", min_value=10, max_value=200, value=30, step=10)
            if st.button("🔄 Refresh Logs", use_container_width=True):
                st.rerun()

        if health_status.is_connected:
            logs = client.get_audit_logs(limit=log_limit)
            if logs:
                df_logs = pd.DataFrame(logs)
                # Reorder and format columns
                cols_to_show = [c for c in ["request_id", "timestamp", "model", "duration_ms", "status_code", "input_tokens", "output_tokens", "error_message"] if c in df_logs.columns]
                st.dataframe(
                    df_logs[cols_to_show],
                    use_container_width=True,
                    height=360,
                )

                # Quick Latency Metrics
                if "duration_ms" in df_logs.columns and len(df_logs) > 0:
                    durations = df_logs["duration_ms"].dropna()
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Mean Latency", f"{durations.mean():.1f} ms")
                    c2.metric("Median (P50)", f"{durations.median():.1f} ms")
                    c3.metric("P95 Latency", f"{durations.quantile(0.95):.1f} ms")
            else:
                st.info("No audit logs recorded yet. Run a decision in the Playground to see telemetry.")
        else:
            st.warning("Cannot fetch audit logs: Nirnaya server is offline.")

    # =========================================================================
    # TAB 3: System-One Architecture
    # =========================================================================
    with tab_docs:
        st.markdown("### 🏛️ The Nirnaya System-One Architecture")
        st.markdown(
            """
            Nirnaya is an open-source, local-first **System-One typed probabilistic decision engine** designed to replicate and match commercial engines like **TypeSafe AI Jev** without relying on external cloud APIs or closed weights.

            ---

            #### 1. Why Zero Generation Beats Generative Decoding
            Traditional LLM workflows invoke autoregressive token generation:
            - Generates dozens or hundreds of conversational tokens (`"Based on the analysis, I believe..."`).
            - Requires regex parsing, JSON mode, or constrained decoders (`json-mode`, `instructor`).
            - Suffers from hallucinations, non-deterministic phrasing, and high latency (1–5 seconds).

            **Nirnaya completely eliminates autoregressive generation:**
            - Strictly evaluates decisions at the **first token position** after the prompt prefix.
            - Decodes the model's raw probability mass directly into structured typed answers.
            - Latency drops to **150–600 ms** per case.

            ---

            #### 2. The Three Typed Decision Primitives
            | Type | Mathematical Formulation | Output |
            | :--- | :--- | :--- |
            | **`choice`** | Softmax over normalized symbol logits + cyclic permutation pooling | Discrete Category + Confidence |
            | **`score`** | 1D Center of Gravity: $\\text{value} = \\sum_{k=0}^{K-1} k \\cdot p_k$ | Continuous Floating Point (0.00 – K-1.00) |
            | **`noul`** | Calibrated binary probability $P(\\text{true})$ vs $P(\\text{false})$ | Boolean (`true`/`false`) + Confidence |

            ---

            #### 3. 100% TypeSafe AI Jev Contract Parity
            Nirnaya outputs exact Jev-compatible JSON:
            ```json
            {
              "question_id": "urgency",
              "decision_type": "score",
              "predicted_value": 1.84,
              "confidence": 0.86,
              "distribution": {
                "0": 0.04,
                "1": 0.08,
                "2": 0.88
              }
            }
            ```
            """
        )


if __name__ == "__main__":
    main()
