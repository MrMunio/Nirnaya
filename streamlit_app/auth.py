"""Authentication and session management for Nirnaya Streamlit UI."""
from __future__ import annotations

import streamlit as st

try:
    from .config import settings
except (ImportError, ValueError):
    from config import settings



def init_auth_state() -> None:
    """Initializes session state keys for user authentication."""
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if "username" not in st.session_state:
        st.session_state["username"] = ""


def is_authenticated() -> bool:
    """Returns True if the current user session is authenticated."""
    init_auth_state()
    return st.session_state.get("authenticated", False)


def login(username: str, password: str) -> bool:
    """Verifies credentials and sets session state."""
    if settings.verify_credentials(username.strip(), password):
        st.session_state["authenticated"] = True
        st.session_state["username"] = username.strip()
        return True
    return False


def logout() -> None:
    """Clears current session authentication."""
    st.session_state["authenticated"] = False
    st.session_state["username"] = ""
    st.rerun()


def render_login_page() -> None:
    """Renders the sleek, centered login screen."""
    st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            """
            <div style="text-align: center; margin-bottom: 2rem;">
                <h1 style="font-size: 2.2rem; font-weight: 700; background: linear-gradient(90deg, #818cf8, #34d399); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.5rem;">
                    Nirnaya System-One
                </h1>
                <p style="color: #94a3b8; font-size: 0.95rem;">
                    Typed Probabilistic Decision Engine & Showcase UI
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("login_form", clear_on_submit=False):
            st.markdown("### 🔐 Sign In")
            username_input = st.text_input(
                "Username",
                value=settings.admin_username,
                placeholder="Enter username (e.g. admin)",
                key="login_user",
            )
            password_input = st.text_input(
                "Password",
                type="password",
                value="",
                placeholder="Enter password",
                key="login_pass",
            )
            submit_button = st.form_submit_button("Sign In to Decision Studio", use_container_width=True)

            if submit_button:
                if login(username_input, password_input):
                    st.success(f"Welcome, {username_input}! Authenticated successfully.")
                    st.rerun()
                else:
                    st.error("Invalid username or password. Check credentials in .env.streamlit.")

        st.info(
            f"💡 **Demo Quick Start:** Default credentials are `{settings.admin_username}` / `{settings.admin_password}` "
            f"(configurable via `.env`)."
        )

