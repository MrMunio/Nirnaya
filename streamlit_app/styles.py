"""Custom modern CSS styles and theme enhancements for Nirnaya Streamlit UI."""

CUSTOM_CSS = """
<style>
/* Main Container & Modern Typography */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

code, pre, .stCodeBlock {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Header & Banner Styles */
.nirnaya-header-card {
    background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 50%, #022c22 100%);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 12px;
    padding: 1.5rem 1.8rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
    color: #ffffff;
}

.nirnaya-title {
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin-bottom: 0.3rem;
    background: linear-gradient(90deg, #ffffff, #a5b4fc, #6ee7b7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.nirnaya-subtitle {
    font-size: 0.95rem;
    color: #cbd5e1;
    line-height: 1.5;
}

.nirnaya-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 12px;
}

.pill-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 9999px;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.15);
    color: #e2e8f0;
}

/* Status Indicator Dots */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 14px;
    border-radius: 9999px;
    font-size: 0.85rem;
    font-weight: 600;
}

.status-online {
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid #10b981;
    color: #34d399;
}

.status-offline {
    background: rgba(239, 68, 68, 0.15);
    border: 1px solid #ef4444;
    color: #f87171;
}

.dot-pulse {
    width: 9px;
    height: 9px;
    border-radius: 50%;
    display: inline-block;
}

.dot-online {
    background-color: #10b981;
    box-shadow: 0 0 8px #10b981;
}

.dot-offline {
    background-color: #ef4444;
    box-shadow: 0 0 8px #ef4444;
}

/* Card Visualizer for System-One Answers */
.decision-card {
    background: rgba(30, 41, 59, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 10px;
    padding: 1.25rem;
    margin-bottom: 1rem;
    transition: transform 0.15s ease, border-color 0.15s ease;
}

.decision-card:hover {
    border-color: rgba(165, 180, 252, 0.4);
}

.decision-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.75rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
}

.type-tag-choice {
    background: rgba(139, 92, 246, 0.2);
    border: 1px solid #8b5cf6;
    color: #c4b5fd;
}

.type-tag-score {
    background: rgba(16, 185, 129, 0.2);
    border: 1px solid #10b981;
    color: #6ee7b7;
}

.type-tag-noul {
    background: rgba(14, 165, 233, 0.2);
    border: 1px solid #0ea5e9;
    color: #7dd3fc;
}

.val-badge {
    font-size: 1.2rem;
    font-weight: 700;
    color: #ffffff;
}

/* Metric Display Boxes */
.metric-box {
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 8px;
    padding: 0.9rem;
    text-align: center;
}

.metric-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    color: #94a3b8;
    letter-spacing: 0.05em;
    margin-bottom: 4px;
}

.metric-val {
    font-size: 1.35rem;
    font-weight: 700;
    color: #f8fafc;
}

/* Login Card Center Container */
.login-container {
    max-width: 440px;
    margin: 3rem auto;
    padding: 2.2rem;
    background: rgba(30, 41, 59, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    box-shadow: 0 16px 36px rgba(0, 0, 0, 0.35);
}
</style>
"""
