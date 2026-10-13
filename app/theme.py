"""
theme.py — Estilos, paletas visuais e injeção de CSS customizado no Streamlit.
"""

import sys
from pathlib import Path

# Permite imports de src se app for executado diretamente
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import streamlit as st

from config import (
    BALL_COLOR,
    CLUSTER_PALETTE,
    DARK_BG,
    DARK_PAPER,
    TEAM_COLORS,
    TEAM_FILL_COLORS,
    TEAM_NAMES,
)

_CUSTOM_CSS = """
<style>
    .stApp { background-color: #0e1117; }
    .scoreboard {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 1.5rem;
        padding: 0.6rem 1.5rem;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        border: 1px solid #0f3460;
        margin-bottom: 0.5rem;
    }
    .scoreboard .team-score {
        font-size: 2rem;
        font-weight: 700;
    }
    .scoreboard .blue { color: #3498db; }
    .scoreboard .orange { color: #e67e22; }
    .scoreboard .match-time {
        font-size: 1.1rem;
        color: #a0a0a0;
        font-family: monospace;
        min-width: 60px;
        text-align: center;
    }
    .scoreboard .separator {
        font-size: 1.5rem;
        color: #555;
    }
    .error-card {
        padding: 0.8rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
    }
    .error-alta {
        background: rgba(233, 69, 96, 0.15);
        border-left: 4px solid #e94560;
    }
    .error-media {
        background: rgba(255, 165, 0, 0.15);
        border-left: 4px solid orange;
    }
    .uu-info {
        font-size: 0.75rem;
        color: #666;
        font-style: italic;
    }
</style>
"""


def apply_theme() -> None:
    """Aplica o tema escuro e as classes CSS customizadas na interface do Streamlit."""
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)
