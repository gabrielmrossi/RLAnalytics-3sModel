"""
main.py — Ponto de entrada da aplicação Streamlit (RL Analytics).
Roteador principal de abas com interface desacoplada e modular.
"""

import sys
from pathlib import Path

# Garante acesso à raiz do projeto quando executado via `streamlit run app/main.py`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app.theme import apply_theme
from app.tabs.higher_vs_lower import tab_higher_vs_lower
from app.tabs.live_replay import tab_live_replay
from app.tabs.models import tab_models
from app.tabs.tactical_errors import tab_errors


def main() -> None:
    """Função principal da interface do Streamlit."""
    st.set_page_config(
        page_title="RL Analytics - TCC",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    apply_theme()

    st.title("⚽ Rocket League 3v3 Analytics")
    st.caption("Plataforma de análise de telemetria e detecção de erros táticos para Rocket League.")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🎮 Replay ao Vivo",
        "📊 Higher vs Lower",
        "🧠 Modelos de IA",
        "⚠️ Erros Táticos",
    ])

    with tab1:
        tab_live_replay()

    with tab2:
        tab_higher_vs_lower()

    with tab3:
        tab_models()

    with tab4:
        tab_errors()


if __name__ == "__main__":
    main()
