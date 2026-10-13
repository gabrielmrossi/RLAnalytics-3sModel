"""
app/tabs — Abas decompostas da interface do Streamlit.
"""

from .higher_vs_lower import tab_higher_vs_lower
from .live_replay import tab_live_replay
from .models import tab_models
from .tactical_errors import tab_errors

__all__ = ["tab_higher_vs_lower", "tab_live_replay", "tab_models", "tab_errors"]
