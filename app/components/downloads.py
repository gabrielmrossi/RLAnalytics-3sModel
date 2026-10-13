"""
downloads.py — Componente padronizado para downloads em memória sem I/O síncrono em disco.
"""

from typing import Any

import streamlit as st


def render_download_button(
    label: str,
    data: bytes,
    file_name: str,
    mime: str = "image/png",
    key: str | None = None,
) -> None:
    """Renderiza um st.download_button com buffer em memória."""
    st.download_button(
        label=label,
        data=data,
        file_name=file_name,
        mime=mime,
        key=key,
    )
