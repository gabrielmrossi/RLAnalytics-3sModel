"""
higher_vs_lower.py — Aba 2: Comparativo Higher vs Lower Rank (Radar + Diferenças).
"""

import pandas as pd
import streamlit as st

from app.components.downloads import render_download_button
from app.services import load_feature_dataset
from config import META_COLS
from feature_catalog import get_feature_label
from visualization.interactive import build_radar_chart
from visualization.static import figure_to_bytes, plot_radar_static


def tab_higher_vs_lower() -> None:
    """Aba 2: Comparativo dos times Higher vs Lower através de gráfico polar e tabela analítica."""
    st.header("Higher vs Lower Rank")

    dataset = load_feature_dataset()
    if dataset is None:
        st.warning("Dataset de features não encontrado. Execute a extração de dados primeiro.")
        return

    df, X, y, feature_names, _ = dataset

    feature_cols = [c for c in df.columns if c not in META_COLS]

    higher_means = df[df["label"] == "higher"][feature_cols].mean()
    lower_means = df[df["label"] == "lower"][feature_cols].mean()

    # Normalizar para radar (0 a 100%)
    all_vals = pd.concat([higher_means, lower_means])
    max_vals = all_vals.groupby(level=0).max().replace(0, 1)
    higher_norm = (higher_means / max_vals * 100).round(1)
    lower_norm = (lower_means / max_vals * 100).round(1)

    translated_cols = [get_feature_label(c) for c in feature_cols]

    # Gráfico radar interativo
    fig_radar = build_radar_chart(
        categories=translated_cols,
        higher_values=higher_norm.values,
        lower_values=lower_norm.values,
        title="Comparativo Tático Médio (Higher vs Lower)",
    )
    st.plotly_chart(fig_radar, use_container_width=True)
    st.caption(
        "Compara o formato do estilo de jogo entre os grupos. "
        "Formas mais amplas indicam maior domínio dos fundamentos analisados."
    )

    # Download em alta resolução (300 DPI)
    radar_bytes = figure_to_bytes(
        plot_radar_static(translated_cols, higher_norm.values, lower_norm.values)
    )
    render_download_button(
        label="📥 Baixar Gráfico Radar (PNG 300 DPI)",
        data=radar_bytes,
        file_name="radar_higher_vs_lower.png",
        mime="image/png",
    )

    # Tabela de diferenças
    st.subheader("Diferenças por Característica")
    diffs = []
    for col in feature_cols:
        h_val = higher_means[col]
        l_val = lower_means[col]
        pct_diff = ((l_val - h_val) / abs(h_val)) * 100 if h_val != 0 else 0
        diffs.append({
            "Característica": get_feature_label(col),
            "Higher": round(h_val, 1),
            "Lower": round(l_val, 1),
            "Diferença %": round(pct_diff, 1),
        })

    diffs_df = pd.DataFrame(diffs).sort_values("Diferença %", key=abs, ascending=False)
    st.dataframe(diffs_df, use_container_width=True, hide_index=True)
