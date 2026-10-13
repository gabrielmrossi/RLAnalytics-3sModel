"""
models.py — Aba 3: Modelos de Inteligência Artificial (Comparativo, Importância, Clusterização).
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from app.components.downloads import render_download_button
from app.services import (
    get_benchmarking_results,
    get_kmeans_clusters_data,
    get_kmeans_elbow_data,
    get_rf_importances_data,
    load_feature_dataset,
)
from config import CLUSTER_PALETTE, DARK_BG, DARK_PAPER
from feature_catalog import FEATURE_LABELS, get_feature_label
from knn_model import RankClassifierTree, TacticalClusterKMeans
from visualization.interactive import (
    build_cluster_donut_chart,
    build_confusion_matrices_chart,
    build_elbow_chart,
    build_feature_importances_chart,
    build_pca_clusters_chart,
)
from visualization.static import (
    figure_to_bytes,
    plot_confusion_matrices_static,
    plot_elbow_static,
    plot_feature_importances_static,
    plot_pca_clusters_static,
)


def tab_models() -> None:
    """Aba 3: Benchmarking comparativo, importância de features e agrupamento não supervisionado."""
    st.header("Modelos de IA")

    dataset = load_feature_dataset()
    if dataset is None:
        st.error("Dataset de features não encontrado. Execute a extração de dados primeiro.")
        return

    df, X, y, feature_names, k = dataset

    st.caption(
        f"Dataset: **{len(X)}** amostras · **{len(feature_names)}** features · "
        f"Higher: {sum(y == 'higher')} · Lower: {sum(y == 'lower')}"
    )

    sub1, sub2, sub3 = st.tabs([
        "📋 Comparativo",
        "📊 Importância",
        "🔬 Clusterização",
    ])

    df_hash = f"{len(df)}_{len(feature_names)}"

    # ── Sub-tab 1: Comparativo ──────────────────
    with sub1:
        st.subheader("Comparativo de Modelos")
        st.markdown(
            "Avaliação dos **3 modelos supervisionados** usando "
            "**StratifiedKFold (cv=5)** nos mesmos folds."
        )

        bench_df, confusion_matrices = get_benchmarking_results(df_hash, X, y, k)

        st.dataframe(
            bench_df.style.set_properties(**{"text-align": "center"}),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("#### Matrizes de Confusão Normalizadas")
        fig_cm = build_confusion_matrices_chart(confusion_matrices, ["lower", "higher"])
        st.plotly_chart(fig_cm, use_container_width=True)

        cm_bytes = figure_to_bytes(
            plot_confusion_matrices_static(confusion_matrices, ["lower", "higher"])
        )
        render_download_button(
            label="📥 Baixar Matrizes de Confusão (PNG 300 DPI)",
            data=cm_bytes,
            file_name="models_comparison_cm.png",
            mime="image/png",
        )

        with st.expander("Detalhes Técnicos (Regras da Decision Tree)", expanded=False):
            st.markdown("Regras extraídas da Decision Tree ajustada com todo o dataset.")
            tree = RankClassifierTree(max_depth=3)
            tree.train(X, y, feature_names)
            st.code(tree.get_rules(), language="text")

    # ── Sub-tab 2: Importância ──────────────────
    with sub2:
        st.subheader("Importância das Features (Random Forest)")
        st.markdown(
            "A importância é calculada via **Mean Decrease in Impurity (MDI)** "
            "do Random Forest — o método mais robusto entre os 3 modelos."
        )

        rf_imp = get_rf_importances_data(df_hash, X, y, feature_names)

        if rf_imp:
            translated_imp = {get_feature_label(k): v for k, v in rf_imp.items()}

            fig_imp = go.Figure(go.Bar(
                x=list(translated_imp.values()),
                y=list(translated_imp.keys()),
                orientation="h",
                marker_color="#2ecc71",
                text=[f"{v:.1%}" for v in translated_imp.values()],
                textposition="auto",
            ))
            fig_imp.update_layout(
                xaxis_title="Importância Relativa",
                plot_bgcolor=DARK_BG,
                paper_bgcolor=DARK_BG,
                font=dict(color="white"),
                height=420,
                margin=dict(l=220, t=20, r=20),
            )
            st.plotly_chart(fig_imp, use_container_width=True)
            st.caption(
                "Mostra o peso de cada variável na decisão da Inteligência Artificial. "
                "Variáveis no topo são os maiores divisores de água entre ranks altos e baixos."
            )

            rf_bytes = figure_to_bytes(plot_feature_importances_static(rf_imp))
            render_download_button(
                label="📥 Baixar Importância das Features (PNG 300 DPI)",
                data=rf_bytes,
                file_name="rf_feature_importance.png",
                mime="image/png",
            )

    # ── Sub-tab 3: Clusterização ────────────────
    with sub3:
        st.subheader("Clusterização Comportamental (K-Means)")
        st.info(
            "Abordagem **não-supervisionada**: agrupa amostras por perfil tático "
            "sem usar labels. Não comparável com as métricas dos modelos supervisionados."
        )

        n_clusters = st.slider(
            "Número de clusters (k)", min_value=2, max_value=8, value=3, key="kmeans_k"
        )

        col_elbow, col_metrics = st.columns([2, 1])
        with col_elbow:
            k_vals, inertias = get_kmeans_elbow_data(df_hash, X)
            fig_elbow = build_elbow_chart(k_vals, inertias)
            st.plotly_chart(fig_elbow, use_container_width=True)

            elbow_bytes = figure_to_bytes(plot_elbow_static(k_vals, inertias))
            render_download_button(
                label="📥 Baixar Curva do Cotovelo (PNG 300 DPI)",
                data=elbow_bytes,
                file_name="kmeans_elbow.png",
                mime="image/png",
            )

        cluster_labels, X_pca, var_explained, loadings, silhouette = get_kmeans_clusters_data(
            df_hash, X, n_clusters
        )

        with col_metrics:
            st.metric(
                "Silhouette Score",
                f"{silhouette:.4f}" if silhouette is not None else "N/A",
            )
            unique, counts = np.unique(cluster_labels, return_counts=True)

        fig_donut = build_cluster_donut_chart(unique, counts, cluster_palette=CLUSTER_PALETTE)

        pca_col, donut_col = st.columns([2, 1])
        with pca_col:
            fig_pca = build_pca_clusters_chart(
                X_pca,
                cluster_labels,
                var_explained,
                cluster_palette=CLUSTER_PALETTE,
                feature_names=feature_names,
                loadings=loadings,
            )
            st.plotly_chart(fig_pca, use_container_width=True)
            st.caption(
                "Agrupa times por similaridade de comportamento, independentemente do rank, "
                "ajudando a identificar estilos de jogo únicos."
            )

            pca_bytes = figure_to_bytes(
                plot_pca_clusters_static(X_pca, cluster_labels, var_explained, CLUSTER_PALETTE)
            )
            render_download_button(
                label="📥 Baixar Clusters PCA (PNG 300 DPI)",
                data=pca_bytes,
                file_name="kmeans_pca.png",
                mime="image/png",
            )

        with donut_col:
            st.plotly_chart(fig_donut, use_container_width=True)

        with st.expander("Perfil médio por cluster", expanded=False):
            clusterer = TacticalClusterKMeans(n_clusters=n_clusters, random_state=42)
            summary_df = clusterer.cluster_summary(df, cluster_labels, feature_names)
            summary_df = summary_df.copy()
            summary_df.columns = [
                get_feature_label(c) if c in FEATURE_LABELS else c for c in summary_df.columns
            ]
            st.dataframe(summary_df, use_container_width=True)
