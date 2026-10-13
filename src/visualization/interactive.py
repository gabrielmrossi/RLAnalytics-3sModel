"""
interactive.py — Construtores de figuras Plotly para exibição interativa na interface.
Padroniza tema escuro, tipografia e paletas de cores sem nenhuma operação de I/O em disco.
"""

from typing import Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import CLUSTER_PALETTE, DARK_BG, DARK_PAPER


def apply_dark_layout(
    fig: go.Figure,
    title: str | dict | None = None,
    height: int | None = None,
    **kwargs,
) -> go.Figure:
    """Aplica o tema escuro padronizado a uma figura Plotly."""
    layout_update = dict(
        plot_bgcolor=DARK_BG,
        paper_bgcolor=DARK_PAPER,
        font=dict(color="white"),
        margin=dict(l=40, r=40, t=50, b=40),
    )
    if title is not None:
        if isinstance(title, str):
            layout_update["title"] = dict(text=title, font=dict(color="white", size=15))
        elif isinstance(title, dict):
            layout_update["title"] = title
    if height is not None:
        layout_update["height"] = height

    layout_update.update(kwargs)
    fig.update_layout(**layout_update)
    return fig


def build_elbow_chart(
    k_values: Sequence[int],
    inertias: Sequence[float],
    title: str = "Curva do Cotovelo - K-Means",
) -> go.Figure:
    """Gera gráfico interativo da curva do cotovelo para o K-Means."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(k_values),
            y=list(inertias),
            mode="lines+markers",
            line=dict(color="#e94560", width=3),
            marker=dict(size=10, color="#e94560"),
            name="Inércia",
            hovertemplate="k=%{x}<br>Inércia=%{y:,.0f}<extra></extra>",
        )
    )
    apply_dark_layout(
        fig,
        title=title,
        height=450,
        xaxis=dict(title="Número de Clusters (k)", dtick=1, gridcolor="#333333"),
        yaxis=dict(title="Inércia (WCSS)", gridcolor="#333333"),
    )
    return fig


def build_pca_clusters_chart(
    pca_coords: np.ndarray,
    labels: np.ndarray,
    var_explained: np.ndarray,
    cluster_palette: Sequence[str] | None = None,
    feature_names: Sequence[str] | None = None,
    loadings: np.ndarray | None = None,
    title: str = "Clusters Táticos (PCA 2D)",
) -> go.Figure:
    """Gera gráfico de dispersão 2D com clusters coloridos via PCA."""
    palette = list(cluster_palette or CLUSTER_PALETTE)
    unique_labels = sorted(set(labels))

    fig = go.Figure()
    for cl in unique_labels:
        mask = labels == cl
        color = palette[cl % len(palette)]
        fig.add_trace(
            go.Scatter(
                x=pca_coords[mask, 0],
                y=pca_coords[mask, 1],
                mode="markers",
                name=f"Cluster {cl}",
                marker=dict(
                    color=color,
                    size=8,
                    opacity=0.75,
                    line=dict(color="white", width=0.5),
                ),
                hovertemplate=f"Cluster {cl}<br>PC1=%{{x:.2f}}<br>PC2=%{{y:.2f}}<extra></extra>",
            )
        )

    pc1_pct = f"{var_explained[0]:.1%}" if len(var_explained) > 0 else ""
    pc2_pct = f"{var_explained[1]:.1%}" if len(var_explained) > 1 else ""

    apply_dark_layout(
        fig,
        title=title,
        height=550,
        xaxis=dict(title=f"PC1 ({pc1_pct} var.)", gridcolor="#333333", zeroline=False),
        yaxis=dict(title=f"PC2 ({pc2_pct} var.)", gridcolor="#333333", zeroline=False),
        legend=dict(bgcolor="rgba(30,30,30,0.8)", bordercolor="#555555", borderwidth=1),
    )

    if feature_names is not None and loadings is not None:
        scale = max(abs(pca_coords[:, 0]).max(), abs(pca_coords[:, 1]).max()) * 0.8
        load_scale = max(abs(loadings).max(), 1e-9)
        for i, name in enumerate(feature_names):
            lx = loadings[i, 0] / load_scale * scale
            ly = loadings[i, 1] / load_scale * scale
            fig.add_annotation(
                x=lx,
                y=ly,
                text=name,
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowcolor="#aaaaaa",
                font=dict(size=9, color="#cccccc"),
                ax=0,
                ay=0,
            )

    return fig


def build_cluster_donut_chart(
    unique_labels: Sequence[int],
    counts: Sequence[int],
    cluster_palette: Sequence[str] | None = None,
    title: str = "Distribuição dos Perfis Táticos",
) -> go.Figure:
    """Gera gráfico de rosca (donut chart) sincronizado com a mesma paleta do PCA."""
    palette = list(cluster_palette or CLUSTER_PALETTE)
    colors = [palette[u % len(palette)] for u in unique_labels]
    donut_labels = [f"Perfil {u}" for u in unique_labels]

    fig = go.Figure(
        go.Pie(
            labels=donut_labels,
            values=list(counts),
            hole=0.55,
            marker=dict(colors=colors, line=dict(color=DARK_BG, width=2)),
            textinfo="label+percent",
            textfont=dict(size=12, color="white"),
            hovertemplate="%{label}<br>Amostras: %{value} (%{percent})<extra></extra>",
        )
    )
    apply_dark_layout(
        fig,
        title=title,
        height=360,
        showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0.5)", font=dict(color="white")),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def build_confusion_matrices_chart(
    confusion_matrices: dict[str, np.ndarray],
    labels: Sequence[str],
    title: str = "Matrizes de Confusão Normalizadas (StratifiedKFold)",
) -> go.Figure:
    """Gera subplot de matrizes de confusão com anotações de porcentagem."""
    model_names = list(confusion_matrices.keys())
    fig = make_subplots(
        rows=1,
        cols=len(model_names),
        subplot_titles=model_names,
        horizontal_spacing=0.08,
    )

    for i, name in enumerate(model_names, 1):
        cm = confusion_matrices[name]
        text_matrix = [[f"{v:.1%}" for v in row] for row in cm]

        fig.add_trace(
            go.Heatmap(
                z=cm,
                x=list(labels),
                y=list(labels),
                text=text_matrix,
                texttemplate="%{text}",
                colorscale="Blues",
                showscale=(i == len(model_names)),
                zmin=0,
                zmax=1,
                colorbar=dict(title="Taxa") if i == len(model_names) else None,
            ),
            row=1,
            col=i,
        )
        fig.update_xaxes(title_text="Previsto", row=1, col=i)
        fig.update_yaxes(title_text="Real" if i == 1 else "", row=1, col=i)

    apply_dark_layout(fig, title=title, height=400)
    for annotation in fig.layout.annotations:
        annotation.font.color = "white"
    return fig


def build_feature_importances_chart(
    importances: dict[str, float],
    title: str = "Importância das Features (Random Forest)",
) -> go.Figure:
    """Gera gráfico de barras horizontais para importância das features."""
    features = list(importances.keys())
    values = list(importances.values())

    fig = go.Figure(
        go.Bar(
            x=values,
            y=features,
            orientation="h",
            marker_color="#2ecc71",
            text=[f"{v:.1%}" for v in values],
            textposition="auto",
        )
    )
    apply_dark_layout(
        fig,
        title=title,
        height=max(420, len(features) * 25),
        xaxis_title="Importância Relativa (MDI)",
        margin=dict(l=220, t=40, r=20, b=40),
    )
    return fig


def build_radar_chart(
    categories: Sequence[str],
    higher_values: Sequence[float],
    lower_values: Sequence[float],
    title: str = "Comparativo Tático Médio",
) -> go.Figure:
    """Gera gráfico de radar comparando times Higher vs Lower."""
    cat_closed = list(categories) + [categories[0]]
    h_closed = list(higher_values) + [higher_values[0]]
    l_closed = list(lower_values) + [lower_values[0]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=h_closed,
            theta=cat_closed,
            fill="toself",
            name="Higher (GC1+)",
            line_color="#3498db",
            fillcolor="rgba(52, 152, 219, 0.2)",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=l_closed,
            theta=cat_closed,
            fill="toself",
            name="Lower (Bronze-Prata)",
            line_color="#e67e22",
            fillcolor="rgba(230, 126, 34, 0.2)",
        )
    )
    apply_dark_layout(
        fig,
        title=title,
        height=480,
        polar=dict(
            bgcolor=DARK_BG,
            radialaxis=dict(visible=True, range=[0, 110], color="#555555"),
            angularaxis=dict(color="#aaaaaa"),
        ),
        legend=dict(bgcolor="rgba(0,0,0,0.5)"),
        margin=dict(t=40, b=40, l=40, r=40),
    )
    return fig
