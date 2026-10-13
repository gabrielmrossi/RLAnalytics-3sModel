"""
static.py — Construtores de figuras estáticas via Matplotlib/Seaborn.
Projetado para exportação em alta resolução (300 DPI, PNG/PDF/SVG) para o TCC ou download sob demanda.
Configura matplotlib.use("Agg") garantindo execução headless segura.
"""

import io
import os
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import CLUSTER_PALETTE


def figure_to_bytes(fig: plt.Figure, fmt: str = "png", dpi: int = 300) -> bytes:
    """Converte uma figura Matplotlib em buffer de bytes na memória (sem salvar no disco)."""
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    data = buf.getvalue()
    buf.close()
    return data


def save_figure(fig: plt.Figure, filepath: str | Path, dpi: int = 300) -> None:
    """Salva uma figura no disco com resolução configurada e diretórios criados."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(filepath), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_elbow_static(
    k_values: Sequence[int],
    inertias: Sequence[float],
    title: str = "Curva do Cotovelo - K-Means",
) -> plt.Figure:
    """Gera figura estática Seaborn/Matplotlib da curva do cotovelo."""
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(list(k_values), list(inertias), "o-", color="#e94560", linewidth=2, markersize=8)
    ax.set_xlabel("Numero de Clusters (k)", fontsize=12)
    ax.set_ylabel("Inercia (WCSS)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(list(k_values))
    fig.tight_layout()
    return fig


def plot_pca_clusters_static(
    pca_coords: np.ndarray,
    labels: np.ndarray,
    var_explained: np.ndarray,
    cluster_palette: Sequence[str] | None = None,
    title: str = "Clusters Taticos (PCA 2D)",
) -> plt.Figure:
    """Gera figura estática de dispersão 2D dos clusters via PCA."""
    palette = list(cluster_palette or CLUSTER_PALETTE)
    unique_labels = sorted(set(labels))

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 6))

    for cl in unique_labels:
        mask = labels == cl
        color = palette[cl % len(palette)]
        ax.scatter(
            pca_coords[mask, 0],
            pca_coords[mask, 1],
            c=color,
            label=f"Cluster {cl}",
            alpha=0.75,
            s=65,
            edgecolors="white",
            linewidth=0.5,
        )

    pc1_lbl = f"PC1 ({var_explained[0]:.1%} var.)" if len(var_explained) > 0 else "PC1"
    pc2_lbl = f"PC2 ({var_explained[1]:.1%} var.)" if len(var_explained) > 1 else "PC2"

    ax.set_xlabel(pc1_lbl, fontsize=12)
    ax.set_ylabel(pc2_lbl, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    fig.tight_layout()
    return fig


def plot_confusion_matrices_static(
    confusion_matrices: dict[str, np.ndarray],
    labels: Sequence[str],
    title: str = "Matrizes de Confusao Normalizadas (StratifiedKFold)",
) -> plt.Figure:
    """Gera subplot estático com matrizes de confusão lado a lado via Seaborn."""
    model_names = list(confusion_matrices.keys())
    sns.set_theme(style="white")
    fig, axes = plt.subplots(1, len(model_names), figsize=(5 * len(model_names), 4))
    if len(model_names) == 1:
        axes = [axes]

    for ax, name in zip(axes, model_names):
        cm_norm = confusion_matrices[name]
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2%",
            cmap="YlOrRd",
            xticklabels=list(labels),
            yticklabels=list(labels),
            ax=ax,
            cbar=False,
            square=True,
            annot_kws={"fontsize": 12},
        )
        ax.set_title(name, fontsize=13, fontweight="bold")
        ax.set_xlabel("Previsto", fontsize=10)
        ax.set_ylabel("Real", fontsize=10)

    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def plot_feature_importances_static(
    importances: dict[str, float],
    title: str = "Importancia das Features (Random Forest)",
) -> plt.Figure:
    """Gera gráfico de barras horizontal de importância relativa de features."""
    if not importances:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Modelo nao treinado", ha="center", va="center")
        return fig

    imp_df = pd.DataFrame(
        {"feature": list(importances.keys()), "importance": list(importances.values())}
    )

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, max(4, len(imp_df) * 0.5)))
    palette = sns.color_palette("viridis", len(imp_df))
    barplot = sns.barplot(
        data=imp_df,
        x="importance",
        y="feature",
        hue="feature",
        palette=palette,
        legend=False,
        ax=ax,
    )

    for p in barplot.patches:
        val = p.get_width()
        ax.annotate(
            f"{val:.2%}",
            (val, p.get_y() + p.get_height() / 2.0),
            ha="left",
            va="center",
            fontsize=10,
            color="black",
            xytext=(5, 0),
            textcoords="offset points",
        )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Importancia Relativa (MDI)", fontsize=11)
    ax.set_ylabel("Feature de Telemetria", fontsize=11)

    max_val = imp_df["importance"].max() if not imp_df.empty else 0.1
    ax.set_xlim(0, max_val * 1.2)
    fig.tight_layout()
    return fig


def plot_radar_static(
    categories: Sequence[str],
    higher_values: Sequence[float],
    lower_values: Sequence[float],
    title: str = "Comparativo Tatico Medio (Higher vs Lower)",
) -> plt.Figure:
    """Gera gráfico de radar polar estático comparando perfis de rank."""
    sns.set_theme(style="whitegrid")
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    h_vals = list(higher_values) + [higher_values[0]]
    l_vals = list(lower_values) + [lower_values[0]]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    ax.plot(angles, h_vals, color="#3498db", linewidth=2, label="Higher (GC1+)")
    ax.fill(angles, h_vals, color="#3498db", alpha=0.25)

    ax.plot(angles, l_vals, color="#e67e22", linewidth=2, label="Lower (Bronze-Prata)")
    ax.fill(angles, l_vals, color="#e67e22", alpha=0.25)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(list(categories), fontsize=10)
    ax.set_ylim(0, 110)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1.1))
    fig.tight_layout()
    return fig
