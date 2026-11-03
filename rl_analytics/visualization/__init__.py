"""
src/visualization — Módulos desacoplados de renderização de gráficos.
interactive.py: Construtores de figuras Plotly (tema escuro, para a UI em memória).
static.py: Construtores Matplotlib/Seaborn (alta resolução, 300 DPI, para CLI/downloads).
"""

from .interactive import (
    apply_dark_layout,
    build_cluster_donut_chart,
    build_confusion_matrices_chart,
    build_elbow_chart,
    build_feature_importances_chart,
    build_pca_clusters_chart,
    build_radar_chart,
)
from .static import (
    figure_to_bytes,
    plot_confusion_matrices_static,
    plot_elbow_static,
    plot_feature_importances_static,
    plot_pca_clusters_static,
    plot_radar_static,
    save_figure,
)

__all__ = [
    "apply_dark_layout",
    "build_cluster_donut_chart",
    "build_confusion_matrices_chart",
    "build_elbow_chart",
    "build_feature_importances_chart",
    "build_pca_clusters_chart",
    "build_radar_chart",
    "figure_to_bytes",
    "plot_confusion_matrices_static",
    "plot_elbow_static",
    "plot_feature_importances_static",
    "plot_pca_clusters_static",
    "plot_radar_static",
    "save_figure",
]
