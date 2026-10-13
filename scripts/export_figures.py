"""
scripts/export_figures.py — CLI para exportação de figuras acadêmicas em alta resolução (300 DPI).
Gera gráficos definitivos para o TCC (PNG, PDF vetorial ou SVG) sem depender do Streamlit.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Adiciona src ao path para permitir imports diretos
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd

from config import CLUSTER_PALETTE, DATASET_CSV_PATH, FIGURES_DIR, PROCESSED_DIR
from feature_catalog import get_feature_label
from knn_model import (
    ModelBenchmarking,
    RankClassifierRandomForest,
    TacticalClusterKMeans,
    get_X_y,
    load_dataset,
)
from visualization.static import (
    plot_confusion_matrices_static,
    plot_elbow_static,
    plot_feature_importances_static,
    plot_pca_clusters_static,
    plot_radar_static,
    save_figure,
)


def export_model_figures(
    df: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    out_dir: Path,
    fmt: str = "png",
    dpi: int = 300,
) -> None:
    """Gera todas as figuras analíticas dos modelos de IA."""
    out_dir.mkdir(parents=True, exist_ok=True)
    k = min(3, len(X) - 1)

    print(f"[*] Exportando figuras em '{out_dir}' (formato={fmt}, dpi={dpi})...")

    # 1. Curva do Cotovelo (K-Means)
    print("  -> Gerando kmeans_elbow...")
    clusterer = TacticalClusterKMeans(n_clusters=3, random_state=42)
    k_vals, inertias = clusterer.compute_elbow(X)
    fig_elbow = plot_elbow_static(k_vals, inertias)
    save_figure(fig_elbow, out_dir / f"kmeans_elbow.{fmt}", dpi=dpi)

    # 2. Clusters PCA (K-Means)
    print("  -> Gerando kmeans_pca...")
    cluster_labels = clusterer.fit_predict(X)
    X_pca, var_explained, _ = clusterer.compute_pca(X)
    fig_pca = plot_pca_clusters_static(
        X_pca, cluster_labels, var_explained, cluster_palette=CLUSTER_PALETTE
    )
    save_figure(fig_pca, out_dir / f"kmeans_pca.{fmt}", dpi=dpi)

    # 3. Matrizes de Confusão (Benchmarking)
    print("  -> Gerando models_comparison_cm...")
    bench = ModelBenchmarking(cv=5, random_state=42)
    bench.evaluate(X, y, knn_k=k)
    fig_cm = plot_confusion_matrices_static(
        bench.confusion_matrices_, labels=["lower", "higher"]
    )
    save_figure(fig_cm, out_dir / f"models_comparison_cm.{fmt}", dpi=dpi)

    # 4. Importância das Features (Random Forest)
    print("  -> Gerando rf_feature_importance...")
    rf = RankClassifierRandomForest(n_estimators=100, random_state=42)
    rf.train(X, y, feature_names)
    rf_imp = rf.get_feature_importances()
    fig_imp = plot_feature_importances_static(rf_imp)
    save_figure(fig_imp, out_dir / f"rf_feature_importance.{fmt}", dpi=dpi)

    # 5. Radar Higher vs Lower
    print("  -> Gerando radar_higher_vs_lower...")
    meta_cols = {"label", "replay_name", "team", "team_label"}
    feature_cols = [c for c in df.columns if c not in meta_cols]
    higher_means = df[df["label"] == "higher"][feature_cols].mean()
    lower_means = df[df["label"] == "lower"][feature_cols].mean()
    all_vals = pd.concat([higher_means, lower_means])
    max_vals = all_vals.groupby(level=0).max().replace(0, 1)
    higher_norm = (higher_means / max_vals * 100).round(1)
    lower_norm = (lower_means / max_vals * 100).round(1)
    translated_cols = [get_feature_label(c) for c in feature_cols]

    fig_radar = plot_radar_static(
        translated_cols, higher_norm.values, lower_norm.values
    )
    save_figure(fig_radar, out_dir / f"radar_higher_vs_lower.{fmt}", dpi=dpi)

    print("[+] Todas as figuras dos modelos foram exportadas com sucesso!")


def main():
    parser = argparse.ArgumentParser(
        description="CLI para exportação de figuras em alta resolução (300 DPI) para o TCC."
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolução da imagem em DPI (padrão: 300)",
    )
    parser.add_argument(
        "--format",
        choices=["png", "pdf", "svg"],
        default="png",
        help="Formato do arquivo: png (raster) ou pdf/svg (vetorial para LaTeX)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(FIGURES_DIR),
        help=f"Diretório de saída para as figuras (padrão: {FIGURES_DIR})",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(DATASET_CSV_PATH),
        help=f"Caminho para o CSV de features (padrão: {DATASET_CSV_PATH})",
    )

    args = parser.parse_args()

    csv_path = Path(args.dataset)
    if not csv_path.exists():
        print(f"[!] Erro: Arquivo de dataset não encontrado em '{csv_path}'")
        sys.exit(1)

    df = load_dataset(str(csv_path))
    X, y, feature_names = get_X_y(df)

    out_path = Path(args.out)
    export_model_figures(
        df=df,
        X=X,
        y=y,
        feature_names=feature_names,
        out_dir=out_path,
        fmt=args.format,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
