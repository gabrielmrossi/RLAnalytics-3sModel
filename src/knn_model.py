"""
knn_model.py — Fachada e pipeline integrado de modelos para telemetria de Rocket League.
Reexporta os módulos especializados:
    - classifiers.py (KNN, Decision Tree, Random Forest)
    - clustering.py (K-Means + PCA)
    - benchmarking.py (StratifiedKFold cv=5)
    - error_detection.py (Detectores Heurístico e Model-Driven)
"""

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from benchmarking import ModelBenchmarking
from classifiers import (
    RankClassifierKNN,
    RankClassifierRandomForest,
    RankClassifierTree,
    loo_evaluate,
)
from clustering import TacticalClusterKMeans
from config import DATASET_CSV_PATH, META_COLS
from error_detection import (
    ModelDrivenErrorDetector,
    PositioningErrorDetector,
    build_reference_profile,
)

_META_COLS = META_COLS


def load_dataset(csv_path: str = str(DATASET_CSV_PATH)) -> pd.DataFrame:
    """Carrega o dataset de features do CSV."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset nao encontrado: {csv_path}")
    return pd.read_csv(csv_path)


def get_X_y(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Separa features (X) e labels (y) do DataFrame.

    Returns:
        (X, y, feature_names) onde X eh array 2D, y eh array 1D de strings.
    """
    feature_cols = [c for c in df.columns if c not in _META_COLS]
    X = df[feature_cols].values.astype(float)
    y = df["label"].values
    return X, y, feature_cols


def run_full_pipeline(csv_path: str = str(DATASET_CSV_PATH)) -> dict[str, Any]:
    """Roda pipeline completo: carrega dados, treina modelos, reporta resultados."""
    print("=== Carregando dataset ===")
    df = load_dataset(csv_path)
    X, y, feature_names = get_X_y(df)
    print(f"  Amostras: {len(X)}, Features: {len(feature_names)}")
    print(f"  Labels: {dict(zip(*np.unique(y, return_counts=True)))}")

    k = min(3, len(X) - 1)

    # ── KNN ──
    print(f"\n=== KNN (k={k}) com Leave-One-Out ===")
    knn = RankClassifierKNN(n_neighbors=k)
    knn_results = knn.evaluate_loo(X, y)
    knn.train(X, y)
    print(f"  Accuracy: {knn_results['accuracy']:.1%}")
    print(f"  Confusion Matrix: {knn_results['confusion_matrix']}")

    importance = knn.feature_importance_loo(X, y, feature_names)
    print("\n  Importancia das features (ablation):")
    for name, drop in importance.items():
        indicator = "***" if drop > 0.05 else ""
        print(f"    {name:30s}: drop = {drop:+.4f} {indicator}")

    # ── Decision Tree ──
    print("\n=== Decision Tree (depth=3) com Leave-One-Out ===")
    tree = RankClassifierTree(max_depth=3)
    tree_results = tree.evaluate_loo(X, y)
    tree.train(X, y, feature_names)
    print(f"  Accuracy: {tree_results['accuracy']:.1%}")
    print(f"  Confusion Matrix: {tree_results['confusion_matrix']}")
    print("\n  Regras da arvore:")
    print(tree.get_rules())

    # ── Random Forest ──
    print("\n=== Random Forest (n_estimators=100) com Leave-One-Out ===")
    rf = RankClassifierRandomForest(n_estimators=100, random_state=42)
    rf_results = rf.evaluate_loo(X, y)
    rf.train(X, y, feature_names)
    print(f"  Accuracy: {rf_results['accuracy']:.1%}")
    print(f"  Confusion Matrix: {rf_results['confusion_matrix']}")

    plot_path = "data/processed/rf_feature_importance.png"
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    rf.plot_feature_importances(save_path=plot_path)

    # ── K-Means Clustering ──
    print("\n=== K-Means Clustering (nao-supervisionado) ===")
    clusterer = TacticalClusterKMeans(n_clusters=3, random_state=42)
    clusterer.elbow_analysis(X, save_path="data/processed/kmeans_elbow.png")
    cluster_labels = clusterer.fit_predict(X)
    print(f"  Silhouette Score: {clusterer.silhouette_:.4f}" if clusterer.silhouette_ else "  Silhouette: N/A")
    print(f"  Distribuicao: {dict(zip(*np.unique(cluster_labels, return_counts=True)))}")

    clusterer.plot_pca_clusters(X, cluster_labels, feature_names, save_path="data/processed/kmeans_pca.png")
    summary = clusterer.cluster_summary(df, cluster_labels, feature_names)
    print("\n  Resumo por cluster:")
    print(summary.to_string())

    # ── Model Benchmarking ──
    print("\n=== Benchmarking Comparativo (StratifiedKFold cv=5) ===")
    bench = ModelBenchmarking(cv=5, random_state=42)
    bench_df = bench.evaluate(X, y, knn_k=k)
    print(bench_df.to_string(index=False))
    bench.plot_confusion_matrices(save_path="data/processed/models_comparison_cm.png")

    # ── Error Detection (Comparação dos dois detectores) ──
    print("\n=== Deteccao de Erros: Comparativo Heurístico vs Model-Driven ===")
    detector_heur = PositioningErrorDetector()
    detector_heur.build_reference(df)

    detector_model = ModelDrivenErrorDetector(n_estimators=100, random_state=42)
    detector_model.build(df, X, y, feature_names)

    lower_rows = df[df["label"] == "lower"]
    heur_err_counts = []
    model_err_counts = []

    for _, row in lower_rows.iterrows():
        features_dict = {col: row[col] for col in feature_names}
        errs_heur = detector_heur.detect_errors(features_dict)
        errs_model = detector_model.detect_errors(features_dict)
        heur_err_counts.append(len(errs_heur))
        model_err_counts.append(len(errs_model))

    print(f"  Amostras Lower avaliadas: {len(lower_rows)}")
    print(f"  Erros médios / time (Heurístico): {np.mean(heur_err_counts):.1f}")
    print(f"  Erros médios / time (Model-Driven): {np.mean(model_err_counts):.1f}")

    return {
        "knn_results": knn_results,
        "tree_results": tree_results,
        "rf_results": rf_results,
        "tree_rules": tree.get_rules(),
        "knn_importance": importance,
        "tree_importance": tree.get_feature_importances(),
        "rf_importance": rf.get_feature_importances(),
        "knn_model": knn,
        "tree_model": tree,
        "rf_model": rf,
        "cluster_model": clusterer,
        "cluster_labels": cluster_labels,
        "cluster_summary": summary,
        "benchmark": bench,
        "benchmark_summary": bench_df,
        "detector": detector_model,
        "detector_heuristico": detector_heur,
        "dataset": df,
    }


if __name__ == "__main__":
    results = run_full_pipeline()
