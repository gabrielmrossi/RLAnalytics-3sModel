"""
services.py — Camada de serviços e caches inteligentes do Streamlit.
Centraliza carregamento de replays, treino/carregamento do detector e cálculos pesados de ML.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from rl_analytics.analyzer import ReplayAnalyzer
from rl_analytics.config import (
    DATASET_CSV_PATH,
    FIELD_HALF_LENGTH,
    FIELD_HALF_WIDTH,
    MODEL_PATH,
    PROCESSED_DIR,
)
from rl_analytics.diagnosis import (
    check_stagnation_anomaly as _domain_check_stagnation_anomaly,
    generate_tactical_narrative as _domain_generate_tactical_narrative,
)
from rl_analytics.knn_model import (
    ModelBenchmarking,
    ModelDrivenErrorDetector,
    RankClassifierRandomForest,
    TacticalClusterKMeans,
    get_X_y,
)
from rl_analytics.parser import discover_replays


@st.cache_data
def load_replay_json(json_path: str) -> dict:
    """Lê o arquivo JSON de telemetria de um replay."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def analyze_replay(json_path: str) -> dict:
    """Executa a análise completa de um replay e cacheia o resultado."""
    data = load_replay_json(json_path)
    analyzer = ReplayAnalyzer(data)
    result = analyzer.analyze()
    return {
        "metadata": result["metadata"],
        "players": result["players"],
        "positions": result["positions"].to_dict() if not result["positions"].empty else {},
        "boost": result["boost"].to_dict() if not result["boost"].empty else {},
        "goals": result["goals"],
        "demolitions": result["demolitions"],
    }


def get_available_replays() -> dict[str, list[dict[str, str]]]:
    """Descobre e mapeia todos os replays processados disponíveis no disco."""
    discovered = discover_replays()
    available: dict[str, list[dict[str, str]]] = {}
    for label, paths in discovered.items():
        for path in paths:
            stem = Path(path).stem
            json_path = Path(PROCESSED_DIR) / f"{stem}.json"
            if json_path.exists():
                available.setdefault(label, []).append({
                    "name": stem,
                    "replay_path": path,
                    "json_path": str(json_path),
                })
    return available


def load_feature_dataset():
    """Carrega dataset de features. Retorna (df, X, y, feature_names, k) ou None se não existir."""
    if not os.path.exists(DATASET_CSV_PATH):
        return None
    df = pd.read_csv(DATASET_CSV_PATH)
    if df.empty or "label" not in df.columns:
        return None
    X, y, feature_names = get_X_y(df)
    k = min(3, len(X) - 1)
    return df, X, y, feature_names, k


@st.cache_resource
def get_detector(_df_hash: str, X: np.ndarray, y: np.ndarray, feature_names: list[str]):
    """Treina ou carrega o ModelDrivenErrorDetector do disco em cache do Streamlit."""
    model_path = str(MODEL_PATH)
    if os.path.exists(model_path):
        try:
            detector = ModelDrivenErrorDetector.load(model_path)
            predictions = detector.predict(X)
            return detector, predictions
        except Exception:
            pass  # Retreina se corrompido

    df = pd.read_csv(DATASET_CSV_PATH)
    detector = ModelDrivenErrorDetector(n_estimators=100, random_state=42)
    detector.build(df, X, y, feature_names)
    detector.save(model_path)
    predictions = detector.predict(X)
    return detector, predictions


@st.cache_data
def get_benchmarking_results(_df_hash: str, X: np.ndarray, y: np.ndarray, k: int):
    """Executa benchmarking comparativo com cache no Streamlit."""
    bench = ModelBenchmarking(cv=5, random_state=42)
    bench_df = bench.evaluate(X, y, knn_k=k)
    return bench_df, bench.confusion_matrices_


@st.cache_data
def get_rf_importances_data(_df_hash: str, X: np.ndarray, y: np.ndarray, feature_names: list[str]):
    """Treina Random Forest e obtém importâncias relativas com cache no Streamlit."""
    rf = RankClassifierRandomForest(n_estimators=100, random_state=42)
    rf.train(X, y, feature_names)
    return rf.get_feature_importances()


@st.cache_data
def get_kmeans_elbow_data(_df_hash: str, X: np.ndarray):
    """Calcula inércias da curva do cotovelo do K-Means com cache no Streamlit."""
    clusterer = TacticalClusterKMeans(n_clusters=3, random_state=42)
    return clusterer.compute_elbow(X, k_range=range(2, 9))


@st.cache_data
def get_kmeans_clusters_data(_df_hash: str, X: np.ndarray, n_clusters: int):
    """Executa K-Means e PCA para um n_clusters específico com cache no Streamlit."""
    clusterer = TacticalClusterKMeans(n_clusters=n_clusters, random_state=42)
    labels = clusterer.fit_predict(X)
    X_pca, var_explained, loadings = clusterer.compute_pca(X)
    return labels, X_pca, var_explained, loadings, clusterer.silhouette_


def check_stagnation_anomaly(positions: pd.DataFrame, team_num: int) -> list[str]:
    """Detecta anomalias de inatividade ou estagnação espacial prolongada por jogador."""
    return _domain_check_stagnation_anomaly(positions, team_num, strip_html=False)


def generate_tactical_narrative(
    team_feats: dict[str, float],
    detector,
) -> tuple[str, float]:
    """Gera diagnóstico em linguagem natural e índice de aderência (0-100)."""
    return _domain_generate_tactical_narrative(team_feats, detector, strip_html=False)
