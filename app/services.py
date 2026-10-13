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

from analyzer import ReplayAnalyzer
from config import (
    DATASET_CSV_PATH,
    FIELD_HALF_LENGTH,
    FIELD_HALF_WIDTH,
    MODEL_PATH,
    PROCESSED_DIR,
)
from feature_catalog import (
    NARRATIVE_TEMPLATES,
    classify_tier,
    compute_z_score,
)
from knn_model import (
    ModelBenchmarking,
    ModelDrivenErrorDetector,
    RankClassifierRandomForest,
    TacticalClusterKMeans,
    get_X_y,
)
from parser import discover_replays


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
    alerts: list[str] = []
    if positions.empty:
        return alerts

    tdata = positions[(positions["team"] == team_num) & (positions["entity"] != "ball")]
    if tdata.empty:
        return alerts

    x_range = [-FIELD_HALF_WIDTH - 300, FIELD_HALF_WIDTH + 300]
    y_range = [-FIELD_HALF_LENGTH - 300, FIELD_HALF_LENGTH + 300]

    for player_name, pgroup in tdata.groupby("entity"):
        if len(pgroup) < 25:
            continue

        h, xedges, yedges = np.histogram2d(
            pgroup["x"].values,
            pgroup["y"].values,
            bins=[16, 20],
            range=[x_range, y_range],
        )
        max_in_cell = float(h.max())
        pct_in_cell = (max_in_cell / len(pgroup)) * 100.0

        if pct_in_cell >= 15.0:
            ix, iy = np.unravel_index(np.argmax(h), h.shape)
            cx = (xedges[ix] + xedges[ix + 1]) / 2
            cy = (yedges[iy] + yedges[iy + 1]) / 2
            alerts.append(
                f"⚠️ <strong>Alerta de Estagnação:</strong> O jogador <strong>{player_name}</strong> "
                f"permaneceu concentrado na coordenada (X≈{cx:.0f}, Y≈{cy:.0f}) durante {pct_in_cell:.0f}% "
                f"do seu tempo em campo (~{int(max_in_cell)}s), indicando ineficiência severa de rotação ou inatividade."
            )

        p_sorted = pgroup.sort_values("frame")
        dx = p_sorted["x"].diff().abs()
        dy = p_sorted["y"].diff().abs()
        is_still = (dx < 30.0) & (dy < 30.0)
        still_streak = 0
        max_still_streak = 0
        still_x, still_y = 0.0, 0.0

        for still, (_, row) in zip(is_still, p_sorted.iterrows()):
            if still:
                still_streak += 1
                if still_streak > max_still_streak:
                    max_still_streak = still_streak
                    still_x, still_y = row["x"], row["y"]
            else:
                still_streak = 0

        if max_still_streak >= 15 and pct_in_cell < 15.0:
            alerts.append(
                f"⚠️ <strong>Alerta de Imobilidade:</strong> O jogador <strong>{player_name}</strong> "
                f"permaneceu praticamente imóvel por ~{max_still_streak}s consecutivos em (X≈{still_x:.0f}, Y≈{still_y:.0f})."
            )

    return alerts


_SCORE_OVERVIEWS = {
    "excellent": "O time demonstra um <strong>jogo sólido</strong>, com fundamentos alinhados ao padrão de jogadores experientes.",
    "good": "O time tem <strong>fundamentos razoáveis</strong>, mas alguns pontos precisam de atenção para subir de nível.",
    "average": "O time apresenta <strong>dificuldades em fundamentos importantes</strong> da rotação e posicionamento.",
    "poor": "O time tem <strong>problemas sérios de rotação</strong> que impedem a evolução no competitivo.",
}


def generate_tactical_narrative(
    team_feats: dict[str, float],
    detector,
) -> tuple[str, float]:
    """Gera diagnóstico em linguagem natural e nota tática (0-100)."""
    if not detector.is_built:
        return "Modelo de IA ainda não disponível para diagnóstico.", 0.0

    good_sentences: list[tuple[float, str]] = []
    neutral_sentences: list[tuple[float, str]] = []
    bad_sentences: list[tuple[float, str]] = []
    score_parts: list[tuple[float, float]] = []

    for feat_name in detector.feature_names:
        val = team_feats.get(feat_name)
        ref_mean = detector.reference_mean.get(feat_name)
        if val is None or ref_mean is None:
            continue

        ref_std = max(detector.reference_std.get(feat_name, 1.0), 0.01)
        importance = detector.feature_importances.get(feat_name, 0.0)
        z = compute_z_score(val, ref_mean, ref_std)

        tmpl = NARRATIVE_TEMPLATES.get(feat_name)
        if not tmpl:
            continue

        direction = tmpl["direction"]
        tier = classify_tier(z, direction)

        if tier == "bad" and direction == "any":
            text = tmpl.get("bad_above" if z > 0 else "bad_below", tmpl.get("bad", ""))
        else:
            text = tmpl.get(tier, "")

        if not text:
            continue

        bucket = {"good": good_sentences, "neutral": neutral_sentences, "bad": bad_sentences}
        bucket[tier].append((importance, text))

        if direction == "below":
            feat_score = 75 + z * 10
        elif direction == "above":
            feat_score = 75 - z * 10
        else:
            feat_score = 85 - abs(z) * 12
        score_parts.append((max(0, min(100, feat_score)), importance))

    if score_parts:
        total_w = sum(w for _, w in score_parts)
        score = sum(s * w for s, w in score_parts) / total_w if total_w > 0 else 50
    else:
        score = 50.0
    score = max(0, min(100, round(score)))

    for group in (good_sentences, neutral_sentences, bad_sentences):
        group.sort(key=lambda x: x[0], reverse=True)

    parts: list[str] = []

    if score >= 80:
        parts.append(_SCORE_OVERVIEWS["excellent"])
    elif score >= 60:
        parts.append(_SCORE_OVERVIEWS["good"])
    elif score >= 40:
        parts.append(_SCORE_OVERVIEWS["average"])
    else:
        parts.append(_SCORE_OVERVIEWS["poor"])

    for _, text in good_sentences[:2]:
        parts.append(text)

    if bad_sentences:
        parts.append("No entanto, foram observados pontos críticos:")
        for _, text in bad_sentences:
            parts.append(f"• {text}")
    elif neutral_sentences:
        for _, text in neutral_sentences[:1]:
            parts.append(text)

    return " ".join(parts), float(score)
