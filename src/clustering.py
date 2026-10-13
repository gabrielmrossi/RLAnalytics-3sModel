"""
clustering.py — Clusterização tática comportamental não supervisionada (K-Means + PCA).
"""

from typing import Any, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from config import CLUSTER_PALETTE, DARK_BG, DARK_PAPER
from visualization.interactive import build_elbow_chart, build_pca_clusters_chart
from visualization.static import plot_elbow_static, plot_pca_clusters_static, save_figure


class TacticalClusterKMeans:
    """Clusterizacao comportamental nao-supervisionada com K-Means.

    Agrupa amostras de telemetria (times/replays) em perfis taticos
    sem utilizar labels.
    """

    _DARK_BG = DARK_BG
    _DARK_PAPER = DARK_PAPER
    _CLUSTER_PALETTE = CLUSTER_PALETTE

    def __init__(self, n_clusters: int = 3, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init="auto",
        )
        self.labels_: np.ndarray | None = None
        self.silhouette_: float | None = None
        self.is_fitted = False

    def compute_elbow(
        self,
        X: np.ndarray,
        k_range: range = range(2, 9),
    ) -> tuple[list[int], list[float]]:
        """Calcula inércias para cada k do range informado (retorna dados puros sem gráficos)."""
        X_scaled = self.scaler.fit_transform(X)
        inertias: list[float] = []
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init="auto")
            km.fit(X_scaled)
            inertias.append(float(km.inertia_))
        return list(k_range), inertias

    def elbow_analysis(
        self,
        X: np.ndarray,
        k_range: range = range(2, 9),
        save_path: str | None = None,
    ) -> go.Figure:
        """Calcula inércia para cada k e retorna figura interativa Plotly."""
        k_vals, inertias = self.compute_elbow(X, k_range=k_range)

        if save_path:
            fig_mpl = plot_elbow_static(k_vals, inertias)
            save_figure(fig_mpl, save_path)

        return build_elbow_chart(k_vals, inertias)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Ajusta scaler + KMeans e retorna rotulos de cluster."""
        X_scaled = self.scaler.fit_transform(X)
        self.labels_ = self.kmeans.fit_predict(X_scaled)
        self.is_fitted = True

        n_unique = len(set(self.labels_))
        if n_unique >= 2:
            max_silhouette_samples = 10_000
            if len(X_scaled) > max_silhouette_samples:
                rng = np.random.RandomState(self.random_state)
                idx = rng.choice(len(X_scaled), max_silhouette_samples, replace=False)
                self.silhouette_ = float(silhouette_score(X_scaled[idx], self.labels_[idx]))
            else:
                self.silhouette_ = float(silhouette_score(X_scaled, self.labels_))
        else:
            self.silhouette_ = None

        return self.labels_

    def compute_pca(
        self,
        X: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Projeta os dados em 2 componentes principais (PCA).

        Retorna:
            (X_pca, var_explained, loadings)
        """
        X_scaled = (
            self.scaler.transform(X)
            if self.is_fitted
            else StandardScaler().fit_transform(X)
        )
        pca = PCA(n_components=2, random_state=self.random_state)
        X_pca = pca.fit_transform(X_scaled)
        var_explained = pca.explained_variance_ratio_
        loadings = pca.components_.T
        return X_pca, var_explained, loadings

    def plot_pca_clusters(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        feature_names: Sequence[str] | None = None,
        save_path: str | None = None,
    ) -> go.Figure:
        """Projeta features em 2D via PCA e plota clusters coloridos."""
        X_pca, var_explained, loadings = self.compute_pca(X)

        if save_path:
            fig_mpl = plot_pca_clusters_static(
                X_pca, labels, var_explained, self._CLUSTER_PALETTE
            )
            save_figure(fig_mpl, save_path)

        return build_pca_clusters_chart(
            X_pca,
            labels,
            var_explained,
            cluster_palette=self._CLUSTER_PALETTE,
            feature_names=feature_names,
            loadings=loadings if feature_names is not None else None,
        )

    def cluster_summary(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
        feature_names: list[str],
    ) -> pd.DataFrame:
        """Gera DataFrame resumo com media de cada feature por cluster."""
        df_work = df[feature_names].copy()
        df_work["cluster"] = labels

        summary = df_work.groupby("cluster")[feature_names].mean().T
        summary.columns = [f"Cluster {c}" for c in summary.columns]
        summary = summary.round(2)
        return summary
