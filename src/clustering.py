"""
clustering.py — Clusterização tática comportamental não supervisionada (K-Means + PCA).
"""

import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from config import CLUSTER_PALETTE, DARK_BG, DARK_PAPER


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

    def elbow_analysis(
        self,
        X: np.ndarray,
        k_range: range = range(2, 9),
        save_path: str = "data/processed/kmeans_elbow.png",
    ) -> go.Figure:
        """Calcula inercia para cada k e gera a curva do cotovelo."""
        X_scaled = self.scaler.fit_transform(X)

        inertias: list[float] = []
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init="auto")
            km.fit(X_scaled)
            inertias.append(float(km.inertia_))

        # Matplotlib/Seaborn (salvar PNG)
        sns.set_theme(style="whitegrid")
        fig_mpl, ax = plt.subplots(figsize=(8, 5))
        ax.plot(list(k_range), inertias, "o-", color="#e94560", linewidth=2, markersize=8)
        ax.set_xlabel("Numero de Clusters (k)", fontsize=12)
        ax.set_ylabel("Inercia (WCSS)", fontsize=12)
        ax.set_title("Curva do Cotovelo - K-Means", fontsize=14, fontweight="bold")
        ax.set_xticks(list(k_range))
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig_mpl.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig_mpl)

        # Plotly (tema escuro)
        fig_plotly = go.Figure()
        fig_plotly.add_trace(go.Scatter(
            x=list(k_range), y=inertias,
            mode="lines+markers",
            line=dict(color="#e94560", width=3),
            marker=dict(size=10, color="#e94560"),
            name="Inercia",
            hovertemplate="k=%{x}<br>Inercia=%{y:,.0f}<extra></extra>",
        ))
        fig_plotly.update_layout(
            title=dict(text="Curva do Cotovelo - K-Means", font=dict(size=16)),
            xaxis=dict(title="Numero de Clusters (k)", dtick=1, gridcolor="#333333"),
            yaxis=dict(title="Inercia (WCSS)", gridcolor="#333333"),
            plot_bgcolor=self._DARK_BG,
            paper_bgcolor=self._DARK_PAPER,
            font=dict(color="white"),
            height=450,
        )

        return fig_plotly

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

    def plot_pca_clusters(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        feature_names: list[str] | None = None,
        save_path: str = "data/processed/kmeans_pca.png",
    ) -> go.Figure:
        """Projeta features em 2D via PCA e plota clusters coloridos."""
        X_scaled = self.scaler.transform(X) if self.is_fitted else StandardScaler().fit_transform(X)

        pca = PCA(n_components=2, random_state=self.random_state)
        X_pca = pca.fit_transform(X_scaled)

        var_explained = pca.explained_variance_ratio_
        pc1_label = f"PC1 ({var_explained[0]:.1%} var.)"
        pc2_label = f"PC2 ({var_explained[1]:.1%} var.)"
        unique_labels = sorted(set(labels))

        # Matplotlib (salvar PNG)
        sns.set_theme(style="whitegrid")
        fig_mpl, ax = plt.subplots(figsize=(9, 6))
        for cl in unique_labels:
            mask = labels == cl
            color = self._CLUSTER_PALETTE[cl % len(self._CLUSTER_PALETTE)]
            ax.scatter(
                X_pca[mask, 0], X_pca[mask, 1],
                c=color, label=f"Cluster {cl}", alpha=0.7, s=60, edgecolors="white", linewidth=0.5,
            )
        ax.set_xlabel(pc1_label, fontsize=12)
        ax.set_ylabel(pc2_label, fontsize=12)
        ax.set_title("Clusters Taticos (PCA 2D)", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig_mpl.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig_mpl)

        # Plotly
        fig_plotly = go.Figure()
        for cl in unique_labels:
            mask = labels == cl
            color = self._CLUSTER_PALETTE[cl % len(self._CLUSTER_PALETTE)]
            fig_plotly.add_trace(go.Scatter(
                x=X_pca[mask, 0], y=X_pca[mask, 1],
                mode="markers",
                marker=dict(size=8, color=color, opacity=0.75, line=dict(width=0.5, color="white")),
                name=f"Cluster {cl}",
                hovertemplate=f"Cluster {cl}<br>{pc1_label}: %{{x:.2f}}<br>{pc2_label}: %{{y:.2f}}<extra></extra>",
            ))

        fig_plotly.update_layout(
            title=dict(text="Clusters Taticos (PCA 2D)", font=dict(size=16)),
            xaxis=dict(title=pc1_label, gridcolor="#333333", zeroline=False),
            yaxis=dict(title=pc2_label, gridcolor="#333333", zeroline=False),
            plot_bgcolor=self._DARK_BG,
            paper_bgcolor=self._DARK_PAPER,
            font=dict(color="white"),
            height=550,
            legend=dict(bgcolor="rgba(30,30,30,0.8)", bordercolor="#555555", borderwidth=1),
        )

        if feature_names is not None:
            loadings = pca.components_.T
            scale = max(abs(X_pca[:, 0]).max(), abs(X_pca[:, 1]).max()) * 0.8
            load_scale = max(abs(loadings).max(), 1e-9)

            for i, name in enumerate(feature_names):
                lx = loadings[i, 0] / load_scale * scale
                ly = loadings[i, 1] / load_scale * scale
                fig_plotly.add_annotation(
                    x=lx, y=ly, text=name,
                    showarrow=True, arrowhead=2, arrowsize=1,
                    arrowcolor="#aaaaaa", font=dict(size=9, color="#cccccc"),
                    ax=0, ay=0,
                )

        return fig_plotly

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
