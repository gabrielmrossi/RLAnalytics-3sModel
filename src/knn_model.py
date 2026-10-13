"""
knn_model.py -- Modelos de classificacao e clusterizacao para telemetria de RL.

Pipeline:
    1. Carrega dataset de features (de feature_engineering.py)
    2. KNN com Leave-One-Out cross-validation
    3. Decision Tree com regras legiveis
    4. Random Forest com importancia nativa (MDI)
    5. K-Means para clusterizacao comportamental nao-supervisionada
    6. Benchmarking comparativo (StratifiedKFold cv=5)
    7. Detector de erros de posicionamento por limiares
"""

import json
import os
from pathlib import Path
from typing import Any

import joblib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Backend nao-interativo para servidores/CI
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text


# Colunas de features (excluindo metadados)
_META_COLS = {"label", "replay_name", "team", "team_label"}

FEATURE_NAMES = [
    "avg_boost_pct",
    "avg_dist_to_ball",
    "avg_speed",
    "ball_chase_ratio",
    "defensive_coverage_pct",
    "min_player_dist_to_ball",
    "team_spread_mean",
    "team_spread_std",
    "triangle_area_mean",
    "triangle_area_std",
]


def load_dataset(csv_path: str = "data/processed/features_dataset.csv") -> pd.DataFrame:
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


# ──────────────────────────────────────────────
#  KNN Classifier
# ──────────────────────────────────────────────

class RankClassifierKNN:
    """Classificador KNN para distinguir replays higher vs lower."""

    def __init__(self, n_neighbors: int = 3):
        self.n_neighbors = n_neighbors
        self.scaler = StandardScaler()
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.is_trained = False

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Treina o modelo com todos os dados."""
        X_scaled = self.scaler.fit_transform(X)
        self.knn.fit(X_scaled, y)
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prediz labels para novas amostras."""
        X_scaled = self.scaler.transform(X)
        return self.knn.predict(X_scaled)

    def evaluate_loo(
        self, X: np.ndarray, y: np.ndarray
    ) -> dict[str, Any]:
        """Leave-One-Out cross-validation.

        Returns:
            Dict com: accuracy, confusion_matrix, classification_report,
            predictions (lista de dicts com indice, real, previsto).
        """
        loo = LeaveOneOut()
        y_pred = np.empty_like(y)

        for train_idx, test_idx in loo.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            knn = KNeighborsClassifier(n_neighbors=min(self.n_neighbors, len(X_train)))
            knn.fit(X_train_s, y_train)
            y_pred[test_idx] = knn.predict(X_test_s)

        acc = accuracy_score(y, y_pred)
        cm = confusion_matrix(y, y_pred, labels=["higher", "lower"])
        report = classification_report(y, y_pred, labels=["higher", "lower"], output_dict=True)

        predictions = []
        for i in range(len(y)):
            predictions.append({
                "index": i,
                "actual": str(y[i]),
                "predicted": str(y_pred[i]),
                "correct": y[i] == y_pred[i],
            })

        return {
            "accuracy": float(acc),
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "predictions": predictions,
        }

    def feature_importance_loo(
        self, X: np.ndarray, y: np.ndarray, feature_names: list[str]
    ) -> dict[str, float]:
        """Calcula importancia de features via ablation (remover 1 por vez).

        Retorna dict feature_name -> queda de accuracy ao remover.
        Queda maior = feature mais importante.
        """
        baseline = self.evaluate_loo(X, y)["accuracy"]
        importance = {}

        for i, name in enumerate(feature_names):
            # Remover coluna i
            X_reduced = np.delete(X, i, axis=1)
            result = self.evaluate_loo(X_reduced, y)
            drop = baseline - result["accuracy"]
            importance[name] = round(drop, 4)

        # Ordenar por importancia descendente
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))


# ──────────────────────────────────────────────
#  Decision Tree Classifier
# ──────────────────────────────────────────────

class RankClassifierTree:
    """Decision Tree para classificacao + geracao de regras legiveis."""

    def __init__(self, max_depth: int = 3):
        self.max_depth = max_depth
        self.scaler = StandardScaler()
        self.tree = DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=42,
        )
        self.feature_names: list[str] = []
        self.is_trained = False

    def train(self, X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> None:
        """Treina a arvore de decisao."""
        X_scaled = self.scaler.fit_transform(X)
        self.tree.fit(X_scaled, y)
        self.feature_names = feature_names
        self.is_trained = True

    def evaluate_loo(
        self, X: np.ndarray, y: np.ndarray
    ) -> dict[str, Any]:
        """Leave-One-Out cross-validation para a Decision Tree."""
        loo = LeaveOneOut()
        y_pred = np.empty_like(y)

        for train_idx, test_idx in loo.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                random_state=42,
            )
            tree.fit(X_train_s, y_train)
            y_pred[test_idx] = tree.predict(X_test_s)

        acc = accuracy_score(y, y_pred)
        cm = confusion_matrix(y, y_pred, labels=["higher", "lower"])
        report = classification_report(y, y_pred, labels=["higher", "lower"], output_dict=True)

        predictions = []
        for i in range(len(y)):
            predictions.append({
                "index": i,
                "actual": str(y[i]),
                "predicted": str(y_pred[i]),
                "correct": y[i] == y_pred[i],
            })

        return {
            "accuracy": float(acc),
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "predictions": predictions,
        }

    def get_rules(self) -> str:
        """Exporta regras da arvore em texto legivel."""
        if not self.is_trained:
            return "Modelo nao treinado."
        return export_text(
            self.tree,
            feature_names=self.feature_names,
            class_names=["higher", "lower"],
        )

    def get_feature_importances(self) -> dict[str, float]:
        """Retorna importancia de cada feature segundo a arvore."""
        if not self.is_trained:
            return {}
        importances = self.tree.feature_importances_
        result = {}
        for name, imp in zip(self.feature_names, importances):
            result[name] = round(float(imp), 4)
        return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))


# ──────────────────────────────────────────────
#  Random Forest Classifier
# ──────────────────────────────────────────────

class RankClassifierRandomForest:
    """Random Forest para classificacao de rank + importancia nativa de features."""

    def __init__(self, n_estimators: int = 100, max_depth: int | None = None, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )
        self.feature_names: list[str] = []
        self.is_trained = False

    def train(self, X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> None:
        """Treina o Random Forest com todos os dados."""
        X_scaled = self.scaler.fit_transform(X)
        self.rf.fit(X_scaled, y)
        self.feature_names = feature_names
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prediz labels para novas amostras."""
        X_scaled = self.scaler.transform(X)
        return self.rf.predict(X_scaled)

    def evaluate_loo(
        self, X: np.ndarray, y: np.ndarray
    ) -> dict[str, Any]:
        """Leave-One-Out cross-validation para o Random Forest."""
        loo = LeaveOneOut()
        y_pred = np.empty_like(y)

        for train_idx, test_idx in loo.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            rf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
            )
            rf.fit(X_train_s, y_train)
            y_pred[test_idx] = rf.predict(X_test_s)

        acc = accuracy_score(y, y_pred)
        cm = confusion_matrix(y, y_pred, labels=["higher", "lower"])
        report = classification_report(y, y_pred, labels=["higher", "lower"], output_dict=True)

        predictions = []
        for i in range(len(y)):
            predictions.append({
                "index": i,
                "actual": str(y[i]),
                "predicted": str(y_pred[i]),
                "correct": y[i] == y_pred[i],
            })

        return {
            "accuracy": float(acc),
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "predictions": predictions,
        }

    def get_feature_importances(self) -> dict[str, float]:
        """Retorna importancia MDI (Mean Decrease in Impurity) de cada feature."""
        if not self.is_trained:
            return {}
        importances = self.rf.feature_importances_
        result = {}
        for name, imp in zip(self.feature_names, importances):
            result[name] = round(float(imp), 4)
        return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))

    def plot_feature_importances(
        self,
        save_path: str | None = None,
        title: str = "Importancia das Features (Random Forest)",
    ) -> plt.Figure:
        """Gera grafico de barras horizontal com Seaborn/Matplotlib.

        Args:
            save_path: Caminho para salvar PNG (opcional).
            title: Titulo do grafico.

        Returns:
            Objeto Figure do Matplotlib.
        """
        imp_dict = self.get_feature_importances()
        if not imp_dict:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Modelo nao treinado", ha="center", va="center")
            return fig

        imp_df = pd.DataFrame(
            {"feature": list(imp_dict.keys()), "importance": list(imp_dict.values())}
        )

        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(10, max(4, len(imp_df) * 0.5)))

        palette = sns.color_palette("viridis", len(imp_df))
        barplot = sns.barplot(
            data=imp_df, x="importance", y="feature",
            palette=palette, ax=ax,
        )

        # Anotacao percentual ao lado de cada barra
        for p in barplot.patches:
            val = p.get_width()
            ax.annotate(
                f"{val:.2%}",
                (val, p.get_y() + p.get_height() / 2.0),
                ha="left", va="center", fontsize=10, color="black",
                xytext=(5, 0), textcoords="offset points",
            )

        ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
        ax.set_xlabel("Importancia Relativa (MDI)", fontsize=11)
        ax.set_ylabel("Feature de Telemetria", fontsize=11)

        max_val = imp_df["importance"].max() if not imp_df.empty else 0.1
        ax.set_xlim(0, max_val * 1.2)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig


# ──────────────────────────────────────────────
#  K-Means Tactical Clustering (nao supervisionado)
# ──────────────────────────────────────────────

class TacticalClusterKMeans:
    """Clusterizacao comportamental nao-supervisionada com K-Means.

    Agrupa amostras de telemetria (times/replays) em perfis taticos
    sem utilizar labels. Otimizado para volumes maiores vindos da API
    do Ballchasing — evita abordagens iterativas como LOO.
    """

    _DARK_BG = "#1e1e1e"
    _DARK_PAPER = "#1e1e1e"
    _CLUSTER_PALETTE = [
        "#e94560", "#2ecc71", "#3498db", "#e67e22",
        "#9b59b6", "#1abc9c", "#f39c12", "#e74c3c",
    ]

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

    # ── Elbow Analysis ──────────────────────────

    def elbow_analysis(
        self,
        X: np.ndarray,
        k_range: range = range(2, 9),
        save_path: str = "data/processed/kmeans_elbow.png",
    ) -> go.Figure:
        """Calcula inercia para cada k e gera a curva do cotovelo.

        Args:
            X: Matriz de features (n_samples, n_features).
            k_range: Faixa de valores de k a testar.
            save_path: Caminho para salvar o grafico Matplotlib (PNG).

        Returns:
            Figura Plotly com tema escuro para renderizacao no Streamlit.
        """
        X_scaled = self.scaler.fit_transform(X)

        inertias: list[float] = []
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init="auto")
            km.fit(X_scaled)
            inertias.append(float(km.inertia_))

        # ── Matplotlib/Seaborn (salvar PNG) ──
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

        # ── Plotly (tema escuro para Streamlit) ──
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
            xaxis=dict(
                title="Numero de Clusters (k)",
                dtick=1,
                gridcolor="#333333",
            ),
            yaxis=dict(
                title="Inercia (WCSS)",
                gridcolor="#333333",
            ),
            plot_bgcolor=self._DARK_BG,
            paper_bgcolor=self._DARK_PAPER,
            font=dict(color="white"),
            height=450,
        )

        return fig_plotly

    # ── Fit + Predict ───────────────────────────

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Ajusta scaler + KMeans e retorna rotulos de cluster.

        Calcula e armazena silhouette_score automaticamente.
        Para datasets grandes (>10k amostras), usa subamostragem
        no calculo do silhouette para manter performance.

        Args:
            X: Matriz de features (n_samples, n_features).

        Returns:
            Array 1-D com o rotulo de cluster de cada amostra.
        """
        X_scaled = self.scaler.fit_transform(X)
        self.labels_ = self.kmeans.fit_predict(X_scaled)
        self.is_fitted = True

        # Silhouette score (requer >= 2 clusters com amostras)
        n_unique = len(set(self.labels_))
        if n_unique >= 2:
            max_silhouette_samples = 10_000
            if len(X_scaled) > max_silhouette_samples:
                rng = np.random.RandomState(self.random_state)
                idx = rng.choice(len(X_scaled), max_silhouette_samples, replace=False)
                self.silhouette_ = float(silhouette_score(
                    X_scaled[idx], self.labels_[idx],
                ))
            else:
                self.silhouette_ = float(silhouette_score(X_scaled, self.labels_))
        else:
            self.silhouette_ = None

        return self.labels_

    # ── PCA 2-D Visualization ───────────────────

    def plot_pca_clusters(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        feature_names: list[str] | None = None,
        save_path: str = "data/processed/kmeans_pca.png",
    ) -> go.Figure:
        """Projeta features em 2D via PCA e plota clusters coloridos.

        Args:
            X: Matriz de features original (nao-escalada).
            labels: Array de rotulos de cluster.
            feature_names: Nomes das features (para mostrar loadings).
            save_path: Caminho para salvar PNG.

        Returns:
            Figura Plotly interativa com tema escuro.
        """
        X_scaled = self.scaler.transform(X) if self.is_fitted else StandardScaler().fit_transform(X)

        pca = PCA(n_components=2, random_state=self.random_state)
        X_pca = pca.fit_transform(X_scaled)

        var_explained = pca.explained_variance_ratio_
        pc1_label = f"PC1 ({var_explained[0]:.1%} var.)"
        pc2_label = f"PC2 ({var_explained[1]:.1%} var.)"

        unique_labels = sorted(set(labels))

        # ── Matplotlib (salvar PNG) ──
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

        # ── Plotly (interativo, tema escuro) ──
        fig_plotly = go.Figure()
        for cl in unique_labels:
            mask = labels == cl
            color = self._CLUSTER_PALETTE[cl % len(self._CLUSTER_PALETTE)]
            fig_plotly.add_trace(go.Scatter(
                x=X_pca[mask, 0], y=X_pca[mask, 1],
                mode="markers",
                marker=dict(size=8, color=color, opacity=0.75,
                            line=dict(width=0.5, color="white")),
                name=f"Cluster {cl}",
                hovertemplate=(
                    f"Cluster {cl}<br>"
                    f"{pc1_label}: %{{x:.2f}}<br>"
                    f"{pc2_label}: %{{y:.2f}}<extra></extra>"
                ),
            ))

        fig_plotly.update_layout(
            title=dict(
                text="Clusters Taticos (PCA 2D)",
                font=dict(size=16),
            ),
            xaxis=dict(title=pc1_label, gridcolor="#333333", zeroline=False),
            yaxis=dict(title=pc2_label, gridcolor="#333333", zeroline=False),
            plot_bgcolor=self._DARK_BG,
            paper_bgcolor=self._DARK_PAPER,
            font=dict(color="white"),
            height=550,
            legend=dict(
                bgcolor="rgba(30,30,30,0.8)",
                bordercolor="#555555",
                borderwidth=1,
            ),
        )

        # Anotar loadings das features no grafico se disponiveis
        if feature_names is not None:
            loadings = pca.components_.T  # (n_features, 2)
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

    # ── Resumo dos Clusters ─────────────────────

    def cluster_summary(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
        feature_names: list[str],
    ) -> pd.DataFrame:
        """Gera DataFrame resumo com media de cada feature por cluster.

        Args:
            df: DataFrame original com as features.
            labels: Rotulos de cluster.
            feature_names: Colunas de features.

        Returns:
            DataFrame pivotado: linhas = features, colunas = clusters.
        """
        df_work = df[feature_names].copy()
        df_work["cluster"] = labels

        summary = df_work.groupby("cluster")[feature_names].mean().T
        summary.columns = [f"Cluster {c}" for c in summary.columns]
        summary = summary.round(2)
        return summary


# ──────────────────────────────────────────────
#  Model Benchmarking (comparativo supervisionado)
# ──────────────────────────────────────────────

class ModelBenchmarking:
    """Avaliacao comparativa dos modelos supervisionados via StratifiedKFold.

    Compara KNN, Decision Tree e Random Forest nos mesmos folds,
    extraindo Acuracia, F1, Precisao e Recall ponderados.
    """

    _DARK_BG = "#1e1e1e"
    _DARK_PAPER = "#1e1e1e"
    _MODEL_COLORS = {
        "KNN": "#e94560",
        "Decision Tree": "#e67e22",
        "Random Forest": "#2ecc71",
    }
    _LABELS = ["higher", "lower"]

    def __init__(self, cv: int = 5, random_state: int = 42):
        self.cv = cv
        self.random_state = random_state
        self.results_: dict[str, dict[str, Any]] = {}
        self.summary_df_: pd.DataFrame | None = None
        self.confusion_matrices_: dict[str, np.ndarray] = {}

    # ── Avaliacao comparativa ───────────────────

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        knn_k: int = 3,
        tree_depth: int = 3,
        rf_n_estimators: int = 100,
    ) -> pd.DataFrame:
        """Executa StratifiedKFold nos 3 modelos e retorna tabela resumo.

        Args:
            X: Matriz de features.
            y: Labels (strings: 'higher', 'lower').
            knn_k: Numero de vizinhos para o KNN.
            tree_depth: Profundidade maxima da Decision Tree.
            rf_n_estimators: Numero de arvores do Random Forest.

        Returns:
            DataFrame com medias e desvios de cada metrica por modelo,
            ordenado por F1-Score descendente.
        """
        models = {
            "KNN": lambda: KNeighborsClassifier(
                n_neighbors=min(knn_k, len(X) - 1)
            ),
            "Decision Tree": lambda: DecisionTreeClassifier(
                max_depth=tree_depth, random_state=self.random_state
            ),
            "Random Forest": lambda: RandomForestClassifier(
                n_estimators=rf_n_estimators, random_state=self.random_state
            ),
        }

        skf = StratifiedKFold(
            n_splits=min(self.cv, len(X)),
            shuffle=True,
            random_state=self.random_state,
        )

        for model_name, model_factory in models.items():
            fold_metrics = {
                "accuracy": [], "f1": [], "precision": [], "recall": [],
            }
            # Acumular predicoes de todos os folds para a CM global
            all_y_true, all_y_pred = [], []

            for train_idx, test_idx in skf.split(X, y):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_test_s = scaler.transform(X_test)

                clf = model_factory()
                clf.fit(X_train_s, y_train)
                y_pred = clf.predict(X_test_s)

                fold_metrics["accuracy"].append(accuracy_score(y_test, y_pred))
                fold_metrics["f1"].append(f1_score(
                    y_test, y_pred, labels=self._LABELS, average="weighted", zero_division=0,
                ))
                fold_metrics["precision"].append(precision_score(
                    y_test, y_pred, labels=self._LABELS, average="weighted", zero_division=0,
                ))
                fold_metrics["recall"].append(recall_score(
                    y_test, y_pred, labels=self._LABELS, average="weighted", zero_division=0,
                ))

                all_y_true.extend(y_test)
                all_y_pred.extend(y_pred)

            self.results_[model_name] = {
                k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                for k, v in fold_metrics.items()
            }

            # Matriz de confusao normalizada (agregada de todos os folds)
            cm = confusion_matrix(all_y_true, all_y_pred, labels=self._LABELS)
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            cm_norm = np.nan_to_num(cm_norm, nan=0.0)
            self.confusion_matrices_[model_name] = cm_norm

        # Montar DataFrame resumo
        rows = []
        for model_name, metrics in self.results_.items():
            rows.append({
                "Modelo": model_name,
                "Acuracia": f"{metrics['accuracy']['mean']:.1%} ± {metrics['accuracy']['std']:.1%}",
                "F1-Score": f"{metrics['f1']['mean']:.1%} ± {metrics['f1']['std']:.1%}",
                "Precisao": f"{metrics['precision']['mean']:.1%} ± {metrics['precision']['std']:.1%}",
                "Recall": f"{metrics['recall']['mean']:.1%} ± {metrics['recall']['std']:.1%}",
                "_f1_mean": metrics["f1"]["mean"],  # para ordenacao
            })

        self.summary_df_ = (
            pd.DataFrame(rows)
            .sort_values("_f1_mean", ascending=False)
            .drop(columns=["_f1_mean"])
            .reset_index(drop=True)
        )
        return self.summary_df_

    # ── Matrizes de confusao normalizadas ───────

    def plot_confusion_matrices(
        self,
        save_path: str = "data/processed/models_comparison_cm.png",
    ) -> go.Figure:
        """Gera 3 matrizes de confusao normalizadas lado a lado.

        Salva PNG via Matplotlib/Seaborn e retorna Plotly Figure (tema escuro).

        Args:
            save_path: Caminho para salvar o PNG.

        Returns:
            go.Figure com 3 subplots para renderizacao no Streamlit.
        """
        if not self.confusion_matrices_:
            raise RuntimeError("Execute evaluate() antes de plotar.")

        model_names = list(self.confusion_matrices_.keys())

        # ── Matplotlib / Seaborn (PNG) ──
        sns.set_theme(style="white")
        fig_mpl, axes = plt.subplots(1, len(model_names), figsize=(5 * len(model_names), 4))
        if len(model_names) == 1:
            axes = [axes]

        for ax, name in zip(axes, model_names):
            cm_norm = self.confusion_matrices_[name]
            sns.heatmap(
                cm_norm, annot=True, fmt=".2%", cmap="YlOrRd",
                xticklabels=self._LABELS, yticklabels=self._LABELS,
                ax=ax, cbar=False, square=True,
                annot_kws={"fontsize": 12},
            )
            ax.set_title(name, fontsize=13, fontweight="bold")
            ax.set_xlabel("Previsto", fontsize=10)
            ax.set_ylabel("Real", fontsize=10)

        plt.suptitle(
            "Matrizes de Confusao Normalizadas (StratifiedKFold)",
            fontsize=15, fontweight="bold", y=1.02,
        )
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig_mpl.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig_mpl)

        # ── Plotly (tema escuro, 3 subplots) ──
        fig_plotly = make_subplots(
            rows=1, cols=len(model_names),
            subplot_titles=model_names,
            horizontal_spacing=0.08,
        )

        for i, name in enumerate(model_names, start=1):
            cm_norm = self.confusion_matrices_[name]
            color = self._MODEL_COLORS.get(name, "#e94560")

            text_matrix = [[f"{cm_norm[r][c]:.1%}" for c in range(len(self._LABELS))]
                           for r in range(len(self._LABELS))]

            fig_plotly.add_trace(
                go.Heatmap(
                    z=cm_norm,
                    x=[f"Pred: {l}" for l in self._LABELS],
                    y=[f"Real: {l}" for l in self._LABELS],
                    text=text_matrix,
                    texttemplate="%{text}",
                    colorscale=[[0, self._DARK_BG], [1, color]],
                    showscale=False,
                    hovertemplate="Real: %{y}<br>Previsto: %{x}<br>Taxa: %{text}<extra></extra>",
                ),
                row=1, col=i,
            )

        fig_plotly.update_layout(
            title=dict(
                text="Matrizes de Confusao Normalizadas (StratifiedKFold)",
                font=dict(size=16),
            ),
            plot_bgcolor=self._DARK_BG,
            paper_bgcolor=self._DARK_PAPER,
            font=dict(color="white"),
            height=400,
            margin=dict(l=60, r=30, t=80, b=60),
        )

        return fig_plotly


# ──────────────────────────────────────────────
#  Detector de Erros de Posicionamento
# ──────────────────────────────────────────────

class PositioningErrorDetector:
    """Detecta erros de posicionamento comparando com perfil de rank alto.

    Usa limiares baseados em media +/- 2*std do perfil de referencia (higher).
    """

    def __init__(self):
        self.reference_mean: dict[str, float] = {}
        self.reference_std: dict[str, float] = {}
        self.is_built = False

    def build_reference(self, df: pd.DataFrame) -> None:
        """Constroi perfil de referencia a partir dos replays 'higher'."""
        higher = df[df["label"] == "higher"]
        if higher.empty:
            return

        feature_cols = [c for c in higher.columns if c not in _META_COLS]

        for col in feature_cols:
            self.reference_mean[col] = float(higher[col].mean())
            self.reference_std[col] = float(higher[col].std()) if len(higher) > 1 else 0.0

        self.is_built = True

    def detect_errors(self, team_features: dict[str, float]) -> list[dict]:
        """Detecta erros de posicionamento em um time comparando com referencia.

        Args:
            team_features: Features de um time (output de extract_team_features).

        Returns:
            Lista de erros detectados, cada um com tipo, severidade e descricao.
        """
        if not self.is_built:
            return []

        errors = []

        # 1. Ball-chasing: ratio muito acima da referencia
        if "ball_chase_ratio" in team_features and "ball_chase_ratio" in self.reference_mean:
            ref_mean = self.reference_mean["ball_chase_ratio"]
            ref_std = max(self.reference_std.get("ball_chase_ratio", 5.0), 1.0)
            val = team_features["ball_chase_ratio"]
            z_score = (val - ref_mean) / ref_std

            if z_score > 1.5:
                errors.append({
                    "type": "BALL_CHASING",
                    "severity": "alta" if z_score > 2.5 else "media",
                    "value": val,
                    "reference": round(ref_mean, 1),
                    "description": (
                        f"Ball-chase ratio ({val:.1f}%) esta {z_score:.1f} desvios acima "
                        f"da referencia de rank alto ({ref_mean:.1f}%). "
                        f"Jogadores ficam muito proximos da bola ao mesmo tempo."
                    ),
                })

        # 2. Falta de cobertura defensiva
        if "defensive_coverage_pct" in team_features and "defensive_coverage_pct" in self.reference_mean:
            ref_mean = self.reference_mean["defensive_coverage_pct"]
            ref_std = max(self.reference_std.get("defensive_coverage_pct", 5.0), 1.0)
            val = team_features["defensive_coverage_pct"]
            z_score = (ref_mean - val) / ref_std  # Invertido: menor = pior

            if z_score > 1.5:
                errors.append({
                    "type": "NO_DEFENSIVE_COVER",
                    "severity": "alta" if z_score > 2.5 else "media",
                    "value": val,
                    "reference": round(ref_mean, 1),
                    "description": (
                        f"Cobertura defensiva ({val:.1f}%) esta {z_score:.1f} desvios abaixo "
                        f"da referencia ({ref_mean:.1f}%). "
                        f"Time fica sem defensores com frequencia."
                    ),
                })

        # 3. Cluster (triangulo pequeno demais)
        if "triangle_area_mean" in team_features and "triangle_area_mean" in self.reference_mean:
            ref_mean = self.reference_mean["triangle_area_mean"]
            ref_std = max(self.reference_std.get("triangle_area_mean", 100000.0), 1.0)
            val = team_features["triangle_area_mean"]
            z_score = (ref_mean - val) / ref_std  # Invertido: menor = pior

            if z_score > 1.5:
                errors.append({
                    "type": "CLUSTER",
                    "severity": "alta" if z_score > 2.5 else "media",
                    "value": val,
                    "reference": round(ref_mean, 1),
                    "description": (
                        f"Area media do triangulo ({val:.0f}) esta {z_score:.1f} desvios abaixo "
                        f"da referencia ({ref_mean:.0f}). "
                        f"Jogadores ficam muito agrupados, sem cobrir o campo."
                    ),
                })

        # 4. Dispersao muito inconsistente (std alto)
        if "team_spread_std" in team_features and "team_spread_std" in self.reference_mean:
            ref_mean = self.reference_mean["team_spread_std"]
            ref_std = max(self.reference_std.get("team_spread_std", 200.0), 1.0)
            val = team_features["team_spread_std"]
            z_score = (val - ref_mean) / ref_std

            if z_score > 1.5:
                errors.append({
                    "type": "INCONSISTENT_ROTATION",
                    "severity": "alta" if z_score > 2.5 else "media",
                    "value": val,
                    "reference": round(ref_mean, 1),
                    "description": (
                        f"Variacao da dispersao ({val:.0f}) esta {z_score:.1f} desvios acima "
                        f"da referencia ({ref_mean:.0f}). "
                        f"Rotacao inconsistente: hora agrupados, hora dispersos."
                    ),
                })

        # 5. Boost management (boost medio baixo)
        if "avg_boost_pct" in team_features and "avg_boost_pct" in self.reference_mean:
            ref_mean = self.reference_mean["avg_boost_pct"]
            ref_std = max(self.reference_std.get("avg_boost_pct", 5.0), 1.0)
            val = team_features["avg_boost_pct"]
            z_score = (ref_mean - val) / ref_std

            if z_score > 1.5:
                errors.append({
                    "type": "LOW_BOOST",
                    "severity": "media",
                    "value": val,
                    "reference": round(ref_mean, 1),
                    "description": (
                        f"Boost medio ({val:.1f}%) esta {z_score:.1f} desvios abaixo "
                        f"da referencia ({ref_mean:.1f}%). "
                        f"Time fica frequentemente sem boost disponivel."
                    ),
                })

        return errors


class ModelDrivenErrorDetector:
    """Detecta erros taticos usando o Random Forest treinado.

    Em vez de regras manuais com z-scores fixos, usa a importancia de cada
    feature (MDI do RF) para ponderar os desvios em relacao ao perfil de
    rank alto. Features que o modelo considera mais relevantes para
    classificar o rank geram erros com prioridade maior.

    Fluxo:
        1. Treina o RF com todo o dataset (X, y).
        2. Extrai feature_importances_ do RF.
        3. Calcula media e std de cada feature no grupo 'higher'.
        4. Para cada amostra, calcula z-scores ponderados pela importancia.
        5. Retorna erros ordenados pelo impacto (importance * |z-score|).
    """

    # Mapeamento de features para descricoes humanas
    _FEATURE_DESCRIPTIONS = {
        "ball_chase_ratio": {
            "name": "Ball Chasing",
            "direction": "above",  # valor alto = ruim
            "desc": "Jogadores ficam muito proximos da bola ao mesmo tempo.",
        },
        "defensive_coverage_pct": {
            "name": "Cobertura Defensiva",
            "direction": "below",  # valor baixo = ruim
            "desc": "Time fica sem defensores com frequencia.",
        },
        "triangle_area_mean": {
            "name": "Area do Triangulo",
            "direction": "below",
            "desc": "Jogadores ficam muito agrupados, sem cobrir o campo.",
        },
        "triangle_area_std": {
            "name": "Variacao do Triangulo",
            "direction": "above",
            "desc": "Formacao do time varia demais entre frames.",
        },
        "team_spread_mean": {
            "name": "Dispersao Media",
            "direction": "below",
            "desc": "Jogadores ficam muito proximos uns dos outros.",
        },
        "team_spread_std": {
            "name": "Rotacao Inconsistente",
            "direction": "above",
            "desc": "Rotacao inconsistente: hora agrupados, hora dispersos.",
        },
        "avg_boost_pct": {
            "name": "Gestao de Boost",
            "direction": "below",
            "desc": "Time fica frequentemente sem boost disponivel.",
        },
        "avg_dist_to_ball": {
            "name": "Distancia Media da Bola",
            "direction": "any",
            "desc": "Distancia media a bola fora do padrao de rank alto.",
        },
        "min_player_dist_to_ball": {
            "name": "Proximidade Minima da Bola",
            "direction": "any",
            "desc": "Distancia minima a bola fora do padrao de rank alto.",
        },
        "avg_speed": {
            "name": "Velocidade Media",
            "direction": "below",
            "desc": "Velocidade media dos jogadores abaixo do padrao.",
        },
    }

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        self.rf = RankClassifierRandomForest(
            n_estimators=n_estimators, random_state=random_state
        )
        self.reference_mean: dict[str, float] = {}
        self.reference_std: dict[str, float] = {}
        self.feature_importances: dict[str, float] = {}
        self.feature_names: list[str] = []
        self.is_built = False

    def build(self, df: pd.DataFrame, X: np.ndarray, y: np.ndarray,
              feature_names: list[str]) -> dict[str, float]:
        """Treina o RF e constroi perfil de referencia.

        Args:
            df: DataFrame completo com coluna 'label'.
            X: Matriz de features (scaled).
            y: Labels (higher/lower).
            feature_names: Nomes das features.

        Returns:
            Dict com importancias do RF.
        """
        self.feature_names = feature_names

        # 1. Treinar RF com todos os dados
        self.rf.train(X, y, feature_names)
        self.feature_importances = self.rf.get_feature_importances()

        # 2. Construir perfil de referencia (media/std do grupo higher)
        higher = df[df["label"] == "higher"]
        if higher.empty:
            return self.feature_importances

        for col in feature_names:
            if col in higher.columns:
                self.reference_mean[col] = float(higher[col].mean())
                self.reference_std[col] = float(higher[col].std()) if len(higher) > 1 else 0.0

        self.is_built = True
        return self.feature_importances

    def save(self, path: str = "data/processed/model_detector.joblib") -> None:
        """Serializa o detector treinado para disco."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            "rf_model": self.rf,
            "reference_mean": self.reference_mean,
            "reference_std": self.reference_std,
            "feature_importances": self.feature_importances,
            "feature_names": self.feature_names,
            "is_built": self.is_built,
        }
        joblib.dump(state, path)

    @classmethod
    def load(cls, path: str = "data/processed/model_detector.joblib") -> "ModelDrivenErrorDetector":
        """Carrega um detector previamente treinado do disco."""
        state = joblib.load(path)
        detector = cls.__new__(cls)
        detector.rf = state["rf_model"]
        detector.reference_mean = state["reference_mean"]
        detector.reference_std = state["reference_std"]
        detector.feature_importances = state["feature_importances"]
        detector.feature_names = state["feature_names"]
        detector.is_built = state["is_built"]
        return detector

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prediz o rank de cada amostra usando o RF treinado."""
        if not self.rf.is_trained:
            return np.array([])
        return self.rf.predict(X)

    def detect_errors(
        self, team_features: dict[str, float], z_threshold: float = 1.5
    ) -> list[dict]:
        """Detecta erros ponderados pela importancia do modelo.

        Args:
            team_features: Features de um time.
            z_threshold: Limiar minimo de z-score para reportar.

        Returns:
            Lista de erros ordenados por impacto (importance * |z_score|).
        """
        if not self.is_built:
            return []

        errors = []

        for feat_name in self.feature_names:
            if feat_name not in team_features or feat_name not in self.reference_mean:
                continue

            val = team_features[feat_name]
            ref_mean = self.reference_mean[feat_name]
            ref_std = max(self.reference_std.get(feat_name, 1.0), 0.01)
            importance = self.feature_importances.get(feat_name, 0.0)

            # Z-score: quanto essa feature desvia do padrao higher
            raw_z = (val - ref_mean) / ref_std

            # Verificar direcao do desvio
            feat_info = self._FEATURE_DESCRIPTIONS.get(feat_name, {})
            direction = feat_info.get("direction", "any")

            if direction == "above" and raw_z <= z_threshold:
                continue  # Valor alto = ruim, mas nao esta alto o bastante
            elif direction == "below" and -raw_z <= z_threshold:
                continue  # Valor baixo = ruim, mas nao esta baixo o bastante
            elif direction == "any" and abs(raw_z) <= z_threshold:
                continue

            # Impacto = importancia do modelo * magnitude do desvio
            impact = importance * abs(raw_z)

            # Severidade baseada no impacto ponderado
            if impact > 0.15 or abs(raw_z) > 2.5:
                severity = "alta"
            else:
                severity = "media"

            feat_display = feat_info.get("name", feat_name)
            feat_desc = feat_info.get("desc", f"Feature '{feat_name}' fora do padrao.")

            if direction == "above":
                desvio_text = f"{val:.1f} esta {raw_z:.1f}σ acima da ref. ({ref_mean:.1f})"
            elif direction == "below":
                desvio_text = f"{val:.1f} esta {abs(raw_z):.1f}σ abaixo da ref. ({ref_mean:.1f})"
            else:
                dir_word = "acima" if raw_z > 0 else "abaixo"
                desvio_text = f"{val:.1f} esta {abs(raw_z):.1f}σ {dir_word} da ref. ({ref_mean:.1f})"

            errors.append({
                "type": feat_display,
                "severity": severity,
                "value": val,
                "reference": round(ref_mean, 1),
                "z_score": round(raw_z, 2),
                "importance": round(importance, 4),
                "impact": round(impact, 4),
                "description": f"{desvio_text}. {feat_desc} [Peso RF: {importance:.1%}]",
            })

        # Ordenar por impacto (mais importante primeiro)
        errors.sort(key=lambda e: e["impact"], reverse=True)

        return errors


# ──────────────────────────────────────────────
#  Execucao Principal
# ──────────────────────────────────────────────

def run_full_pipeline(csv_path: str = "data/processed/features_dataset.csv") -> dict:
    """Roda pipeline completo: carrega dados, treina modelos, reporta resultados.

    Returns:
        Dict com todos os resultados.
    """
    print("=== Carregando dataset ===")
    df = load_dataset(csv_path)
    X, y, feature_names = get_X_y(df)
    print(f"  Amostras: {len(X)}, Features: {len(feature_names)}")
    print(f"  Labels: {dict(zip(*np.unique(y, return_counts=True)))}")

    # Ajustar k para nao ser maior que n_amostras - 1
    k = min(3, len(X) - 1)

    # ── KNN ──
    print(f"\n=== KNN (k={k}) com Leave-One-Out ===")
    knn = RankClassifierKNN(n_neighbors=k)
    knn_results = knn.evaluate_loo(X, y)
    knn.train(X, y)  # Treinar com todos para predicoes futuras

    print(f"  Accuracy: {knn_results['accuracy']:.1%}")
    print(f"  Confusion Matrix: {knn_results['confusion_matrix']}")
    print(f"  Predicoes:")
    for p in knn_results["predictions"]:
        status = "OK" if p["correct"] else "ERRO"
        row = df.iloc[p["index"]]
        print(f"    [{status}] {row['replay_name'][:12]}... (team {row['team_label']}) "
              f"real={p['actual']} pred={p['predicted']}")

    # Feature importance
    print(f"\n  Importancia das features (ablation):")
    importance = knn.feature_importance_loo(X, y, feature_names)
    for name, drop in importance.items():
        indicator = "***" if drop > 0.05 else ""
        print(f"    {name:30s}: drop = {drop:+.4f} {indicator}")

    # ── Decision Tree ──
    print(f"\n=== Decision Tree (depth={3}) com Leave-One-Out ===")
    tree = RankClassifierTree(max_depth=3)
    tree_results = tree.evaluate_loo(X, y)
    tree.train(X, y, feature_names)

    print(f"  Accuracy: {tree_results['accuracy']:.1%}")
    print(f"  Confusion Matrix: {tree_results['confusion_matrix']}")

    print(f"\n  Regras da arvore:")
    print(tree.get_rules())

    print(f"  Importancia (Gini):")
    for name, imp in tree.get_feature_importances().items():
        if imp > 0:
            print(f"    {name:30s}: {imp:.4f}")

    # ── Random Forest ──
    print(f"\n=== Random Forest (n_estimators=100) com Leave-One-Out ===")
    rf = RankClassifierRandomForest(n_estimators=100, random_state=42)
    rf_results = rf.evaluate_loo(X, y)
    rf.train(X, y, feature_names)

    print(f"  Accuracy: {rf_results['accuracy']:.1%}")
    print(f"  Confusion Matrix: {rf_results['confusion_matrix']}")

    print(f"\n  Importancia (MDI):")
    for name, imp in rf.get_feature_importances().items():
        if imp > 0.01:
            print(f"    {name:30s}: {imp:.4f}")

    # Salvar grafico de importancia
    plot_path = "data/processed/rf_feature_importance.png"
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    rf.plot_feature_importances(save_path=plot_path)
    print(f"  Grafico salvo em: {plot_path}")

    # ── K-Means Clustering ──
    print(f"\n=== K-Means Clustering (nao-supervisionado) ===")
    clusterer = TacticalClusterKMeans(n_clusters=3, random_state=42)
    clusterer.elbow_analysis(X, save_path="data/processed/kmeans_elbow.png")
    cluster_labels = clusterer.fit_predict(X)
    print(f"  Silhouette Score: {clusterer.silhouette_:.4f}" if clusterer.silhouette_ else "  Silhouette: N/A")
    print(f"  Distribuicao: {dict(zip(*np.unique(cluster_labels, return_counts=True)))}")

    clusterer.plot_pca_clusters(X, cluster_labels, feature_names,
                                save_path="data/processed/kmeans_pca.png")
    print(f"  Graficos salvos em data/processed/")

    summary = clusterer.cluster_summary(df, cluster_labels, feature_names)
    print(f"\n  Resumo por cluster:")
    print(summary.to_string())

    # ── Model Benchmarking ──
    print(f"\n=== Benchmarking Comparativo (StratifiedKFold cv=5) ===")
    bench = ModelBenchmarking(cv=5, random_state=42)
    bench_df = bench.evaluate(X, y, knn_k=k)
    print(bench_df.to_string(index=False))
    bench.plot_confusion_matrices(save_path="data/processed/models_comparison_cm.png")
    print(f"  Matrizes de confusao salvas em data/processed/models_comparison_cm.png")

    # ── Error Detection ──
    print(f"\n=== Deteccao de Erros ===")
    detector = PositioningErrorDetector()
    detector.build_reference(df)

    lower_rows = df[df["label"] == "lower"]
    for _, row in lower_rows.iterrows():
        features_dict = {col: row[col] for col in feature_names}
        errors = detector.detect_errors(features_dict)

        replay_short = row["replay_name"][:12]
        team_label = row["team_label"]
        print(f"\n  {replay_short}... ({team_label}): {len(errors)} erro(s)")
        for err in errors:
            print(f"    [{err['severity'].upper():5s}] {err['type']}: {err['description']}")

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
        "detector": detector,
        "dataset": df,
    }


if __name__ == "__main__":
    results = run_full_pipeline()
