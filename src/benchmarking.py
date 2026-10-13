"""
benchmarking.py — Avaliação comparativa de modelos supervisionados via StratifiedKFold.
"""

import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from config import DARK_BG, DARK_PAPER


class ModelBenchmarking:
    """Avaliacao comparativa dos modelos supervisionados via StratifiedKFold.

    Compara KNN, Decision Tree e Random Forest nos mesmos folds,
    extraindo Acuracia, F1, Precisao e Recall ponderados.
    """

    _DARK_BG = DARK_BG
    _DARK_PAPER = DARK_PAPER
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

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        knn_k: int = 3,
        tree_depth: int = 3,
        rf_n_estimators: int = 100,
    ) -> pd.DataFrame:
        """Executa StratifiedKFold nos 3 modelos e retorna tabela resumo."""
        models = {
            "KNN": lambda: KNeighborsClassifier(n_neighbors=min(knn_k, len(X) - 1)),
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
            fold_metrics = {"accuracy": [], "f1": [], "precision": [], "recall": []}
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
                fold_metrics["f1"].append(
                    f1_score(y_test, y_pred, labels=self._LABELS, average="weighted", zero_division=0)
                )
                fold_metrics["precision"].append(
                    precision_score(y_test, y_pred, labels=self._LABELS, average="weighted", zero_division=0)
                )
                fold_metrics["recall"].append(
                    recall_score(y_test, y_pred, labels=self._LABELS, average="weighted", zero_division=0)
                )

                all_y_true.extend(y_test)
                all_y_pred.extend(y_pred)

            self.results_[model_name] = {
                k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                for k, v in fold_metrics.items()
            }

            cm = confusion_matrix(all_y_true, all_y_pred, labels=self._LABELS)
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            cm_norm = np.nan_to_num(cm_norm, nan=0.0)
            self.confusion_matrices_[model_name] = cm_norm

        rows = []
        for model_name, metrics in self.results_.items():
            rows.append({
                "Modelo": model_name,
                "Acuracia": f"{metrics['accuracy']['mean']:.1%} ± {metrics['accuracy']['std']:.1%}",
                "F1-Score": f"{metrics['f1']['mean']:.1%} ± {metrics['f1']['std']:.1%}",
                "Precisao": f"{metrics['precision']['mean']:.1%} ± {metrics['precision']['std']:.1%}",
                "Recall": f"{metrics['recall']['mean']:.1%} ± {metrics['recall']['std']:.1%}",
                "_f1_mean": metrics["f1"]["mean"],
            })

        self.summary_df_ = (
            pd.DataFrame(rows)
            .sort_values("_f1_mean", ascending=False)
            .drop(columns=["_f1_mean"])
            .reset_index(drop=True)
        )
        return self.summary_df_

    def plot_confusion_matrices(
        self,
        save_path: str = "data/processed/models_comparison_cm.png",
    ) -> go.Figure:
        """Gera 3 matrizes de confusao normalizadas lado a lado."""
        if not self.confusion_matrices_:
            raise RuntimeError("Execute evaluate() antes de plotar.")

        model_names = list(self.confusion_matrices_.keys())

        # Matplotlib / Seaborn (PNG)
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

        # Plotly
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
            title=dict(text="Matrizes de Confusao Normalizadas (StratifiedKFold)", font=dict(size=16)),
            plot_bgcolor=self._DARK_BG,
            paper_bgcolor=self._DARK_PAPER,
            font=dict(color="white"),
            height=400,
            margin=dict(l=60, r=30, t=80, b=60),
        )

        return fig_plotly
