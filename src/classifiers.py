"""
classifiers.py — Classificadores supervisionados de rank de telemetria.
Implementa KNN, Decision Tree e Random Forest com validação Leave-One-Out compartilhada.
"""

from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text


def loo_evaluate(
    model_factory: Callable[[int], Any],
    X: np.ndarray,
    y: np.ndarray,
    labels: list[str] | None = None,
) -> dict[str, Any]:
    """Executa Leave-One-Out cross-validation de forma compartilhada.

    Args:
        model_factory: Função que recebe n_train e retorna o classificador instanciado.
        X: Matriz de features.
        y: Vetor de rótulos.
        labels: Lista de classes para matriz de confusão e relatório (padrão: ['higher', 'lower']).

    Returns:
        Dict com accuracy, confusion_matrix, classification_report e predictions.
    """
    if labels is None:
        labels = ["higher", "lower"]

    loo = LeaveOneOut()
    y_pred = np.empty_like(y)

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = model_factory(len(X_train))
        clf.fit(X_train_s, y_train)
        y_pred[test_idx] = clf.predict(X_test_s)

    acc = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred, labels=labels)
    report = classification_report(y, y_pred, labels=labels, output_dict=True)

    predictions = [
        {
            "index": i,
            "actual": str(y[i]),
            "predicted": str(y_pred[i]),
            "correct": bool(y[i] == y_pred[i]),
        }
        for i in range(len(y))
    ]

    return {
        "accuracy": float(acc),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "predictions": predictions,
    }


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

    def evaluate_loo(self, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        """Leave-One-Out cross-validation."""
        return loo_evaluate(
            lambda n_train: KNeighborsClassifier(n_neighbors=min(self.n_neighbors, n_train)),
            X,
            y,
        )

    def feature_importance_loo(
        self, X: np.ndarray, y: np.ndarray, feature_names: list[str]
    ) -> dict[str, float]:
        """Calcula importância de features via ablation (remover 1 por vez)."""
        baseline = self.evaluate_loo(X, y)["accuracy"]
        importance = {}

        for i, name in enumerate(feature_names):
            X_reduced = np.delete(X, i, axis=1)
            result = self.evaluate_loo(X_reduced, y)
            drop = baseline - result["accuracy"]
            importance[name] = round(drop, 4)

        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))


class _SklearnRankClassifierBase:
    """Classe base compartilhada para classificadores baseados em árvores (Tree e RF)."""

    def __init__(self, model: Any):
        self._model = model
        self.scaler = StandardScaler()
        self.feature_names: list[str] = []
        self.is_trained = False

    @property
    def model(self) -> Any:
        if "_model" in self.__dict__:
            return self._model
        if "rf" in self.__dict__:
            return self.__dict__["rf"]
        if "tree" in self.__dict__:
            return self.__dict__["tree"]
        raise AttributeError("Estimador não encontrado no classificador.")

    @model.setter
    def model(self, value: Any) -> None:
        self._model = value

    @property
    def estimator(self) -> Any:
        return self.model

    def train(self, X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> None:
        """Treina o estimador com todos os dados escalados."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.feature_names = feature_names
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prediz labels para novas amostras."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def get_feature_importances(self) -> dict[str, float]:
        """Retorna importância MDI / Gini de cada feature."""
        if not self.is_trained:
            return {}
        importances = self.model.feature_importances_
        result = {name: round(float(imp), 4) for name, imp in zip(self.feature_names, importances)}
        return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))


class RankClassifierTree(_SklearnRankClassifierBase):
    """Decision Tree para classificação + geração de regras legíveis."""

    def __init__(self, max_depth: int = 3):
        self.max_depth = max_depth
        super().__init__(DecisionTreeClassifier(max_depth=max_depth, random_state=42))

    @property
    def tree(self):
        return self.model

    @tree.setter
    def tree(self, value):
        self._model = value

    def evaluate_loo(self, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        """Leave-One-Out cross-validation para a Decision Tree."""
        return loo_evaluate(
            lambda _: DecisionTreeClassifier(max_depth=self.max_depth, random_state=42),
            X,
            y,
        )

    def get_rules(self) -> str:
        """Exporta regras da árvore em texto legível."""
        if not self.is_trained:
            return "Modelo nao treinado."
        return export_text(
            self.model,
            feature_names=self.feature_names,
            class_names=["higher", "lower"],
        )


class RankClassifierRandomForest(_SklearnRankClassifierBase):
    """Random Forest para classificação de rank + importância nativa de features."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        super().__init__(
            RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
            )
        )

    @property
    def rf(self):
        return self.model

    @rf.setter
    def rf(self, value):
        self._model = value

    def evaluate_loo(self, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        """Leave-One-Out cross-validation para o Random Forest."""
        return loo_evaluate(
            lambda _: RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
            ),
            X,
            y,
        )

    def plot_feature_importances(
        self,
        save_path: str | None = None,
        title: str = "Importancia das Features (Random Forest)",
    ) -> Any:
        """Gera gráfico de barras horizontal com Seaborn/Matplotlib."""
        from visualization.static import plot_feature_importances_static, save_figure

        imp_dict = self.get_feature_importances()
        fig = plot_feature_importances_static(imp_dict, title=title)
        if save_path:
            save_figure(fig, save_path)
        return fig
