"""
error_detection.py — Detectores de erros de posicionamento e inconsistências táticas.

Contém:
    1. PositioningErrorDetector: Detector heurístico baseado em limiares z-score fixos
       (mantido como baseline de comparação para o TCC).
    2. ModelDrivenErrorDetector: Detector avançado ponderado pela importância nativa (MDI)
       do modelo Random Forest.
"""

import os
from typing import Any

import joblib
import numpy as np
import pandas as pd

from classifiers import RankClassifierRandomForest
from config import META_COLS, MODEL_PATH
from feature_catalog import FEATURE_DESCRIPTIONS, compute_z_score


def build_reference_profile(
    df: pd.DataFrame, feature_names: list[str]
) -> tuple[dict[str, float], dict[str, float]]:
    """Calcula média e desvio padrão de referência a partir das amostras de rank alto (higher).

    Args:
        df: DataFrame com coluna 'label' e colunas de features.
        feature_names: Lista com os nomes das features a calcular.

    Returns:
        (reference_mean, reference_std)
    """
    higher = df[df["label"] == "higher"]
    if higher.empty:
        return {}, {}

    ref_mean: dict[str, float] = {}
    ref_std: dict[str, float] = {}

    for col in feature_names:
        if col in higher.columns:
            ref_mean[col] = float(higher[col].mean())
            ref_std[col] = float(higher[col].std()) if len(higher) > 1 else 0.0

    return ref_mean, ref_std


class PositioningErrorDetector:
    """Detector heurístico de erros de posicionamento (baseline comparativo por regras fixas).

    Compara features de um time com limiares z-score fixos (1.5 / 2.5 desvios padrão)
    em relação ao perfil médio do grupo higher.
    """

    def __init__(self):
        self.reference_mean: dict[str, float] = {}
        self.reference_std: dict[str, float] = {}
        self.is_built = False

    def build_reference(self, df: pd.DataFrame) -> None:
        """Constrói perfil de referência a partir dos replays 'higher'."""
        feature_cols = [c for c in df.columns if c not in META_COLS]
        self.reference_mean, self.reference_std = build_reference_profile(df, feature_cols)
        if self.reference_mean:
            self.is_built = True

    def detect_errors(self, team_features: dict[str, float]) -> list[dict]:
        """Detecta erros de posicionamento em um time usando limiares fixos."""
        if not self.is_built:
            return []

        errors = []

        # 1. Ball-chasing: ratio muito acima da referência
        if "ball_chase_ratio" in team_features and "ball_chase_ratio" in self.reference_mean:
            ref_mean = self.reference_mean["ball_chase_ratio"]
            ref_std = max(self.reference_std.get("ball_chase_ratio", 5.0), 1.0)
            val = team_features["ball_chase_ratio"]
            z_score = compute_z_score(val, ref_mean, ref_std, min_std=1.0)

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

        # 3. Cluster (triângulo pequeno demais)
        if "triangle_area_mean" in team_features and "triangle_area_mean" in self.reference_mean:
            ref_mean = self.reference_mean["triangle_area_mean"]
            ref_std = max(self.reference_std.get("triangle_area_mean", 100000.0), 1.0)
            val = team_features["triangle_area_mean"]
            z_score = (ref_mean - val) / ref_std

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

        # 4. Dispersão muito inconsistente (std alto)
        if "team_spread_std" in team_features and "team_spread_std" in self.reference_mean:
            ref_mean = self.reference_mean["team_spread_std"]
            ref_std = max(self.reference_std.get("team_spread_std", 200.0), 1.0)
            val = team_features["team_spread_std"]
            z_score = compute_z_score(val, ref_mean, ref_std, min_std=1.0)

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

        # 5. Boost management (boost médio baixo)
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
    """Detecta erros táticos usando o Random Forest treinado.

    Usa a importância de cada feature (MDI do RF) para ponderar os desvios
    em relação ao perfil de rank alto. Features com peso maior no modelo
    têm prioridade maior no diagnóstico.
    """

    _FEATURE_DESCRIPTIONS = FEATURE_DESCRIPTIONS

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        self.rf = RankClassifierRandomForest(
            n_estimators=n_estimators, random_state=random_state
        )
        self.reference_mean: dict[str, float] = {}
        self.reference_std: dict[str, float] = {}
        self.feature_importances: dict[str, float] = {}
        self.feature_names: list[str] = []
        self.is_built = False

    def build(
        self,
        df: pd.DataFrame,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
    ) -> dict[str, float]:
        """Treina o RF e constrói perfil de referência."""
        self.feature_names = feature_names
        self.rf.train(X, y, feature_names)
        self.feature_importances = self.rf.get_feature_importances()

        self.reference_mean, self.reference_std = build_reference_profile(df, feature_names)
        if self.reference_mean:
            self.is_built = True

        return self.feature_importances

    def save(self, path: str = str(MODEL_PATH)) -> None:
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
    def load(cls, path: str = str(MODEL_PATH)) -> "ModelDrivenErrorDetector":
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
        """Detecta erros ponderados pela importância do modelo."""
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

            raw_z = compute_z_score(val, ref_mean, ref_std)

            feat_info = self._FEATURE_DESCRIPTIONS.get(feat_name, {})
            direction = feat_info.get("direction", "any")

            if direction == "above" and raw_z <= z_threshold:
                continue
            elif direction == "below" and -raw_z <= z_threshold:
                continue
            elif direction == "any" and abs(raw_z) <= z_threshold:
                continue

            impact = importance * abs(raw_z)

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

        errors.sort(key=lambda e: e["impact"], reverse=True)
        return errors
