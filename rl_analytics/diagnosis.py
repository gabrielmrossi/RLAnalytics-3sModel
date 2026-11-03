"""
diagnosis.py — Motor puro de diagnóstico tático, narrativa XAI (Z-Score × MDI) e anomalias espaciais.
Livre de dependências de interface (Streamlit) e de marcação HTML nos payloads de domínio.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from rl_analytics.config import FIELD_HALF_LENGTH, FIELD_HALF_WIDTH
from rl_analytics.feature_catalog import (
    FEATURE_CATALOG,
    NARRATIVE_TEMPLATES,
    classify_tier,
    compute_z_score,
    get_feature_label,
)

if TYPE_CHECKING:
    from rl_analytics.error_detection import ModelDrivenErrorDetector

_HTML_TAG_RE = re.compile(r"</?strong>")

_SCORE_OVERVIEWS_HTML: dict[str, str] = {
    "excellent": (
        "O time demonstra um <strong>jogo sólido</strong>, com fundamentos "
        "alinhados ao padrão de jogadores experientes."
    ),
    "good": (
        "O time tem <strong>fundamentos razoáveis</strong>, mas alguns pontos "
        "precisam de atenção para subir de nível."
    ),
    "average": (
        "O time apresenta <strong>dificuldades em fundamentos importantes</strong> "
        "da rotação e posicionamento."
    ),
    "poor": (
        "O time tem <strong>problemas sérios de rotação</strong> que impedem "
        "a evolução no competitivo."
    ),
}


def _clean_html(text: str, strip_html: bool = True) -> str:
    """Remove tags HTML de ênfase quando strip_html=True."""
    if not strip_html:
        return text
    return _HTML_TAG_RE.sub("", text)


def score_to_tier(score: float) -> str:
    """Converte a nota tática (0–100) no tier qualitativo do time."""
    if score >= 80:
        return "excellent"
    if score >= 60:
        return "good"
    if score >= 40:
        return "average"
    return "poor"


def compute_partial_feature_score(z: float, direction: str) -> float:
    """Calcula o score parcial (0–100) de uma feature a partir do z-score e direção."""
    if direction == "below":
        raw_score = 75.0 + z * 10.0
    elif direction == "above":
        raw_score = 75.0 - z * 10.0
    else:
        raw_score = 85.0 - abs(z) * 12.0
    return float(max(0.0, min(100.0, raw_score)))


def diagnose_team(
    team_feats: dict[str, float],
    detector: ModelDrivenErrorDetector,
    *,
    strip_html: bool = True,
) -> dict[str, Any]:
    """Gera o diagnóstico estruturado completo (nota 0–100, tier, resumo e findings por feature).

    Args:
        team_feats: Dicionário de features extraídas para o time.
        detector: Instância treinada/carregada de ModelDrivenErrorDetector.
        strip_html: Se True (padrão da API), remove qualquer tag HTML dos textos.

    Returns:
        Dict com 'score', 'tier', 'summary', 'narrative' e 'findings'.
    """
    if not detector.is_built:
        msg = "Modelo de IA ainda não disponível para diagnóstico."
        return {
            "score": 0.0,
            "tier": "poor",
            "summary": msg,
            "narrative": msg,
            "findings": [],
        }

    good_sentences: list[tuple[float, str]] = []
    neutral_sentences: list[tuple[float, str]] = []
    bad_sentences: list[tuple[float, str]] = []
    score_parts: list[tuple[float, float]] = []
    findings: list[dict[str, Any]] = []

    for feat_name in detector.feature_names:
        val = team_feats.get(feat_name)
        ref_mean = detector.reference_mean.get(feat_name)
        if val is None or ref_mean is None:
            continue

        ref_std = max(detector.reference_std.get(feat_name, 1.0), 0.01)
        importance = float(detector.feature_importances.get(feat_name, 0.0))
        z = float(compute_z_score(val, ref_mean, ref_std))

        tmpl = NARRATIVE_TEMPLATES.get(feat_name)
        if not tmpl:
            continue

        direction = tmpl["direction"]
        tier = classify_tier(z, direction)

        if tier == "bad" and direction == "any":
            raw_text = tmpl.get("bad_above" if z > 0 else "bad_below", tmpl.get("bad", ""))
        else:
            raw_text = tmpl.get(tier, "")

        if not raw_text:
            continue

        clean_text = _clean_html(raw_text, strip_html=strip_html)

        bucket = {
            "good": good_sentences,
            "neutral": neutral_sentences,
            "bad": bad_sentences,
        }
        bucket[tier].append((importance, clean_text))

        feat_score = compute_partial_feature_score(z, direction)
        score_parts.append((feat_score, importance))

        impact = importance * abs(z)
        if tier == "bad":
            severity = "alta" if (impact > 0.15 or abs(z) > 2.5) else "media"
        elif tier == "neutral":
            severity = "baixa"
        else:
            severity = "ok"

        findings.append({
            "feature": feat_name,
            "label": get_feature_label(feat_name),
            "display_name": FEATURE_CATALOG.get(feat_name, {}).get("display_name", feat_name),
            "value": round(float(val), 4),
            "ref_mean": round(float(ref_mean), 4),
            "ref_std": round(float(ref_std), 4),
            "z": round(z, 4),
            "mdi": round(importance, 4),
            "direction": direction,
            "tier": tier,
            "severity": severity,
            "partial_score": round(feat_score, 1),
            "weight": round(importance, 4),
            "text": _clean_html(raw_text, strip_html=True),
        })

    if score_parts:
        total_w = sum(w for _, w in score_parts)
        score = sum(s * w for s, w in score_parts) / total_w if total_w > 0 else 50.0
    else:
        score = 50.0
    score = float(max(0, min(100, round(score))))

    for group in (good_sentences, neutral_sentences, bad_sentences):
        group.sort(key=lambda x: x[0], reverse=True)

    findings.sort(key=lambda f: f["mdi"], reverse=True)

    team_tier = score_to_tier(score)
    overview = _clean_html(_SCORE_OVERVIEWS_HTML[team_tier], strip_html=strip_html)

    parts: list[str] = [overview]
    for _, text in good_sentences[:2]:
        parts.append(text)

    if bad_sentences:
        parts.append("No entanto, foram observados pontos críticos:")
        for _, text in bad_sentences:
            parts.append(f"• {text}")
    elif neutral_sentences:
        for _, text in neutral_sentences[:1]:
            parts.append(text)

    narrative = " ".join(parts)
    return {
        "score": score,
        "tier": team_tier,
        "summary": _clean_html(narrative, strip_html=True),
        "narrative": narrative,
        "findings": findings,
    }


def generate_tactical_narrative(
    team_feats: dict[str, float],
    detector: ModelDrivenErrorDetector,
    *,
    strip_html: bool = True,
) -> tuple[str, float]:
    """Gera diagnóstico em linguagem natural e nota tática (0–100)."""
    diag = diagnose_team(team_feats, detector, strip_html=strip_html)
    return diag["narrative"], float(diag["score"])


def detect_stagnation_anomalies(
    positions: pd.DataFrame,
    team_num: int,
) -> list[dict[str, Any]]:
    """Detecta anomalias de inatividade ou estagnação espacial prolongada de forma estruturada."""
    anomalies: list[dict[str, Any]] = []
    if positions.empty:
        return anomalies

    tdata = positions[(positions["team"] == team_num) & (positions["entity"] != "ball")]
    if tdata.empty:
        return anomalies

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
            cx = float((xedges[ix] + xedges[ix + 1]) / 2)
            cy = float((yedges[iy] + yedges[iy + 1]) / 2)
            seconds = int(max_in_cell)
            anomalies.append({
                "type": "stagnation",
                "player": str(player_name),
                "x": round(cx, 1),
                "y": round(cy, 1),
                "pct_time": round(pct_in_cell, 1),
                "seconds": seconds,
                "text": (
                    f"Alerta de Estagnação: O jogador {player_name} permaneceu concentrado "
                    f"na coordenada (X≈{cx:.0f}, Y≈{cy:.0f}) durante {pct_in_cell:.0f}% "
                    f"do seu tempo em campo (~{seconds}s), indicando ineficiência severa "
                    f"de rotação ou inatividade."
                ),
                "html": (
                    f"⚠️ <strong>Alerta de Estagnação:</strong> O jogador <strong>{player_name}</strong> "
                    f"permaneceu concentrado na coordenada (X≈{cx:.0f}, Y≈{cy:.0f}) durante {pct_in_cell:.0f}% "
                    f"do seu tempo em campo (~{seconds}s), indicando ineficiência severa de rotação ou inatividade."
                ),
            })

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
                    still_x, still_y = float(row["x"]), float(row["y"])
            else:
                still_streak = 0

        if max_still_streak >= 15 and pct_in_cell < 15.0:
            anomalies.append({
                "type": "immobility",
                "player": str(player_name),
                "x": round(still_x, 1),
                "y": round(still_y, 1),
                "pct_time": round((max_still_streak / len(pgroup)) * 100.0, 1),
                "seconds": int(max_still_streak),
                "text": (
                    f"Alerta de Imobilidade: O jogador {player_name} permaneceu praticamente "
                    f"imóvel por ~{max_still_streak}s consecutivos em (X≈{still_x:.0f}, Y≈{still_y:.0f})."
                ),
                "html": (
                    f"⚠️ <strong>Alerta de Imobilidade:</strong> O jogador <strong>{player_name}</strong> "
                    f"permaneceu praticamente imóvel por ~{max_still_streak}s consecutivos em (X≈{still_x:.0f}, Y≈{still_y:.0f})."
                ),
            })

    return anomalies


def check_stagnation_anomaly(
    positions: pd.DataFrame,
    team_num: int,
    *,
    strip_html: bool = False,
) -> list[str]:
    """Retorna lista de mensagens de alerta de estagnação/imobilidade."""
    anomalies = detect_stagnation_anomalies(positions, team_num)
    key = "text" if strip_html else "html"
    return [a[key] for a in anomalies]
