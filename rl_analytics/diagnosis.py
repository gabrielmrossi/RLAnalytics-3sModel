"""
diagnosis.py — Motor puro de diagnóstico tático, narrativa XAI (Z-Score × MDI) e anomalias espaciais.
Livre de dependências de interface (Streamlit) e de marcação HTML nos payloads de domínio.
Apresenta os resultados como ferramenta de apoio à análise do replay:
- Observação (o que foi medido e comparado à referência GC+);
- Possível interpretação (sem afirmação de causalidade ou erro);
- O que observar no replay (sugestão prática de investigação contextual);
- Síntese Geral do Posicionamento integrando geometria, pressão e ritmo coletivo.
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
    get_feature_diagnostic_guidance,
    get_feature_label,
)

if TYPE_CHECKING:
    from rl_analytics.error_detection import ModelDrivenErrorDetector

_HTML_TAG_RE = re.compile(r"</?strong>")

_SCORE_OVERVIEWS_HTML: dict[str, str] = {
    "excellent": (
        "A equipe apresenta <strong>alta aderência</strong> ao perfil estatístico de referência Grand Champion+, "
        "com a maioria dos indicadores de dispersão, ritmo e geometria alinhada aos padrões das equipes de topo."
    ),
    "good": (
        "A equipe apresenta <strong>aderência moderada</strong> ao perfil de referência Grand Champion+, "
        "com proximidade estatística na maioria das características e variações pontuais que merecem revisão contextual no replay."
    ),
    "average": (
        "A equipe apresenta <strong>aderência parcial</strong> ao perfil de referência Grand Champion+, "
        "com desvios consideráveis em indicadores espaciais ou de ritmo em relação ao padrão das equipes de rank alto."
    ),
    "poor": (
        "A equipe apresenta <strong>baixa aderência</strong> ao perfil estatístico de referência Grand Champion+, "
        "com divergência acentuada na maior parte das características coletivas analisadas."
    ),
}


def _clean_html(text: str, strip_html: bool = True) -> str:
    """Remove tags HTML de ênfase quando strip_html=True."""
    if not strip_html:
        return text
    return _HTML_TAG_RE.sub("", text)


def score_to_tier(score: float) -> str:
    """Converte o score numérico (0–100) no tier qualitativo de aderência ao perfil GC+."""
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


def generate_positioning_synthesis(
    findings: list[dict[str, Any]],
    team_tier: str,
    score: float,
) -> str:
    """Gera a Síntese Geral do Posicionamento interpretando o conjunto de indicadores.

    Reúne os principais resultados coletivos, correlaciona dimensões táticas fundamentais
    (geometria espacial, abordagem da bola vs. retaguarda, dinâmica cinética e boost)
    e propõe caminhos de investigação no replay sem afirmar intenção ou juízo determinista.
    """
    if not findings:
        return "Dados de telemetria insuficientes para elaboração da síntese de posicionamento."

    z_map = {f["feature"]: float(f["z"]) for f in findings}

    # 1. Dimensão Espacial e Geometria da Formação
    spread_z = z_map.get("team_spread_mean", 0.0)
    area_z = z_map.get("triangle_area_mean", 0.0)
    spread_std_z = z_map.get("team_spread_std", 0.0)
    area_std_z = z_map.get("triangle_area_std", 0.0)

    geo_parts: list[str] = []
    if spread_z < -0.8 and area_z < -0.8:
        geo_parts.append(
            "Quanto à geometria coletiva, a equipe atuou em um bloco sensivelmente mais compacto que a média de referência GC+, "
            "com menor distanciamento entre companheiros e área de formação reduzida."
        )
        if spread_std_z > 0.8 or area_std_z > 0.8:
            geo_parts.append(
                "Essa estrutura compacta alternou-se com variações bruscas de abertura ao longo da partida, indicando transições espaciais desiguais entre os momentos de contenção e avanço."
            )
        else:
            geo_parts.append(
                "Essa proximidade manteve-se uniforme durante o jogo. No replay, cabe averiguar se essa postura decorreu de uma escolha deliberada de cobertura curta e dobras defensivas na área, ou se acabou gerando aglomeração e perda de amplitude para saídas de bola."
            )
    elif spread_z < -0.5 or area_z < -0.5:
        geo_parts.append(
            "Quanto à geometria coletiva, a equipe operou com distâncias e área triangular moderadamente mais contidas que a média GC+, tendendo a um agrupamento mais próximo na maior parte das fases."
        )
        if spread_std_z > 0.8:
            geo_parts.append(
                "A variação acentuada no espaçamento sugere que as recomposições ocorreram de forma oscilante durante a partida."
            )
        else:
            geo_parts.append(
                "O espaçamento reduzido favoreceu o suporte imediato entre companheiros, embora exija atenção no replay para avaliar se houve espaço para inversões de jogada."
            )
    elif spread_z > 0.5 or area_z > 0.5:
        geo_parts.append(
            "No aspecto espacial, a equipe apresentou formação ampla e espaçamento expandido pelo campo, cobrindo faixas extensas da arena."
        )
        if spread_std_z > 0.8 or area_std_z > 0.8:
            geo_parts.append(
                "Entretanto, a elevada oscilação nessas distâncias indica recomposições assimétricas, com momentos de dispersão exagerada que podem afastar o terceiro homem da disputa."
            )
        else:
            geo_parts.append(
                "A estabilidade nessa dispersão favoreceu a distribuição equilibrada em campo, oferecendo opções abertas de passe e cobertura preventiva de rebotes."
            )
    else:
        geo_parts.append(
            "A distribuição espacial e a área de formação triangular da equipe mantiveram-se equilibradas e consistentes com o perfil de referência GC+, sem agrupamentos setoriais desnecessários nem dispersões excessivas."
        )

    # 2. Dimensão de Pressão na Bola e Equilíbrio da Retaguarda
    chase_z = z_map.get("ball_chase_ratio", 0.0)
    cov_z = z_map.get("defensive_coverage_pct", 0.0)
    ball_dist_z = z_map.get("avg_dist_to_ball", 0.0)
    min_dist_z = z_map.get("min_player_dist_to_ball", 0.0)

    pres_parts: list[str] = []
    if chase_z > 0.8 and cov_z < -0.6:
        pres_parts.append(
            "Na relação entre contestação de bola e proteção da retaguarda, observa-se uma dinâmica de maior exposição: registrou-se frequência elevada de múltiplos jogadores atuando simultaneamente nas imediações da bola combinada a um menor tempo de cobertura recuada no meio defensivo."
        )
        pres_parts.append(
            "Esse comportamento aponta para avanços de bloco com forte ímpeto ofensivo, mas que deixam o campo vulnerável a lançamentos diretos e contra-ataques rápidos. No replay, recomenda-se verificar se essas aproximações conjuntas foram contestações coordenadas ou se resultaram em disputas concorrentes (double commits)."
        )
    elif chase_z > 0.8:
        pres_parts.append(
            "Na abordagem da bola, a equipe apresentou maior índice de aproximações simultâneas de dois ou mais jogadores na mesma zona da bola, embora a proteção recuada tenha permanecido intermediária."
        )
        pres_parts.append(
            "Convém conferir no replay se as disputas divididas trouxeram controle imediato de posse ou se geraram sobreposições que atrasaram a transição para a defesa."
        )
    elif cov_z < -0.8:
        pres_parts.append(
            "Em relação à retaguarda territorial, a proporção de frames com jogador posicionado no próprio meio-campo foi inferior ao padrão GC+, sinalizando permanência prolongada dos três integrantes no campo de ataque."
        )
        pres_parts.append(
            "Vale avaliar no replay se essa postura refletiu pressão ofensiva sustentada na área adversária ou se decorreu de lentidão no retorno após perdas de posse."
        )
    elif min_dist_z > 0.8 or ball_dist_z > 0.8:
        pres_parts.append(
            "A distância média da equipe e do primeiro homem em relação à bola esteve mais recuada que a referência, sugerindo marcação paciente (shadow defense) ou abordagem mais cautelosa ao portador adversário."
        )
    else:
        pres_parts.append(
            "A articulação entre disputa da bola e cobertura defensiva mostrou-se bem calibrada: a equipe alternou os desafios ao portador adversário sem comprometer a presença contínua de uma peça de segurança na retaguarda."
        )

    # 3. Dimensão Cinética e Gestão de Recursos (Boost)
    speed_z = z_map.get("avg_speed", 0.0)
    boost_z = z_map.get("avg_boost_pct", 0.0)

    din_parts: list[str] = []
    if speed_z < -0.8 and boost_z < -0.8:
        din_parts.append(
            "No ritmo de deslocamento e sustentação de velocidade, a equipe combinou velocidade média linear inferior e reservas instantâneas de boost reduzidas."
        )
        din_parts.append(
            "Essa correlação sugere que os jogadores enfrentaram momentos de desabastecimento de combustível, o que impacta diretamente a velocidade das recomposições e a capacidade de reação rápida em bolas aéreas. Vale checar no replay se as rotas de recuo priorizaram a coleta de pads pequenos ou se exigiram desvios longos para boosts de 100."
        )
    elif speed_z < -0.8:
        din_parts.append(
            "A velocidade média dos veículos esteve abaixo da referência das equipes de rank alto, apesar de o nível de boost ter permanecido em patamar intermediário."
        )
        din_parts.append(
            "Esse dado pode indicar uma cadência de jogo mais cautelosa e posicional, ou hesitação na leitura de jogadas antes da aceleração. A análise do vídeo permite diferenciar se foi uma escolha de controle ou lentidão de reação."
        )
    elif boost_z < -0.8:
        din_parts.append(
            "A equipe manteve velocidade dinâmica de movimentação, mas operou com reservas instantâneas de boost mais baixas que o habitual em GC+."
        )
        din_parts.append(
            "Esse padrão indica alto gasto energético em transições contínuas, exigindo atenção especial no replay para verificar se houve vulnerabilidade em disputas no ar por falta de combustível residual."
        )
    else:
        din_parts.append(
            "A velocidade de jogo e a gestão de combustível apresentaram harmonia, sustentando deslocamentos velozes e recuperação eficiente de pads ao longo das fases da partida."
        )

    paragraph_1 = " ".join(geo_parts)
    paragraph_2 = " ".join(pres_parts)
    paragraph_3 = " ".join(din_parts)

    return f"{paragraph_1}\n\n{paragraph_2}\n\n{paragraph_3}"


def diagnose_team(
    team_feats: dict[str, float],
    detector: ModelDrivenErrorDetector,
    *,
    strip_html: bool = True,
) -> dict[str, Any]:
    """Gera o diagnóstico estruturado completo (nota 0–100, tier de aderência, resumo e findings por feature).

    Args:
        team_feats: Dicionário de features extraídas para o time.
        detector: Instância treinada/carregada de ModelDrivenErrorDetector.
        strip_html: Se True (padrão da API), remove qualquer tag HTML dos textos.

    Returns:
        Dict com 'score', 'tier', 'summary', 'narrative', 'positioning_synthesis' e 'findings'.
    """
    if not detector.is_built:
        msg = "Modelo de IA ainda não disponível para diagnóstico."
        return {
            "score": 0.0,
            "tier": "poor",
            "summary": msg,
            "narrative": msg,
            "positioning_synthesis": msg,
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

        guidance = get_feature_diagnostic_guidance(feat_name, z, tier)
        observation = guidance["observation"]
        interpretation = guidance["interpretation"]
        replay_suggestion = guidance["replay_suggestion"]

        # Texto explicativo simplificado, objetivo e direto (sem repetições desnecessárias)
        clean_text = f"{observation} {interpretation}"

        bucket = {
            "good": good_sentences,
            "neutral": neutral_sentences,
            "bad": bad_sentences,
        }
        # Adiciona resumo curto no bucket para construção do overview
        bucket[tier].append((importance, f"{get_feature_label(feat_name)}: {observation}"))

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
            "text": _clean_html(clean_text, strip_html=True),
            "observation": observation,
            "interpretation": interpretation,
            "replay_suggestion": replay_suggestion,
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
    if bad_sentences:
        parts.append("Indicadores com maior divergência para investigação no replay:")
        for _, text in bad_sentences:
            parts.append(f"• {text}")
    elif neutral_sentences:
        parts.append("Indicadores com variações intermediárias a acompanhar:")
        for _, text in neutral_sentences[:2]:
            parts.append(f"• {text}")
    elif good_sentences:
        parts.append("Características com padrão mais próximo da referência GC+:")
        for _, text in good_sentences[:2]:
            parts.append(f"• {text}")

    narrative = " ".join(parts)
    positioning_synthesis = generate_positioning_synthesis(findings, team_tier, score)

    return {
        "score": score,
        "tier": team_tier,
        "summary": _clean_html(narrative, strip_html=True),
        "narrative": narrative,
        "positioning_synthesis": positioning_synthesis,
        "findings": findings,
    }


def generate_tactical_narrative(
    team_feats: dict[str, float],
    detector: ModelDrivenErrorDetector,
    *,
    strip_html: bool = True,
) -> tuple[str, float]:
    """Gera diagnóstico em linguagem natural e nota de aderência ao perfil GC+ (0–100)."""
    diag = diagnose_team(team_feats, detector, strip_html=strip_html)
    return diag["narrative"], float(diag["score"])


def detect_stagnation_anomalies(
    positions: pd.DataFrame,
    team_num: int,
) -> list[dict[str, Any]]:
    """Detecta concentrações espaciais e períodos de baixa variação de posição de forma estruturada.

    Limita-se a constatações geométricas objetivas e propõe verificação contextual no replay,
    sem afirmar ineficiência ou erro tático a priori.
    """
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
                    f"Concentração Espacial: O jogador {player_name} esteve presente na região "
                    f"(X≈{cx:.0f}, Y≈{cy:.0f}) em {pct_in_cell:.0f}% das amostras (~{seconds}s). "
                    f"Sugere-se verificar no replay se a permanência prolongada nessa zona atendeu a uma cobertura defensiva posicional "
                    f"ou se gerou isolamento das fases ofensivas da equipe."
                ),
                "html": (
                    f"📍 <strong>Concentração Espacial:</strong> O jogador <strong>{player_name}</strong> "
                    f"esteve presente na região (X≈{cx:.0f}, Y≈{cy:.0f}) em {pct_in_cell:.0f}% das amostras (~{seconds}s). "
                    f"Sugere-se verificar no replay se a permanência prolongada nessa zona atendeu a uma cobertura defensiva posicional "
                    f"ou se gerou isolamento das fases ofensivas da equipe."
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
                    f"Baixa Variação Posicional: O jogador {player_name} registrou deslocamento mínimo "
                    f"por ~{max_still_streak}s contínuos em (X≈{still_x:.0f}, Y≈{still_y:.0f}). "
                    f"Investigar no replay se o jogador estava contendo avanço adversário (shadow defense / guarda de trave) "
                    f"ou se houve desconexão temporária da dinâmica da partida."
                ),
                "html": (
                    f"⏱️ <strong>Baixa Variação Posicional:</strong> O jogador <strong>{player_name}</strong> "
                    f"registrou deslocamento mínimo por ~{max_still_streak}s contínuos em (X≈{still_x:.0f}, Y≈{still_y:.0f}). "
                    f"Investigar no replay se o jogador estava contendo avanço adversário (shadow defense / guarda de trave) "
                    f"ou se houve desconexão temporária da dinâmica da partida."
                ),
            })

    return anomalies


def check_stagnation_anomaly(
    positions: pd.DataFrame,
    team_num: int,
    *,
    strip_html: bool = False,
) -> list[str]:
    """Retorna lista de mensagens de concentração espacial e baixa variação posicional."""
    anomalies = detect_stagnation_anomalies(positions, team_num)
    key = "text" if strip_html else "html"
    return [a[key] for a in anomalies]
