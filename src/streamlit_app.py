"""
streamlit_app.py -- Dashboard de analise de replays 3v3 de Rocket League.

Abas:
    1. Replay ao Vivo (campo + triangulo de rotacao + boost)
    2. Higher vs Lower (radar + diferencas)
    3. Modelos de IA (comparativo, importancia, clusterizacao)
    4. Erros Taticos (deteccao de erros de posicionamento)
"""

import json
import os
import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

from analyzer import ReplayAnalyzer, compute_triangle_area, compute_distance
from parser import discover_replays, PROCESSED_DIR
from feature_engineering import build_dataset, extract_team_features
from knn_model import (
    RankClassifierTree,
    RankClassifierRandomForest,
    TacticalClusterKMeans,
    ModelBenchmarking,
    PositioningErrorDetector,
    ModelDrivenErrorDetector,
    load_dataset,
    get_X_y,
)

st.set_page_config(
    page_title="RL Analytics - TCC",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────
#  CSS
# ──────────────────────────────────────────────

st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .scoreboard {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 1.5rem;
        padding: 0.6rem 1.5rem;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        border: 1px solid #0f3460;
        margin-bottom: 0.5rem;
    }
    .scoreboard .team-score {
        font-size: 2rem;
        font-weight: 700;
    }
    .scoreboard .blue { color: #3498db; }
    .scoreboard .orange { color: #e67e22; }
    .scoreboard .match-time {
        font-size: 1.1rem;
        color: #a0a0a0;
        font-family: monospace;
        min-width: 60px;
        text-align: center;
    }
    .scoreboard .separator {
        font-size: 1.5rem;
        color: #555;
    }
    .error-card {
        padding: 0.8rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
    }
    .error-alta {
        background: rgba(233, 69, 96, 0.15);
        border-left: 4px solid #e94560;
    }
    .error-media {
        background: rgba(255, 165, 0, 0.15);
        border-left: 4px solid orange;
    }
    .uu-info {
        font-size: 0.75rem;
        color: #666;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
#  Constantes
# ──────────────────────────────────────────────

FIELD_HALF_LENGTH = 5120
FIELD_HALF_WIDTH = 4096

TEAM_COLORS = {0: "#3498db", 1: "#e67e22"}
TEAM_FILL_COLORS = {0: "rgba(52, 152, 219, 0.15)", 1: "rgba(230, 126, 34, 0.15)"}
TEAM_NAMES = {0: "Azul", 1: "Laranja"}
BALL_COLOR = "#ffffff"

_DARK_BG = "#0e1117"
_META_COLS = {"label", "replay_name", "team", "team_label"}

# ──────────────────────────────────────────────
#  Dicionario de Traducao de Features
# ──────────────────────────────────────────────

# Mapeia nomes tecnicos internos -> portugues claro para exibicao.
# NUNCA altere as chaves; elas devem coincidir com os nomes reais das colunas.
FEATURE_LABELS: dict[str, str] = {
    "avg_speed":                "Velocidade Média",
    "triangle_area_mean":       "Área Média do Triângulo",
    "team_spread_mean":         "Dispersão Média da Equipe",
    "triangle_area_std":        "Variação da Área do Triângulo",
    "team_spread_std":          "Variação da Dispersão",
    "avg_dist_to_ball":         "Dist. Média até a Bola",
    "min_player_dist_to_ball":  "Dist. Mínima à Bola",
    "ball_chase_ratio":         "Taxa de Ball Chasing",
    "defensive_coverage_pct":   "Cobertura Defensiva (%)",
    "avg_boost_pct":            "Média de Boost (%)",
}


def _translate(name: str) -> str:
    """Retorna o label PT-BR de uma feature, ou o proprio nome se nao mapeado."""
    return FEATURE_LABELS.get(name, name)


# ──────────────────────────────────────────────
#  Motor de Narrativa Tática
# ──────────────────────────────────────────────

_NARRATIVE_TEMPLATES: dict[str, dict] = {
    "ball_chase_ratio": {
        "direction": "above",
        "good": "A disciplina de rotação é boa — os jogadores evitam disputar a bola ao mesmo tempo.",
        "neutral": "A rotação é aceitável, mas há momentos em que mais de um jogador vai na bola ao mesmo tempo.",
        "bad": "O time sofre com <strong>ball chasing</strong>: com frequência, dois ou mais jogadores disputam a bola juntos, deixando o campo desprotegido.",
    },
    "defensive_coverage_pct": {
        "direction": "below",
        "good": "A cobertura defensiva é sólida, com pelo menos um jogador protegendo a retaguarda na maior parte do jogo.",
        "neutral": "A defesa está presente na maioria das jogadas, mas há janelas de vulnerabilidade onde o gol fica exposto.",
        "bad": "O time deixa o gol <strong>desprotegido com frequência</strong>, faltando um jogador recuado para cobrir contra-ataques.",
    },
    "triangle_area_mean": {
        "direction": "below",
        "good": "O espaçamento entre os jogadores é <strong>amplo e eficiente</strong>, cobrindo bastante área do campo.",
        "neutral": "A cobertura de campo está na média, sem se destacar positiva ou negativamente.",
        "bad": "O time fica <strong>muito agrupado</strong>, cobrindo pouca área do campo e facilitando jogadas adversárias.",
    },
    "triangle_area_std": {
        "direction": "above",
        "good": "A formação é <strong>estável</strong> ao longo da partida, indicando rotações bem coordenadas.",
        "neutral": "A formação oscila moderadamente ao longo do jogo, o que é aceitável.",
        "bad": "A formação do time é <strong>instável</strong>: ora muito agrupados, ora muito espalhados, sem um padrão claro de rotação.",
    },
    "team_spread_mean": {
        "direction": "below",
        "good": "A distância entre os jogadores é <strong>bem distribuída</strong>, facilitando passes e coberturas.",
        "neutral": "A distribuição dos jogadores pelo campo está dentro do esperado.",
        "bad": "Os jogadores ficam <strong>muito próximos entre si</strong>, limitando as opções de passe e cobertura.",
    },
    "team_spread_std": {
        "direction": "above",
        "good": "A rotação é <strong>consistente</strong>, com os jogadores mantendo uma distribuição regular ao longo da partida.",
        "neutral": "A consistência da rotação é razoável, com algumas variações.",
        "bad": "A rotação é <strong>irregular</strong> — há momentos de boa distribuição intercalados com agrupamentos desnecessários.",
    },
    "avg_boost_pct": {
        "direction": "below",
        "good": "A gestão de boost é <strong>eficiente</strong>, com boas reservas mantidas ao longo da partida.",
        "neutral": "O consumo de boost está dentro do normal, sem excessos nem escassez.",
        "bad": "O time opera com <strong>boost baixo</strong> na maior parte do tempo, limitando a capacidade de reagir rapidamente.",
    },
    "avg_dist_to_ball": {
        "direction": "any",
        "good": "O posicionamento em relação à bola está <strong>equilibrado</strong>, sem ficar longe nem perto demais.",
        "neutral": "A distância média até a bola está aceitável, mas pode ser otimizada.",
        "bad_above": "O time tende a ficar <strong>longe demais da bola</strong>, perdendo oportunidades de pressão.",
        "bad_below": "O time se posiciona <strong>muito próximo da bola</strong>, reforçando a falta de rotação para trás.",
    },
    "min_player_dist_to_ball": {
        "direction": "any",
        "good": "O jogador mais próximo da bola mantém uma <strong>distância saudável</strong>, com espaço para ler e reagir.",
        "neutral": "A distância do jogador mais próximo à bola está dentro do padrão.",
        "bad_above": "O jogador mais próximo fica <strong>distante demais da bola</strong>, demorando a chegar nas disputas.",
        "bad_below": "O jogador mais próximo fica <strong>colado na bola</strong>, sem espaço de reação para mudanças de direção.",
    },
    "avg_speed": {
        "direction": "below",
        "good": "O ritmo de jogo é <strong>alto</strong>, com movimentação rápida e intencional pelo campo.",
        "neutral": "A velocidade de movimentação está na média.",
        "bad": "O ritmo de jogo está <strong>abaixo do esperado</strong>, com movimentação lenta que dificulta a pressão no adversário.",
    },
}

_SCORE_OVERVIEWS = {
    "excellent": "O time demonstra um <strong>jogo sólido</strong>, com fundamentos alinhados ao padrão de jogadores experientes.",
    "good": "O time tem <strong>fundamentos razoáveis</strong>, mas alguns pontos precisam de atenção para subir de nível.",
    "average": "O time apresenta <strong>dificuldades em fundamentos importantes</strong> da rotação e posicionamento.",
    "poor": "O time tem <strong>problemas sérios de rotação</strong> que impedem a evolução no competitivo.",
}


def _classify_tier(z: float, direction: str) -> str:
    """Classifica z-score em 'good', 'neutral', ou 'bad' baseado na direcao."""
    if direction == "above":  # valor alto = ruim
        if z > 1.2:
            return "bad"
        if z > 0.5:
            return "neutral"
        return "good"
    elif direction == "below":  # valor baixo = ruim
        if z < -1.2:
            return "bad"
        if z < -0.5:
            return "neutral"
        return "good"
    else:  # "any" — qualquer desvio grande e ruim
        if abs(z) > 1.2:
            return "bad"
        if abs(z) > 0.5:
            return "neutral"
        return "good"


def _check_stagnation_anomaly(positions: pd.DataFrame, team_num: int) -> list[str]:
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

        # 1. Concentração espacial excessiva em uma única célula
        h, xedges, yedges = np.histogram2d(
            pgroup["x"].values, pgroup["y"].values,
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

        # 2. Congelamento/imobilidade estrita por amostras consecutivas
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


def _generate_tactical_narrative(
    team_feats: dict[str, float],
    detector,
) -> tuple[str, float]:
    """Gera diagnostico em linguagem natural e nota tatica (0-100).

    Args:
        team_feats: Features calculadas para um time.
        detector: ModelDrivenErrorDetector treinado.

    Returns:
        (texto_narrativo, nota_tatica)
    """
    if not detector.is_built:
        return "Modelo de IA ainda não disponível para diagnóstico.", 0.0

    good_sentences: list[tuple[float, str]] = []   # (importance, text)
    neutral_sentences: list[tuple[float, str]] = []
    bad_sentences: list[tuple[float, str]] = []
    score_parts: list[tuple[float, float]] = []     # (feat_score, importance)

    for feat_name in detector.feature_names:
        val = team_feats.get(feat_name)
        ref_mean = detector.reference_mean.get(feat_name)
        if val is None or ref_mean is None:
            continue

        ref_std = max(detector.reference_std.get(feat_name, 1.0), 0.01)
        importance = detector.feature_importances.get(feat_name, 0.0)
        z = (val - ref_mean) / ref_std

        tmpl = _NARRATIVE_TEMPLATES.get(feat_name)
        if not tmpl:
            continue

        direction = tmpl["direction"]
        tier = _classify_tier(z, direction)

        # Selecionar texto apropriado
        if tier == "bad" and direction == "any":
            text = tmpl.get("bad_above" if z > 0 else "bad_below", tmpl.get("bad", ""))
        else:
            text = tmpl.get(tier, "")

        if not text:
            continue

        bucket = {"good": good_sentences, "neutral": neutral_sentences, "bad": bad_sentences}
        bucket[tier].append((importance, text))

        # Calcular score individual da feature (referencia higher ≈ 75 pts)
        if direction == "below":
            feat_score = 75 + z * 10
        elif direction == "above":
            feat_score = 75 - z * 10
        else:
            feat_score = 85 - abs(z) * 12
        score_parts.append((max(0, min(100, feat_score)), importance))

    # Nota tatica ponderada
    if score_parts:
        total_w = sum(w for _, w in score_parts)
        score = sum(s * w for s, w in score_parts) / total_w if total_w > 0 else 50
    else:
        score = 50.0
    score = max(0, min(100, round(score)))

    # Ordenar cada grupo por importancia (mais relevante primeiro)
    for group in (good_sentences, neutral_sentences, bad_sentences):
        group.sort(key=lambda x: x[0], reverse=True)

    # Montar paragrafo
    parts: list[str] = []

    # Abertura baseada na nota
    if score >= 80:
        parts.append(_SCORE_OVERVIEWS["excellent"])
    elif score >= 60:
        parts.append(_SCORE_OVERVIEWS["good"])
    elif score >= 40:
        parts.append(_SCORE_OVERVIEWS["average"])
    else:
        parts.append(_SCORE_OVERVIEWS["poor"])

    # Pontos positivos (top 2 por importancia)
    for _, text in good_sentences[:2]:
        parts.append(text)

    # Transicao + pontos negativos
    if bad_sentences:
        transitions = [
            "Por outro lado,", "Como ponto de melhoria,",
            "No entanto,", "Em contrapartida,",
        ]
        # Escolher transicao deterministicamente baseada na nota
        trans = transitions[int(score) % len(transitions)]
        first_bad = bad_sentences[0][1]
        # Primeira frase negativa comeca com letra minuscula apos a transicao
        first_bad_lower = first_bad[0].lower() + first_bad[1:] if first_bad else first_bad
        parts.append(f"{trans} {first_bad_lower}")
        # Demais frases negativas
        for _, text in bad_sentences[1:]:
            parts.append(text)
    elif not bad_sentences and len(good_sentences) >= 3:
        # Tudo positivo — fechar com elogio
        parts.append("No geral, o desempenho tático está dentro do esperado para o nível competitivo.")

    narrative = " ".join(parts)
    return narrative, float(score)


# ──────────────────────────────────────────────
#  Helpers & Cache
# ──────────────────────────────────────────────

@st.cache_data
def load_replay_json(json_path: str) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def analyze_replay(json_path: str) -> dict:
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


def get_available_replays() -> dict[str, list[str]]:
    discovered = discover_replays()
    available = {}
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


def _load_feature_dataset():
    """Carrega dataset de features, gerando se necessario. Retorna (df, X, y, feature_names, k) ou None."""
    csv_path = "data/processed/features_dataset.csv"
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    if df.empty or "label" not in df.columns:
        return None
    X, y, feature_names = get_X_y(df)
    k = min(3, len(X) - 1)
    return df, X, y, feature_names, k


_MODEL_PATH = "data/processed/model_detector.joblib"


@st.cache_resource
def _get_trained_detector(_df_hash, X, y, feature_names):
    """Treina ou carrega o detector do disco. Resultado fica em cache do Streamlit."""
    import hashlib

    model_path = _MODEL_PATH

    # Tenta carregar modelo salvo
    if os.path.exists(model_path):
        try:
            detector = ModelDrivenErrorDetector.load(model_path)
            predictions = detector.predict(X)
            return detector, predictions
        except Exception:
            pass  # Modelo corrompido, retreina

    # Treina do zero e salva
    # Precisamos do df original para build_reference, mas nao pode ser hashable
    # Entao reconstruimos a partir do CSV
    csv_path = "data/processed/features_dataset.csv"
    df = pd.read_csv(csv_path)

    detector = ModelDrivenErrorDetector(n_estimators=100, random_state=42)
    detector.build(df, X, y, feature_names)
    detector.save(model_path)
    predictions = detector.predict(X)
    return detector, predictions


# ──────────────────────────────────────────────
#  Visualizacao do Campo
# ──────────────────────────────────────────────

def draw_field(fig: go.Figure) -> None:
    fig.add_shape(
        type="rect",
        x0=-FIELD_HALF_WIDTH, y0=-FIELD_HALF_LENGTH,
        x1=FIELD_HALF_WIDTH, y1=FIELD_HALF_LENGTH,
        line=dict(color="#333333", width=2),
        fillcolor="rgba(30, 60, 30, 0.3)",
    )
    fig.add_shape(
        type="line",
        x0=-FIELD_HALF_WIDTH, y0=0,
        x1=FIELD_HALF_WIDTH, y1=0,
        line=dict(color="#444444", width=1, dash="dash"),
    )
    goal_width = 1800
    goal_depth = 400
    for y_pos, color in [(-FIELD_HALF_LENGTH, "rgba(52, 152, 219, 0.25)"), (FIELD_HALF_LENGTH, "rgba(230, 126, 34, 0.25)")]:
        sign = 1 if y_pos < 0 else -1
        fig.add_shape(
            type="rect",
            x0=-goal_width / 2, y0=y_pos,
            x1=goal_width / 2, y1=y_pos + sign * goal_depth,
            line=dict(color="#555555", width=1),
            fillcolor=color,
        )


def create_rotation_triangle_plot(
    positions_df: pd.DataFrame,
    frame_idx: int,
    players: list[dict],
) -> go.Figure:
    fig = go.Figure()
    draw_field(fig)

    frame_data = positions_df[positions_df["frame"] == frame_idx]
    if frame_data.empty:
        return fig

    for team_num in [0, 1]:
        team_data = frame_data[
            (frame_data["team"] == team_num) & (frame_data["entity"] != "ball")
        ]
        if team_data.empty:
            continue

        color = TEAM_COLORS[team_num]
        team_name = TEAM_NAMES[team_num]

        fig.add_trace(go.Scatter(
            x=team_data["x"].values,
            y=team_data["y"].values,
            mode="markers+text",
            marker=dict(size=14, color=color, line=dict(width=2, color="white")),
            text=team_data["entity"].values,
            textposition="top center",
            textfont=dict(size=10, color=color),
            name=f"Time {team_name}",
            hovertemplate="%{text}<br>x=%{x:.0f}, y=%{y:.0f}<extra></extra>",
        ))

        if len(team_data) >= 3:
            pts = team_data[["x", "y"]].values[:3]
            x_tri = list(pts[:, 0]) + [pts[0, 0]]
            y_tri = list(pts[:, 1]) + [pts[0, 1]]

            fig.add_trace(go.Scatter(
                x=x_tri, y=y_tri,
                mode="lines",
                line=dict(color=color, width=2),
                fill="toself",
                fillcolor=TEAM_FILL_COLORS[team_num],
                name=f"Triangulo {team_name}",
                showlegend=True,
            ))

    ball_data = frame_data[frame_data["entity"] == "ball"]
    if not ball_data.empty:
        fig.add_trace(go.Scatter(
            x=ball_data["x"].values,
            y=ball_data["y"].values,
            mode="markers",
            marker=dict(size=12, color=BALL_COLOR, symbol="circle",
                        line=dict(width=2, color="#888888")),
            name="Bola",
            hovertemplate="Bola<br>x=%{x:.0f}, y=%{y:.0f}<extra></extra>",
        ))

    fig.update_layout(
        xaxis=dict(
            range=[-FIELD_HALF_WIDTH - 500, FIELD_HALF_WIDTH + 500],
            scaleanchor="y", scaleratio=1,
            showgrid=False, zeroline=False,
            title="X (largura)",
            color="#888888",
        ),
        yaxis=dict(
            range=[-FIELD_HALF_LENGTH - 500, FIELD_HALF_LENGTH + 500],
            showgrid=False, zeroline=False,
            title="Y (comprimento)",
            color="#888888",
        ),
        plot_bgcolor=_DARK_BG,
        paper_bgcolor=_DARK_BG,
        font=dict(color="white"),
        height=650,
        legend=dict(
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="#333333",
            font=dict(color="white"),
        ),
        margin=dict(l=50, r=50, t=20, b=50),
    )

    return fig


def render_scoreboard(goals: list[dict], current_time: float, total_time: float):
    blue_goals = sum(1 for g in goals if g.get("player_team") == 0)
    orange_goals = sum(1 for g in goals if g.get("player_team") == 1)

    minutes = int(current_time) // 60
    seconds = int(current_time) % 60

    st.markdown(
        f'<div class="scoreboard">'
        f'<span class="team-score blue">{blue_goals}</span>'
        f'<span class="separator">-</span>'
        f'<span class="match-time">{minutes}:{seconds:02d}</span>'
        f'<span class="separator">-</span>'
        f'<span class="team-score orange">{orange_goals}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────
#  Tab 1: Replay ao Vivo
# ──────────────────────────────────────────────

def tab_replay():
    st.header("Replay ao Vivo")

    available = get_available_replays()
    if not available:
        st.error("Nenhum replay processado encontrado. Execute o parser primeiro.")
        return

    all_replays = []
    for label, replays in available.items():
        for r in replays:
            all_replays.append({
                "display": f"[{label.upper()}] {r['name'][:20]}...",
                "json_path": r["json_path"],
                "label": label,
                "name": r["name"],
            })

    # Se veio da aba Erros Taticos, atualiza o seletor ANTES de criar o widget
    if "pending_replay" in st.session_state:
        st.session_state.replay_selector = st.session_state.pop("pending_replay")

    selected_idx = st.selectbox(
        "Selecione um replay:",
        range(len(all_replays)),
        format_func=lambda i: all_replays[i]["display"],
        key="replay_selector",
    )
    selected = all_replays[selected_idx]

    with st.spinner(f"Analisando {selected['name'][:20]}..."):
        result = analyze_replay(selected["json_path"])

    meta = result["metadata"]
    positions = pd.DataFrame(result["positions"]) if result["positions"] else pd.DataFrame()
    boost = pd.DataFrame(result["boost"]) if result["boost"] else pd.DataFrame()

    # Metricas do replay
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mapa", meta["map_name"])
    with col2:
        st.metric("Placar", f"{meta['team_0_score']} x {meta['team_1_score']}")
    with col3:
        st.metric("Duracao", f"{meta['total_seconds']:.0f}s")
    with col4:
        st.metric("Frames", meta["num_frames"])

    # Tabela de jogadores
    if meta.get("player_stats"):
        with st.expander("Jogadores", expanded=False):
            stats_df = pd.DataFrame(meta["player_stats"])
            stats_df["time_name"] = stats_df["team"].map({0: "Azul", 1: "Laranja"})
            st.dataframe(
                stats_df[["name", "time_name", "score", "goals", "assists", "saves", "shots"]],
                use_container_width=True,
                hide_index=True,
            )

    if positions.empty:
        st.warning("Sem dados de posicao para este replay.")
        return

    # Triangulo de rotacao
    st.markdown(
        '<span class="uu-info">'
        'Coordenadas em UU (Unreal Units). Campo: 10240 × 8192 UU.'
        '</span>',
        unsafe_allow_html=True,
    )

    frames_available = sorted(positions["frame"].unique())
    if not frames_available:
        return

    frame_to_time = {}
    for f in frames_available:
        frame_data = positions[positions["frame"] == f]
        if not frame_data.empty:
            frame_to_time[f] = frame_data.iloc[0]["time"]

    start_time = frame_to_time.get(frames_available[0], 0)

    if "play_active" not in st.session_state:
        st.session_state.play_active = False
    if "current_idx" not in st.session_state:
        st.session_state.current_idx = 0

    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 1])
    with ctrl_col1:
        play_label = "⏸ Pause" if st.session_state.play_active else "▶ Play"
        if st.button(play_label, key="play_btn"):
            st.session_state.play_active = not st.session_state.play_active
            st.rerun()
    with ctrl_col2:
        if st.button("⏹ Reset", key="reset_btn"):
            st.session_state.current_idx = 0
            st.session_state.play_active = False
            st.rerun()
    with ctrl_col3:
        speed = st.selectbox(
            "Velocidade", ["0.25x", "0.5x", "1x", "2x"],
            index=2, key="speed_sel", label_visibility="collapsed",
        )

    slider_val = st.slider(
        "Tempo do jogo",
        min_value=0,
        max_value=len(frames_available) - 1,
        value=st.session_state.current_idx,
        key="manual_slider",
        format="",
    )

    if not st.session_state.play_active and slider_val != st.session_state.current_idx:
        st.session_state.current_idx = slider_val

    current_frame = frames_available[st.session_state.current_idx]
    current_time = frame_to_time.get(current_frame, 0)
    total_time = frame_to_time.get(frames_available[-1], 0)
    game_time = current_time - start_time

    current_goals = [g for g in result["goals"] if g["frame"] <= current_frame]
    render_scoreboard(current_goals, game_time, total_time - start_time)

    fig = create_rotation_triangle_plot(positions, current_frame, result["players"])
    st.plotly_chart(fig, use_container_width=True)

    if st.session_state.play_active:
        speed_map = {"0.25x": 2.0, "0.5x": 1.0, "1x": 0.5, "2x": 0.2}
        delay = speed_map.get(speed, 0.5)
        _time.sleep(delay)
        if st.session_state.current_idx < len(frames_available) - 1:
            st.session_state.current_idx += 1
        else:
            st.session_state.play_active = False
        st.rerun()

    # Metricas do frame (triangulo)
    frame_data = positions[positions["frame"] == current_frame]
    col1, col2 = st.columns(2)

    for team_num, col in [(0, col1), (1, col2)]:
        with col:
            team_name = TEAM_NAMES[team_num]
            team_data = frame_data[
                (frame_data["team"] == team_num) & (frame_data["entity"] != "ball")
            ]
            if len(team_data) >= 3:
                pts = team_data[["x", "y"]].values[:3]
                area = compute_triangle_area(
                    tuple(pts[0]), tuple(pts[1]), tuple(pts[2])
                )
                spread_vals = []
                for i in range(3):
                    for j in range(i + 1, 3):
                        spread_vals.append(compute_distance(tuple(pts[i]), tuple(pts[j])))
                avg_spread = np.mean(spread_vals)

                st.markdown(f"**Time {team_name}**")
                st.markdown(f"- Area do triangulo: **{area:,.0f}** UU²")
                st.markdown(f"- Dispersao media: **{avg_spread:,.0f}** UU")

    # Boost timeline
    if not boost.empty:
        with st.expander("Timeline de Boost", expanded=False):
            fig_boost = go.Figure()
            for player in sorted(boost["player_name"].unique()):
                player_boost = boost[boost["player_name"] == player].sort_values("time")
                team = player_boost.iloc[0]["team"]
                color = TEAM_COLORS.get(team, "#888888")

                fig_boost.add_trace(go.Scatter(
                    x=player_boost["time"],
                    y=player_boost["boost_pct"],
                    mode="lines",
                    name=player,
                    line=dict(color=color, width=1.5),
                    hovertemplate=f"{player}<br>Tempo: %{{x:.1f}}s<br>Boost: %{{y:.1f}}%<extra></extra>",
                ))

            fig_boost.update_layout(
                xaxis_title="Tempo (s)",
                yaxis_title="Boost (%)",
                yaxis=dict(range=[0, 105]),
                plot_bgcolor=_DARK_BG,
                paper_bgcolor=_DARK_BG,
                font=dict(color="white"),
                height=300,
                legend=dict(bgcolor="rgba(0,0,0,0.5)", bordercolor="#333333"),
                margin=dict(l=50, r=50, t=20, b=50),
            )
            st.plotly_chart(fig_boost, use_container_width=True)

    # Gols
    if result["goals"]:
        with st.expander(f"Gols ({len(result['goals'])})", expanded=False):
            for g in result["goals"]:
                team_icon = "🔵" if g["player_team"] == 0 else "🟠"
                team_name = "Azul" if g["player_team"] == 0 else "Laranja"
                st.markdown(f"⚽ **{g['player_name']}** ({team_icon} Time {team_name}) — Frame {g['frame']}")

    # ── Mapa de Calor Espacial ───────────────────
    if not positions.empty:
        st.markdown("---")
        st.subheader("🗺️ Mapas de Calor Espaciais")
        st.caption(
            "O mapa de calor revela as zonas de maior ocupação do time, "
            "evidenciando domínio ofensivo ou pressão sofrida na defesa."
        )

        hm_col_ctrl1, hm_col_ctrl2 = st.columns([1, 1])
        with hm_col_ctrl1:
            hm_team = st.radio(
                "Exibir calor do time:",
                ["Ambos", "Azul (Time 0)", "Laranja (Time 1)"],
                horizontal=True,
                key="hm_team_sel",
            )
        with hm_col_ctrl2:
            hm_metric = st.radio(
                "Métrica de intensidade:",
                ["Ocupação Relativa (% do tempo)", "Ocorrências (Frames)"],
                horizontal=True,
                key="hm_metric_sel",
            )

        x_range = [-FIELD_HALF_WIDTH - 300, FIELD_HALF_WIDTH + 300]
        y_range = [-FIELD_HALF_LENGTH - 300, FIELD_HALF_LENGTH + 300]
        nbinsx, nbinsy = 24, 32

        tdata0 = positions[(positions["team"] == 0) & (positions["entity"] != "ball")]
        tdata1 = positions[(positions["team"] == 1) & (positions["entity"] != "ball")]

        h0, _, _ = np.histogram2d(tdata0["x"], tdata0["y"], bins=[nbinsx, nbinsy], range=[x_range, y_range]) if not tdata0.empty else (np.zeros((1, 1)), 0, 0)
        h1, _, _ = np.histogram2d(tdata1["x"], tdata1["y"], bins=[nbinsx, nbinsy], range=[x_range, y_range]) if not tdata1.empty else (np.zeros((1, 1)), 0, 0)

        is_pct = (hm_metric == "Ocupação Relativa (% do tempo)")

        if is_pct:
            pct0 = (h0 / max(len(tdata0), 1)) * 100.0
            pct1 = (h1 / max(len(tdata1), 1)) * 100.0
            global_max = max(float(pct0.max()), float(pct1.max()), 1.0)
            cbar_title = "% Tempo"
        else:
            global_max = max(float(h0.max()), float(h1.max()), 1.0)
            cbar_title = "Frames"

        step = max(global_max / 16.0, 0.1 if is_pct else 1.0)

        def _heatmap_for_team(team_num: int, color_scale: str) -> go.Figure:
            tdata = positions[
                (positions["team"] == team_num) & (positions["entity"] != "ball")
            ]
            fig_hm = go.Figure()
            draw_field(fig_hm)
            if not tdata.empty:
                trace_kwargs = dict(
                    x=tdata["x"].values,
                    y=tdata["y"].values,
                    colorscale=color_scale,
                    showscale=True,
                    opacity=0.70,
                    nbinsx=nbinsx,
                    nbinsy=nbinsy,
                    zmin=0,
                    zmax=global_max,
                    autocontour=False,
                    ncontours=16,
                    contours=dict(start=0, end=global_max, size=step, showlines=False),
                    colorbar=dict(
                        title=dict(text=cbar_title, font=dict(color="#d0d0d0", size=12)),
                        tickfont=dict(color="#a0a0a0"),
                        thickness=14,
                        len=0.85,
                    ),
                    name=TEAM_NAMES[team_num],
                )
                if is_pct:
                    trace_kwargs["histnorm"] = "percent"
                fig_hm.add_trace(go.Histogram2dContour(**trace_kwargs))
            fig_hm.update_layout(
                xaxis=dict(
                    range=x_range,
                    scaleanchor="y", scaleratio=1,
                    showgrid=False, zeroline=False,
                    title="X (largura)", color="#888888",
                ),
                yaxis=dict(
                    range=y_range,
                    showgrid=False, zeroline=False,
                    title="Y (comprimento)", color="#888888",
                ),
                plot_bgcolor=_DARK_BG, paper_bgcolor=_DARK_BG,
                font=dict(color="white"), height=600,
                margin=dict(l=40, r=40, t=30, b=40),
            )
            return fig_hm

        if hm_team == "Azul (Time 0)":
            st.caption(
                f"ℹ️ **Escala sincronizada compartilhada:** limite máximo de **{global_max:.1f} {cbar_title}**."
            )
            st.plotly_chart(_heatmap_for_team(0, "Blues"), use_container_width=True)
        elif hm_team == "Laranja (Time 1)":
            st.caption(
                f"ℹ️ **Escala sincronizada compartilhada:** limite máximo de **{global_max:.1f} {cbar_title}**."
            )
            st.plotly_chart(_heatmap_for_team(1, "Oranges"), use_container_width=True)
        else:
            st.caption(
                f"ℹ️ **Escala sincronizada compartilhada:** limite de **0 a {global_max:.1f} {cbar_title}** "
                "para comparação visual e estatística direta entre ambos os times."
            )
            hm_col1, hm_col2 = st.columns(2)
            with hm_col1:
                st.markdown("**Time Azul**")
                st.plotly_chart(_heatmap_for_team(0, "Blues"), use_container_width=True)
            with hm_col2:
                st.markdown("**Time Laranja**")
                st.plotly_chart(_heatmap_for_team(1, "Oranges"), use_container_width=True)

    # ── Diagnóstico de IA — Narrativa Tática ──────────
    st.markdown("---")
    st.subheader("🧠 Diagnóstico de IA")
    st.caption(
        "Análise gerada automaticamente pelo modelo de Inteligência Artificial, "
        "comparando o desempenho tático de cada time com o padrão de jogadores de rank alto."
    )

    # Carregar detector (cache/disco)
    _detector_available = False
    if os.path.exists(_MODEL_PATH):
        try:
            _detector = ModelDrivenErrorDetector.load(_MODEL_PATH)
            _detector_available = _detector.is_built
        except Exception:
            _detector_available = False

    if not _detector_available:
        dataset = _load_feature_dataset()
        if dataset is not None:
            df_feat, X_feat, y_feat, feat_names, _ = dataset
            df_hash = f"{len(df_feat)}_{len(feat_names)}"
            _detector, _ = _get_trained_detector(df_hash, X_feat, y_feat, feat_names)
            _detector_available = _detector.is_built

    if not _detector_available:
        st.info(
            "Modelo de IA ainda não treinado. Visite a aba **Modelos** ou "
            "**Erros Táticos** para que o diagnóstico seja gerado."
        )
    elif not positions.empty:
        diag_col1, diag_col2 = st.columns(2)

        for team_num, diag_col in [(0, diag_col1), (1, diag_col2)]:
            with diag_col:
                team_name = TEAM_NAMES[team_num]
                team_color = TEAM_COLORS[team_num]

                team_feats = extract_team_features(positions, boost, team_num)
                if not team_feats:
                    st.warning(f"Time {team_name}: dados insuficientes para diagnóstico.")
                    continue

                narrative, score = _generate_tactical_narrative(team_feats, _detector)
                stagnation_alerts = _check_stagnation_anomaly(positions, team_num)

                # Cor e emoji da nota
                if score >= 80:
                    score_color, score_emoji = "#2ecc71", "🟢"
                elif score >= 60:
                    score_color, score_emoji = "#f1c40f", "🟡"
                elif score >= 40:
                    score_color, score_emoji = "#e67e22", "🟠"
                else:
                    score_color, score_emoji = "#e94560", "🔴"

                alerts_html = "".join([
                    f'<div style="margin-top:0.8rem;padding:0.6rem 0.8rem;background:rgba(230,126,34,0.15);'
                    f'border-radius:8px;border-left:3px solid #e67e22;font-size:0.88rem;color:#f39c12;line-height:1.5;">'
                    f'{alert}</div>'
                    for alert in stagnation_alerts
                ]) if stagnation_alerts else ""

                # Card do diagnóstico
                st.markdown(
                    f'<div style="'
                    f'padding:1.2rem 1.4rem;'
                    f'border-radius:12px;'
                    f'background:linear-gradient(135deg, rgba(26,26,46,0.9) 0%, rgba(22,33,62,0.9) 100%);'
                    f'border-left:5px solid {team_color};'
                    f'margin-bottom:0.5rem;'
                    f'">'
                    # Header: Time + Nota
                    f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.8rem;">'
                    f'<strong style="color:{team_color};font-size:1.1rem">Time {team_name}</strong>'
                    f'<span style="'
                    f'background:{score_color}22;'
                    f'color:{score_color};'
                    f'padding:0.25rem 0.8rem;'
                    f'border-radius:20px;'
                    f'font-weight:700;'
                    f'font-size:0.95rem;'
                    f'">{score_emoji} {score:.0f}/100</span>'
                    f'</div>'
                    # Barra de progresso
                    f'<div style="height:5px;border-radius:3px;background:rgba(255,255,255,0.08);margin-bottom:1rem;">'
                    f'<div style="width:{score:.0f}%;height:100%;border-radius:3px;'
                    f'background:linear-gradient(90deg, {score_color}88, {score_color});"></div>'
                    f'</div>'
                    # Texto narrativo
                    f'<p style="color:#d0d0d0;font-size:0.92rem;line-height:1.65;margin:0;">'
                    f'{narrative}'
                    f'</p>'
                    f'{alerts_html}'
                    f'</div>',
                    unsafe_allow_html=True,
                )


# ──────────────────────────────────────────────
#  Tab 2: Higher vs Lower
# ──────────────────────────────────────────────

def tab_comparison():
    st.header("Higher vs Lower Rank")

    csv_path = "data/processed/features_dataset.csv"
    if not os.path.exists(csv_path):
        st.warning("Dataset de features nao encontrado. Gerando...")
        with st.spinner("Extraindo features de todos os replays..."):
            df = build_dataset()
            if df.empty:
                st.error("Falha ao gerar dataset.")
                return
    else:
        df = pd.read_csv(csv_path)

    if df.empty or "label" not in df.columns:
        st.error("Dataset vazio ou invalido.")
        return

    feature_cols = [c for c in df.columns if c not in _META_COLS]

    higher_means = df[df["label"] == "higher"][feature_cols].mean()
    lower_means = df[df["label"] == "lower"][feature_cols].mean()

    # Normalizar para radar
    all_vals = pd.concat([higher_means, lower_means])
    max_vals = all_vals.groupby(level=0).max().replace(0, 1)
    higher_norm = (higher_means / max_vals * 100).round(1)
    lower_norm = (lower_means / max_vals * 100).round(1)

    # Labels traduzidos para o eixo angular do radar
    translated_cols = [_translate(c) for c in feature_cols]
    categories_translated = translated_cols + [translated_cols[0]]

    # Radar chart
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=list(higher_norm.values) + [higher_norm.values[0]],
        theta=categories_translated, fill="toself", name="Higher (GC1+)",
        line_color="#3498db", fillcolor="rgba(52, 152, 219, 0.2)",
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=list(lower_norm.values) + [lower_norm.values[0]],
        theta=categories_translated, fill="toself", name="Lower (Bronze-Prata)",
        line_color="#e67e22", fillcolor="rgba(230, 126, 34, 0.2)",
    ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor=_DARK_BG,
            radialaxis=dict(visible=True, range=[0, 110], color="#555555"),
            angularaxis=dict(color="#aaaaaa"),
        ),
        plot_bgcolor=_DARK_BG, paper_bgcolor=_DARK_BG,
        font=dict(color="white"), height=480,
        legend=dict(bgcolor="rgba(0,0,0,0.5)"),
        margin=dict(t=30, b=30),
    )
    st.plotly_chart(fig_radar, use_container_width=True)
    st.caption(
        "Compara o formato do estilo de jogo entre os grupos. "
        "Formas mais amplas indicam maior domínio dos fundamentos analisados."
    )

    # Tabela de diferencas (com labels traduzidos)
    st.subheader("Diferenças por Característica")
    diffs = []
    for col in feature_cols:
        h_val = higher_means[col]
        l_val = lower_means[col]
        pct_diff = ((l_val - h_val) / abs(h_val)) * 100 if h_val != 0 else 0
        diffs.append({
            "Característica": _translate(col),
            "Higher": round(h_val, 1),
            "Lower": round(l_val, 1),
            "Diferença %": round(pct_diff, 1),
        })

    diffs_df = pd.DataFrame(diffs).sort_values("Diferença %", key=abs, ascending=False)
    st.dataframe(diffs_df, use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────
#  Tab 3: Modelos de IA
# ──────────────────────────────────────────────

def tab_models():
    st.header("Modelos de IA")

    dataset = _load_feature_dataset()
    if dataset is None:
        st.error("Dataset de features nao encontrado. Va na aba Higher vs Lower primeiro.")
        return

    df, X, y, feature_names, k = dataset

    st.caption(
        f"Dataset: **{len(X)}** amostras · **{len(feature_names)}** features · "
        f"Higher: {sum(y == 'higher')} · Lower: {sum(y == 'lower')}"
    )

    sub1, sub2, sub3 = st.tabs([
        "📋 Comparativo",
        "📊 Importancia",
        "🔬 Clusterizacao",
    ])

    # ── Sub-tab 1: Comparativo ──────────────────

    with sub1:
        st.subheader("Comparativo de Modelos")
        st.markdown(
            "Avaliacao dos **3 modelos supervisionados** usando "
            "**StratifiedKFold (cv=5)** nos mesmos folds."
        )

        bench = ModelBenchmarking(cv=5, random_state=42)
        bench_df = bench.evaluate(X, y, knn_k=k)

        # Tabela
        st.dataframe(
            bench_df.style.set_properties(**{"text-align": "center"}),
            use_container_width=True,
            hide_index=True,
        )

        # Matrizes de confusao
        st.markdown("#### Matrizes de Confusao Normalizadas")
        fig_cm = bench.plot_confusion_matrices(
            save_path="data/processed/models_comparison_cm.png"
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        
        # Detalhes tecnicos (Decision Tree Rules)
        with st.expander("Detalhes Tecnicos (Regras da Decision Tree)", expanded=False):
            st.markdown("Regras extraidas da Decision Tree ajustada com todo o dataset.")
            tree = RankClassifierTree(max_depth=3)
            tree.train(X, y, feature_names)
            st.code(tree.get_rules(), language="text")

    # ── Sub-tab 2: Importancia ──────────────────

    with sub2:
        st.subheader("Importancia das Features (Random Forest)")
        st.markdown(
            "A importancia e calculada via **Mean Decrease in Impurity (MDI)** "
            "do Random Forest — o metodo mais robusto entre os 3 modelos."
        )

        rf = RankClassifierRandomForest(n_estimators=100, random_state=42)
        rf.train(X, y, feature_names)
        rf_imp = rf.get_feature_importances()

        if rf_imp:
            # Aplicar traducao nas chaves do dicionario de importancia
            translated_imp = {_translate(k): v for k, v in rf_imp.items()}

            fig_imp = go.Figure(go.Bar(
                x=list(translated_imp.values()), y=list(translated_imp.keys()),
                orientation="h",
                marker_color="#2ecc71",
                text=[f"{v:.1%}" for v in translated_imp.values()],
                textposition="auto",
            ))
            fig_imp.update_layout(
                xaxis_title="Importância Relativa",
                plot_bgcolor=_DARK_BG, paper_bgcolor=_DARK_BG,
                font=dict(color="white"), height=420,
                margin=dict(l=220, t=20, r=20),
            )
            st.plotly_chart(fig_imp, use_container_width=True)
            st.caption(
                "Mostra o peso de cada variável na decisão da Inteligência Artificial. "
                "Variáveis no topo são os maiores divisores de água entre ranks altos e baixos."
            )

            # Salvar PNG para uso externo
            rf.plot_feature_importances(
                save_path="data/processed/rf_feature_importance.png"
            )

    # ── Sub-tab 3: Clusterizacao ────────────────

    with sub3:
        st.subheader("Clusterizacao Comportamental (K-Means)")
        st.info(
            "Abordagem **nao-supervisionada**: agrupa amostras por perfil tatico "
            "sem usar labels. Nao comparavel com as metricas dos modelos supervisionados."
        )

        n_clusters = st.slider(
            "Numero de clusters (k)", min_value=2, max_value=8, value=3, key="kmeans_k"
        )

        clusterer = TacticalClusterKMeans(n_clusters=n_clusters, random_state=42)

        # Elbow
        col_elbow, col_metrics = st.columns([2, 1])
        with col_elbow:
            fig_elbow = clusterer.elbow_analysis(
                X, k_range=range(2, 9),
                save_path="data/processed/kmeans_elbow.png",
            )
            st.plotly_chart(fig_elbow, use_container_width=True)

        # Fit
        cluster_labels = clusterer.fit_predict(X)

        with col_metrics:
            st.metric(
                "Silhouette Score",
                f"{clusterer.silhouette_:.4f}" if clusterer.silhouette_ is not None else "N/A",
            )
            # Contagens de amostras (mantidas aqui como referencia textual)
            unique, counts = np.unique(cluster_labels, return_counts=True)

        # Donut Chart — distribuicao de amostras por cluster (lado a lado com PCA)
        cluster_colors = [
            "#3498db", "#e67e22", "#2ecc71", "#9b59b6",
            "#e74c3c", "#1abc9c", "#f39c12", "#34495e",
        ]
        donut_labels = [f"Perfil {u}" for u in unique]
        fig_donut = go.Figure(go.Pie(
            labels=donut_labels,
            values=counts,
            hole=0.55,
            marker=dict(colors=cluster_colors[:len(unique)], line=dict(color="#0e1117", width=2)),
            textinfo="label+percent",
            textfont=dict(size=12, color="white"),
            hovertemplate="%{label}<br>Amostras: %{value} (%{percent})<extra></extra>",
        ))
        fig_donut.update_layout(
            title=dict(text="Distribuição dos Perfis Táticos", font=dict(color="white", size=14)),
            plot_bgcolor=_DARK_BG, paper_bgcolor=_DARK_BG,
            font=dict(color="white"), height=360,
            showlegend=True,
            legend=dict(bgcolor="rgba(0,0,0,0.5)", font=dict(color="white")),
            margin=dict(l=20, r=20, t=40, b=20),
        )

        # PCA + Donut lado a lado
        pca_col, donut_col = st.columns([2, 1])
        with pca_col:
            fig_pca = clusterer.plot_pca_clusters(
                X, cluster_labels, feature_names=feature_names,
                save_path="data/processed/kmeans_pca.png",
            )
            st.plotly_chart(fig_pca, use_container_width=True)
            st.caption(
                "Agrupa times por similaridade de comportamento, independentemente do rank, "
                "ajudando a identificar estilos de jogo únicos."
            )
        with donut_col:
            st.plotly_chart(fig_donut, use_container_width=True)

        # Resumo
        with st.expander("Perfil medio por cluster", expanded=False):
            summary_df = clusterer.cluster_summary(df, cluster_labels, feature_names)
            # Renomear colunas de features para PT-BR na exibicao
            summary_df = summary_df.copy()
            summary_df.columns = [
                _translate(c) if c in FEATURE_LABELS else c for c in summary_df.columns
            ]
            st.dataframe(summary_df, use_container_width=True)


# ──────────────────────────────────────────────
#  Tab 4: Erros Taticos
# ──────────────────────────────────────────────

def tab_errors():
    st.header("Erros Taticos")
    st.markdown(
        "Diagnostico alimentado pelo **Random Forest** (90.5% de acuracia). "
        "Os erros sao ponderados pela **importancia que o modelo atribui a cada feature** "
        "na hora de diferenciar ranks altos de baixos."
    )

    dataset = _load_feature_dataset()
    if dataset is None:
        st.error("Dataset de features nao encontrado. Va na aba Higher vs Lower primeiro.")
        return

    df, X, y, feature_names, k = dataset

    # Detector alimentado pelo modelo (com cache e persistencia em disco)
    df_hash = f"{len(df)}_{len(feature_names)}"
    detector, predictions = _get_trained_detector(df_hash, X, y, feature_names)

    if not detector.is_built:
        st.warning("Perfil de referencia nao pode ser construido (sem amostras 'higher').")
        return

    # Monta lookup de replay_name -> indice no seletor da tab Replay ao Vivo
    available = get_available_replays()
    replay_name_to_idx = {}
    counter = 0
    for lbl, replays in available.items():
        for r in replays:
            replay_name_to_idx[r["name"]] = counter
            counter += 1

    total_errors = 0
    for i, (idx, row) in enumerate(df.iterrows()):
        replay_short = row["replay_name"][:25] + "..."
        team_label = row.get("team_label", "?")
        rank_badge = "🔵 Higher" if row["label"] == "higher" else "🟠 Lower"

        # Predicao do modelo
        pred = predictions[i] if i < len(predictions) else "?"
        pred_icon = "✅" if pred == row["label"] else "❌"

        features_dict = {col: row[col] for col in feature_names}
        errors = detector.detect_errors(features_dict)
        total_errors += len(errors)

        icon = "🔴" if any(e["severity"] == "alta" for e in errors) else "🟡" if errors else "🟢"
        with st.expander(
            f"{icon} [{rank_badge}] {replay_short} — Time {team_label} "
            f"({len(errors)} erros) — Pred: {pred} {pred_icon}"
        ):

            # Botao para abrir o replay na aba Replay ao Vivo
            if row["replay_name"] in replay_name_to_idx:
                if st.button("▶ Assistir Replay", key=f"play_{idx}"):
                    st.session_state.pending_replay = replay_name_to_idx[row["replay_name"]]
                    st.rerun()

            if not errors:
                st.success("Nenhum erro significativo detectado pelo modelo.")
            else:
                for err in errors:
                    severity_class = "error-alta" if err["severity"] == "alta" else "error-media"
                    severity_label = "ALTA" if err["severity"] == "alta" else "MEDIA"
                    st.markdown(
                        f'<div class="error-card {severity_class}">'
                        f'<strong>[{severity_label}] {err["type"]}</strong> '
                        f'{err["description"]}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    st.caption(f"Total: {total_errors} erro(s) detectado(s) em {len(df)} amostras.")


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────

def main():
    st.title("RL Analytics")
    st.caption("Analise tatica de partidas 3v3 de Rocket League com Machine Learning")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🎮 Replay ao Vivo",
        "📊 Higher vs Lower",
        "🤖 Modelos",
        "🔍 Erros Taticos",
    ])

    with tab1:
        tab_replay()
    with tab2:
        tab_comparison()
    with tab3:
        tab_models()
    with tab4:
        tab_errors()


if __name__ == "__main__":
    main()
