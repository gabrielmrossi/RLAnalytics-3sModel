"""
live_replay.py — Aba 1: Visualização do replay ao vivo, triângulo tático, telemetria e diagnósticos de IA.
"""

import time as _time
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.components.field import draw_field
from app.services import (
    analyze_replay,
    check_stagnation_anomaly,
    generate_tactical_narrative,
    get_available_replays,
    get_detector,
    load_feature_dataset,
)
from config import (
    BALL_COLOR,
    DARK_BG,
    FIELD_HALF_LENGTH,
    FIELD_HALF_WIDTH,
    TEAM_COLORS,
    TEAM_FILL_COLORS,
    TEAM_NAMES,
)
from analyzer import compute_distance, compute_triangle_area
from feature_engineering import extract_team_features


def render_scoreboard(goals: list[dict], current_time: float) -> None:
    """Renderiza o placar e o cronômetro do jogo com estilo de HUD esportivo."""
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


def create_rotation_triangle_plot(
    positions_df: pd.DataFrame,
    frame_idx: int,
) -> go.Figure:
    """Renderiza o campo com a posição dos jogadores, triângulo de rotação e bola."""
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
                name=f"Triângulo {team_name}",
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
        plot_bgcolor=DARK_BG,
        paper_bgcolor=DARK_BG,
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


def tab_live_replay() -> None:
    """Aba 1: Visualização do replay frame a frame, telemetria e diagnósticos de IA."""
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

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mapa", meta["map_name"])
    with col2:
        st.metric("Placar", f"{meta['team_0_score']} x {meta['team_1_score']}")
    with col3:
        st.metric("Duração", f"{meta['total_seconds']:.0f}s")
    with col4:
        st.metric("Frames", meta["num_frames"])

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
        st.warning("Sem dados de posição para este replay.")
        return

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
    game_time = current_time - start_time

    current_goals = [g for g in result["goals"] if g["frame"] <= current_frame]
    render_scoreboard(current_goals, game_time)

    fig = create_rotation_triangle_plot(positions, current_frame)
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
                st.markdown(f"- Área do triângulo: **{area:,.0f}** UU²")
                st.markdown(f"- Dispersão média: **{avg_spread:,.0f}** UU")

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
                plot_bgcolor=DARK_BG,
                paper_bgcolor=DARK_BG,
                font=dict(color="white"),
                height=300,
                legend=dict(bgcolor="rgba(0,0,0,0.5)", bordercolor="#333333"),
                margin=dict(l=50, r=50, t=20, b=50),
            )
            st.plotly_chart(fig_boost, use_container_width=True)

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
                plot_bgcolor=DARK_BG, paper_bgcolor=DARK_BG,
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

    dataset = load_feature_dataset()
    _detector = None
    if dataset is not None:
        df_feat, X_feat, y_feat, feat_names, _ = dataset
        df_hash = f"{len(df_feat)}_{len(feat_names)}"
        _detector, _ = get_detector(df_hash, X_feat, y_feat, feat_names)

    if _detector is None or not _detector.is_built:
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

                narrative, score = generate_tactical_narrative(team_feats, _detector)
                stagnation_alerts = check_stagnation_anomaly(positions, team_num)

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

                st.markdown(
                    f'<div style="'
                    f'padding:1.2rem 1.4rem;'
                    f'border-radius:12px;'
                    f'background:linear-gradient(135deg, rgba(26,26,46,0.9) 0%, rgba(22,33,62,0.9) 100%);'
                    f'border-left:5px solid {team_color};'
                    f'margin-bottom:0.5rem;'
                    f'">'
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
                    f'<div style="height:5px;border-radius:3px;background:rgba(255,255,255,0.08);margin-bottom:1rem;">'
                    f'<div style="width:{score:.0f}%;height:100%;border-radius:3px;'
                    f'background:linear-gradient(90deg, {score_color}88, {score_color});"></div>'
                    f'</div>'
                    f'<p style="color:#d0d0d0;font-size:0.92rem;line-height:1.65;margin:0;">'
                    f'{narrative}'
                    f'</p>'
                    f'{alerts_html}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
