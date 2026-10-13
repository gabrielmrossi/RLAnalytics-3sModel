"""
feature_engineering.py -- Calcula features taticas por time por replay.

Extrai ~10 features numericas que capturam padroes de rotacao, posicionamento,
boost e ritmo de jogo. Produz um DataFrame pronto para o modelo KNN.
"""

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analyzer import ReplayAnalyzer, compute_triangle_area, compute_distance
from parser import discover_replays, PROCESSED_DIR
from config import BALL_CHASE_DIST, META_COLS


def _safe_std(arr: np.ndarray) -> float:
    """Retorna std seguro (0 se array vazio ou len < 2)."""
    if len(arr) < 2:
        return 0.0
    return float(np.std(arr, ddof=1))


def _speed_from_velocity(vx: float, vy: float, vz: float) -> float:
    """Velocidade escalar a partir de componentes."""
    return math.sqrt(vx ** 2 + vy ** 2 + vz ** 2)


def extract_team_features(
    positions_df: pd.DataFrame,
    boost_df: pd.DataFrame,
    team: int,
    team_size: int = 3,
) -> dict[str, float]:
    """Calcula features para 1 time em 1 replay.

    Args:
        positions_df: DataFrame de posicoes (do analyzer).
        boost_df: DataFrame de boost (do analyzer).
        team: 0 (azul) ou 1 (laranja).
        team_size: Numero de jogadores por time (padrao=3).

    Returns:
        Dict com ~10 features numericas.
    """
    if positions_df.empty:
        return {}

    # Separar jogadores do time e a bola
    team_pos = positions_df[positions_df["team"] == team]
    ball_pos = positions_df[positions_df["entity"] == "ball"]

    if team_pos.empty or ball_pos.empty:
        return {}

    # Obter nomes dos jogadores do time
    team_players = sorted(team_pos["entity"].unique())

    # Se nao temos exatamente team_size jogadores, ajustamos
    if len(team_players) < 2:
        return {}

    # ── Extração de features em passada única por frame ──
    frames_sampled = sorted(team_pos["frame"].unique())

    # Mapa O(1) de posição da bola por frame
    ball_map: dict[int, tuple[float, float]] = {
        int(r["frame"]): (float(r["x"]), float(r["y"]))
        for _, r in ball_pos[["frame", "x", "y"]].iterrows()
    }

    team_grouped = team_pos.groupby("frame")

    frame_dists_to_ball: list[float] = []
    min_dists: list[float] = []
    spreads: list[float] = []
    triangle_areas: list[float] = []
    defensive_frames = 0
    total_frames = 0
    chase_frames = 0
    chase_total = 0

    for frame_idx in frames_sampled:
        if frame_idx not in team_grouped.groups:
            continue
        group = team_grouped.get_group(frame_idx)
        if group.empty:
            continue

        coords = list(zip(group["x"].values, group["y"].values))
        n_players = len(coords)

        # 1. Distâncias à bola e Ball chasing
        ball_coord = ball_map.get(frame_idx)
        if ball_coord is not None:
            bx, by = ball_coord
            dists = [compute_distance(p, (bx, by)) for p in coords]
            if dists:
                frame_dists_to_ball.append(float(np.mean(dists)))
                min_dists.append(float(min(dists)))

            chase_total += 1
            close_count = sum(1 for d in dists if d < BALL_CHASE_DIST)
            if close_count >= 2:
                chase_frames += 1

        # 2. Team spread (dispersão média dos pares)
        if n_players >= 2:
            pair_dists = [
                compute_distance(coords[i], coords[j])
                for i in range(n_players)
                for j in range(i + 1, n_players)
            ]
            spreads.append(float(np.mean(pair_dists)))

        # 3. Área do triângulo (3 jogadores)
        if n_players >= 3:
            area = compute_triangle_area(coords[0], coords[1], coords[2])
            triangle_areas.append(area)

        # 4. Cobertura defensiva
        total_frames += 1
        y_vals = group["y"].values
        if team == 0:
            has_defender = bool(np.any(y_vals < 0))
        else:
            has_defender = bool(np.any(y_vals > 0))

        if has_defender:
            defensive_frames += 1

    # Agregações
    avg_dist_to_ball = float(np.mean(frame_dists_to_ball)) if frame_dists_to_ball else 0.0
    min_player_dist_to_ball = float(np.mean(min_dists)) if min_dists else 0.0
    team_spread_mean = float(np.mean(spreads)) if spreads else 0.0
    team_spread_std = _safe_std(np.array(spreads))
    triangle_area_mean = float(np.mean(triangle_areas)) if triangle_areas else 0.0
    triangle_area_std = _safe_std(np.array(triangle_areas))
    defensive_coverage_pct = (
        (defensive_frames / total_frames * 100.0) if total_frames > 0 else 0.0
    )
    ball_chase_ratio = (chase_frames / chase_total * 100.0) if chase_total > 0 else 0.0

    # ── Feature 9: avg_boost_pct ──
    if not boost_df.empty:
        team_boost = boost_df[boost_df["team"] == team]
        avg_boost_pct = float(team_boost["boost_pct"].mean()) if not team_boost.empty else 50.0
    else:
        avg_boost_pct = 50.0

    # ── Feature 10: avg_speed (vetorizado) ──
    vx = team_pos["vx"].fillna(0).to_numpy()
    vy = team_pos["vy"].fillna(0).to_numpy()
    vz = team_pos["vz"].fillna(0).to_numpy()
    speeds = np.sqrt(vx * vx + vy * vy + vz * vz)
    avg_speed = float(np.mean(speeds)) if len(speeds) > 0 else 0.0

    return {
        "avg_dist_to_ball": round(avg_dist_to_ball, 1),
        "min_player_dist_to_ball": round(min_player_dist_to_ball, 1),
        "team_spread_mean": round(team_spread_mean, 1),
        "team_spread_std": round(team_spread_std, 1),
        "triangle_area_mean": round(triangle_area_mean, 1),
        "triangle_area_std": round(triangle_area_std, 1),
        "defensive_coverage_pct": round(defensive_coverage_pct, 1),
        "ball_chase_ratio": round(ball_chase_ratio, 1),
        "avg_boost_pct": round(avg_boost_pct, 1),
        "avg_speed": round(avg_speed, 1),
    }


def extract_replay_features(
    analysis: dict[str, Any],
    label: str,
    replay_name: str = "",
) -> list[dict]:
    """Calcula features para ambos os times de 1 replay.

    Args:
        analysis: Resultado de ReplayAnalyzer.analyze().
        label: "higher", "lower", ou "unlabeled".
        replay_name: Nome do arquivo (para referencia).

    Returns:
        Lista com 2 dicts (1 por time), cada um com label + features.
    """
    positions = analysis.get("positions", pd.DataFrame())
    boost = analysis.get("boost", pd.DataFrame())
    meta = analysis.get("metadata", {})
    team_size = meta.get("team_size", 3)

    results = []
    for team_num in [0, 1]:
        features = extract_team_features(positions, boost, team_num, team_size)
        if features:
            features["label"] = label
            features["team"] = team_num
            features["team_label"] = "Blue" if team_num == 0 else "Orange"
            features["replay_name"] = replay_name
            results.append(features)

    return results


def build_dataset(
    processed_dir: str = PROCESSED_DIR,
    raw_dir: str = "data/raw",
) -> pd.DataFrame:
    """Carrega todos os replays categorizados, extrai features e monta dataset.

    Returns:
        DataFrame com colunas: label, team, team_label, replay_name + 10 features.
        Apenas replays "higher" e "lower" (ignora "unlabeled").
    """
    discovered = discover_replays(raw_dir)

    all_features: list[dict] = []

    for label in ["higher", "lower"]:
        replay_paths = discovered.get(label, [])
        if not replay_paths:
            print(f"[features] Nenhum replay '{label}' encontrado.")
            continue

        for replay_path in replay_paths:
            stem = Path(replay_path).stem
            json_path = Path(processed_dir) / f"{stem}.json"

            if not json_path.exists():
                print(f"[features] JSON nao encontrado para {stem}, pulando.")
                continue

            print(f"[features] Processando {stem} ({label})...")
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            analyzer = ReplayAnalyzer(data)
            analysis = analyzer.analyze()

            replay_features = extract_replay_features(analysis, label, stem)
            all_features.extend(replay_features)

    if not all_features:
        print("[features] Nenhuma feature extraida!")
        return pd.DataFrame()

    df = pd.DataFrame(all_features)

    # Reordenar colunas: metadata primeiro, features depois
    meta_cols = ["label", "replay_name", "team", "team_label"]
    feature_cols = [c for c in df.columns if c not in META_COLS]
    df = df[meta_cols + sorted(feature_cols)]

    print(f"[features] Dataset: {len(df)} linhas, {len(feature_cols)} features.")
    return df


if __name__ == "__main__":
    print("=== Construindo dataset de features ===\n")
    df = build_dataset()

    if not df.empty:
        print(f"\n=== Dataset ({len(df)} linhas) ===")
        print(df.to_string(index=False))

        # Resumo por grupo
        feature_cols = [
            c for c in df.columns
            if c not in ["label", "replay_name", "team", "team_label"]
        ]
        print(f"\n=== Medias por label ===")
        grouped = df.groupby("label")[feature_cols].mean()
        print(grouped.round(1).to_string())

        # Salvar dataset
        out_path = "data/processed/features_dataset.csv"
        df.to_csv(out_path, index=False)
        print(f"\nDataset salvo em: {out_path}")
