"""
facade.py — Fachada única de domínio para análise ponta a ponta de replays de Rocket League 3v3.

Unifica:
- Decodificação segura (.replay -> rrrocket -> ReplayAnalyzer -> deleção imediata do JSON bruto -> Parquet)
- Extração de features táticas por time (Azul=0, Laranja=1)
- Inferência Random Forest (<10 ms) + probabilidades + diagnóstico narrativo XAI (Z-Score × MDI)
- Geração de payloads estruturados de Heatmap 2D (cells / grid) e Radar direction-aware (array plano)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn

from rl_analytics import __version__ as PIPELINE_VERSION
from rl_analytics.config import (
    DATASET_CSV_PATH,
    FIELD_HALF_LENGTH,
    FIELD_HALF_WIDTH,
    META_COLS,
    MODEL_PATH,
    PARQUET_DIR,
    TEAM_COLORS,
    TEAM_NAMES,
)
from rl_analytics.diagnosis import (
    compute_partial_feature_score,
    detect_stagnation_anomalies,
    diagnose_team,
)
from rl_analytics.error_detection import ModelDrivenErrorDetector
from rl_analytics.feature_catalog import (
    FEATURE_CATALOG,
    compute_z_score,
    get_feature_label,
)
from rl_analytics.feature_engineering import extract_team_features
from rl_analytics.parser import (
    ReplayParseError,
    parse_and_extract_With_lifecycle,
)


def compute_file_sha256(path: str | Path) -> str:
    """Calcula o SHA-256 de um arquivo em streaming."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_heatmap_payload(
    positions: pd.DataFrame,
    *,
    bins: tuple[int, int] = (24, 32),
    entity: str = "team",
    fmt: str = "cells",
) -> dict[str, Any]:
    """Gera matriz de densidade espacial 2D no formato 'cells' (array plano esparso) ou 'grid'."""
    nx, ny = max(4, min(64, int(bins[0]))), max(4, min(80, int(bins[1])))
    x_range = [-float(FIELD_HALF_WIDTH + 300), float(FIELD_HALF_WIDTH + 300)]
    y_range = [-float(FIELD_HALF_LENGTH + 300), float(FIELD_HALF_LENGTH + 300)]
    cell_w = round((x_range[1] - x_range[0]) / nx, 2)
    cell_h = round((y_range[1] - y_range[0]) / ny, 2)

    grid_meta = {
        "nx": nx,
        "ny": ny,
        "x_range": x_range,
        "y_range": y_range,
        "cell_w": cell_w,
        "cell_h": cell_h,
        "sample_interval_s": 1.0,
    }

    if positions.empty:
        return {
            "grid": grid_meta,
            "entity": entity,
            "totals": {"team_0": 0, "team_1": 0},
            "max": {"team_0": 0, "team_1": 0, "shared": 0},
            "cells": [],
        }

    if entity == "ball":
        bdata = positions[positions["entity"] == "ball"]
        hb, xedges, yedges = (
            np.histogram2d(bdata["x"], bdata["y"], bins=[nx, ny], range=[x_range, y_range])
            if not bdata.empty
            else (np.zeros((nx, ny)), np.linspace(*x_range, nx + 1), np.linspace(*y_range, ny + 1))
        )
        h0 = hb.astype(int)
        h1 = np.zeros((nx, ny), dtype=int)
        tot0, tot1 = int(len(bdata)), 0
    else:
        tdata0 = positions[(positions["team"] == 0) & (positions["entity"] != "ball")]
        tdata1 = positions[(positions["team"] == 1) & (positions["entity"] != "ball")]
        h0_f, xedges, yedges = (
            np.histogram2d(tdata0["x"], tdata0["y"], bins=[nx, ny], range=[x_range, y_range])
            if not tdata0.empty
            else (np.zeros((nx, ny)), np.linspace(*x_range, nx + 1), np.linspace(*y_range, ny + 1))
        )
        h1_f, _, _ = (
            np.histogram2d(tdata1["x"], tdata1["y"], bins=[nx, ny], range=[x_range, y_range])
            if not tdata1.empty
            else (np.zeros((nx, ny)), xedges, yedges)
        )
        h0 = h0_f.astype(int)
        h1 = h1_f.astype(int)
        tot0, tot1 = int(len(tdata0)), int(len(tdata1))

    max0 = int(h0.max()) if h0.size else 0
    max1 = int(h1.max()) if h1.size else 0
    shared_max = max(max0, max1)

    if fmt == "grid":
        # Row-major: y fora, x dentro
        counts0 = h0.T.flatten().tolist()
        counts1 = h1.T.flatten().tolist()
        return {
            "grid": {**grid_meta, "layout": "row-major, y fora / x dentro"},
            "entity": entity,
            "teams": {
                "0": {"samples": tot0, "max": max0, "counts": counts0},
                "1": {"samples": tot1, "max": max1, "counts": counts1},
            },
        }

    cells: list[dict[str, Any]] = []
    for iy in range(ny):
        cy = round(float((yedges[iy] + yedges[iy + 1]) / 2.0), 1)
        for ix in range(nx):
            c0 = int(h0[ix, iy])
            c1 = int(h1[ix, iy])
            if c0 > 0 or c1 > 0:
                cx = round(float((xedges[ix] + xedges[ix + 1]) / 2.0), 1)
                cells.append({
                    "ix": ix,
                    "iy": iy,
                    "x": cx,
                    "y": cy,
                    "team_0": c0,
                    "team_1": c1,
                })

    return {
        "grid": grid_meta,
        "entity": entity,
        "totals": {"team_0": tot0, "team_1": tot1},
        "max": {"team_0": max0, "team_1": max1, "shared": shared_max},
        "cells": cells,
    }


def build_radar_payload(
    team_0_feats: dict[str, float],
    team_1_feats: dict[str, float],
    detector: ModelDrivenErrorDetector,
    reference_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Gera o payload do gráfico de Radar em array plano direction-aware (0–100)."""
    lower_means: dict[str, float] = {}
    if reference_df is not None and not reference_df.empty and "label" in reference_df.columns:
        lower_df = reference_df[reference_df["label"] == "lower"]
        if not lower_df.empty:
            for col in detector.feature_names:
                if col in lower_df.columns:
                    lower_means[col] = float(lower_df[col].mean())

    data_rows: list[dict[str, Any]] = []
    for feat_name in detector.feature_names:
        ref_mean = float(detector.reference_mean.get(feat_name, 0.0))
        ref_std = max(float(detector.reference_std.get(feat_name, 1.0)), 0.01)
        direction = FEATURE_CATALOG.get(feat_name, {}).get("direction", "any")

        val0 = float(team_0_feats.get(feat_name, ref_mean))
        val1 = float(team_1_feats.get(feat_name, ref_mean))
        val_low = float(lower_means.get(feat_name, ref_mean))

        z0 = compute_z_score(val0, ref_mean, ref_std)
        z1 = compute_z_score(val1, ref_mean, ref_std)
        z_low = compute_z_score(val_low, ref_mean, ref_std)

        score0 = round(compute_partial_feature_score(z0, direction), 1)
        score1 = round(compute_partial_feature_score(z1, direction), 1)
        score_high = round(compute_partial_feature_score(0.0, direction), 1)
        score_low = round(compute_partial_feature_score(z_low, direction), 1)

        data_rows.append({
            "feature": feat_name,
            "axis": get_feature_label(feat_name),
            "direction": direction,
            "team_0": score0,
            "team_1": score1,
            "ref_higher": score_high,
            "ref_lower": score_low,
            "raw_team_0": round(val0, 2),
            "raw_team_1": round(val1, 2),
            "raw_ref_higher": round(ref_mean, 2),
            "raw_ref_lower": round(val_low, 2),
        })

    return {
        "scale": {
            "min": 0,
            "max": 100,
            "meaning": "100 = mais próximo/superior ao perfil higher (ajustado pela direção da feature)",
        },
        "series": [
            {"key": "team_0", "label": TEAM_NAMES[0], "color": TEAM_COLORS[0]},
            {"key": "team_1", "label": TEAM_NAMES[1], "color": TEAM_COLORS[1]},
            {"key": "ref_higher", "label": "Média Higher", "color": "#2ecc71", "dashed": True},
            {"key": "ref_lower", "label": "Média Lower", "color": "#e94560", "dashed": True},
        ],
        "data": data_rows,
    }


def analyze_replay_file(
    replay_path: str | Path,
    *,
    detector: ModelDrivenErrorDetector | None = None,
    reference_df: pd.DataFrame | None = None,
    persist_parquet: bool = True,
    parquet_dir: str | Path | None = None,
    require_3v3: bool = True,
) -> dict[str, Any]:
    """Executa o pipeline completo sobre um arquivo .replay e retorna o relatório unificado."""
    path = Path(replay_path)
    replay_id = compute_file_sha256(path)

    if detector is None:
        detector = ModelDrivenErrorDetector.load(str(MODEL_PATH))

    if reference_df is None and Path(DATASET_CSV_PATH).exists():
        reference_df = pd.read_csv(DATASET_CSV_PATH)

    extracted = parse_and_extract_With_lifecycle(
        path,
        replay_id=replay_id,
        parquet_dir=parquet_dir or PARQUET_DIR,
        persist_parquet=persist_parquet,
    )

    meta = extracted["metadata"]
    team_size = meta.get("team_size", 3)
    if require_3v3 and team_size and int(team_size) != 3:
        raise ReplayParseError(
            "UNSUPPORTED_PLAYLIST",
            f"Modelo treinado apenas para 3v3 (team_size={team_size}).",
        )

    positions: pd.DataFrame = extracted["positions"]
    boost: pd.DataFrame = extracted["boost"]

    team_features: dict[int, dict[str, float]] = {}
    teams_diagnosis: list[dict[str, Any]] = []

    for team_num in (0, 1):
        feats = extract_team_features(positions, boost, team_num)
        team_features[team_num] = feats
        if not feats:
            continue

        X_row = np.array([[feats.get(f, 0.0) for f in detector.feature_names]], dtype=float)
        pred_label = str(detector.predict(X_row)[0]) if detector.is_built else "unknown"
        proba_list = detector.predict_proba(X_row)
        proba_dict = proba_list[0] if proba_list else {}

        diag = diagnose_team(feats, detector, strip_html=True)
        alerts = detect_stagnation_anomalies(positions, team_num)

        teams_diagnosis.append({
            "team": team_num,
            "name": TEAM_NAMES.get(team_num, f"Time {team_num}"),
            "score": diag["score"],
            "tier": diag["tier"],
            "prediction": {
                "label": pred_label,
                "proba": proba_dict,
            },
            "summary": diag["summary"],
            "positioning_synthesis": diag.get("positioning_synthesis"),
            "findings": diag["findings"],
            "alerts": [
                {
                    "type": a["type"],
                    "player": a["player"],
                    "x": a["x"],
                    "y": a["y"],
                    "pct_time": a["pct_time"],
                    "seconds": a["seconds"],
                    "text": a["text"],
                }
                for a in alerts
            ],
        })

    is_training_sample = False
    if reference_df is not None and "replay_name" in reference_df.columns:
        is_training_sample = bool((reference_df["replay_name"] == path.stem).any())

    model_sha = compute_file_sha256(MODEL_PATH)[:8] if Path(MODEL_PATH).exists() else "unknown"
    model_info = {
        "version": f"rf-{model_sha}",
        "sklearn": sklearn.__version__,
        "pipeline": PIPELINE_VERSION,
    }

    heatmap_payload = build_heatmap_payload(positions, bins=(24, 32), entity="team", fmt="cells")
    radar_payload = build_radar_payload(
        team_features.get(0, {}),
        team_features.get(1, {}),
        detector,
        reference_df=reference_df,
    )

    return {
        "replay_id": replay_id,
        "replay_name": path.stem,
        "metadata": meta,
        "goals": extracted["goals"],
        "demolitions": extracted["demolitions"],
        "features": {"0": team_features.get(0, {}), "1": team_features.get(1, {})},
        "diagnosis": {
            "replay_id": replay_id,
            "model": model_info,
            "is_training_sample": is_training_sample,
            "teams": teams_diagnosis,
        },
        "heatmap": heatmap_payload,
        "radar": radar_payload,
        "parquet_paths": extracted.get("parquet_paths", {}),
        "raw_json_bytes": extracted.get("raw_json_bytes", 0),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rl-analyze",
        description="CLI da fachada única rl_analytics para análise tática de replays (.replay).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_an = sub.add_parser("analyze", help="Analisa um arquivo .replay e imprime o relatório JSON.")
    p_an.add_argument("replay_path", type=str, help="Caminho para o arquivo .replay")
    p_an.add_argument(
        "--no-parquet",
        action="store_true",
        help="Não persistir arquivos .parquet em disco durante a execução.",
    )

    args = parser.parse_args(argv)
    if args.cmd == "analyze":
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        report = analyze_replay_file(
            args.replay_path,
            persist_parquet=not args.no_parquet,
        )
        try:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        except BrokenPipeError:
            pass
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())

