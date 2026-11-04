"""
seed.py — População do catálogo inicial (`analytics.replay_analysis`) a partir do acervo do TCC.

Modos:
1. `seed_catalog_from_csv`: popula instantaneamente (~100 ms) os diagnósticos, features e radares
   das 105 partidas de `features_dataset.csv` (`is_training_sample=True`), e processa também as
   fixtures `.replay` disponíveis para que possuam telemetria espacial (Heatmap 2D) completa.
2. CLI `rl-seed-catalog`: processa todos os arquivos `.replay` em `RAW_DIR` (`data/raw/`) quando montado.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from rl_analytics import __version__ as PIPELINE_VERSION
from rl_analytics.config import (
    DATASET_CSV_PATH,
    MODEL_PATH,
    PARQUET_DIR,
    PROJECT_ROOT,
    RAW_DIR,
    TEAM_NAMES,
)
from rl_analytics.diagnosis import diagnose_team
from rl_analytics.error_detection import ModelDrivenErrorDetector
from rl_analytics.facade import (
    analyze_replay_file,
    build_heatmap_payload,
    build_radar_payload,
    compute_file_sha256,
)
from rl_analytics.parser import discover_replays
from rl_analytics_api import database
from rl_analytics_api.models import ReplayAnalysisRecord
from rl_analytics_api.tasks import upsert_replay_report


def seed_catalog_from_csv(
    session: Session,
    detector: ModelDrivenErrorDetector,
    reference_df: pd.DataFrame,
    *,
    seed_fixture_replays: bool = True,
    force: bool = False,
) -> int:
    """Popula a tabela `analytics.replay_analysis` a partir de `features_dataset.csv` se estiver vazia (ou se force=True)."""
    if not force:
        existing_count = session.scalar(select(func.count()).select_from(ReplayAnalysisRecord)) or 0
        if existing_count > 0:
            return 0

    if reference_df is None or reference_df.empty:
        return 0

    model_sha = compute_file_sha256(MODEL_PATH)[:8] if Path(MODEL_PATH).exists() else "unknown"
    model_info = {
        "version": f"rf-{model_sha}",
        "sklearn": sklearn.__version__,
        "pipeline": PIPELINE_VERSION,
    }
    empty_heatmap = build_heatmap_payload(pd.DataFrame(), bins=(24, 32), entity="team", fmt="cells")

    # Mapear replays que possuem arquivo físico .replay em tests/fixtures/replays
    fixture_replays_dir = PROJECT_ROOT / "tests" / "fixtures" / "replays"
    fixture_map: dict[str, Path] = {}
    if seed_fixture_replays and fixture_replays_dir.exists():
        for rp in fixture_replays_dir.glob("*.replay"):
            fixture_map[rp.stem] = rp

    inserted = 0
    for replay_name, group in reference_df.groupby("replay_name", sort=False):
        rname = str(replay_name)
        label = str(group.iloc[0].get("label", "unlabeled"))

        # Se temos o .replay físico nas fixtures, rodamos a fachada completa (inclui Heatmap + Parquet!)
        if rname in fixture_map:
            try:
                full_report = analyze_replay_file(
                    fixture_map[rname],
                    detector=detector,
                    reference_df=reference_df,
                    persist_parquet=True,
                    parquet_dir=PARQUET_DIR,
                    require_3v3=False,
                )
                upsert_replay_report(
                    session,
                    full_report,
                    source="catalog",
                    catalog_label=label,
                    has_spatial_telemetry=True,
                )
                inserted += 1
                continue
            except Exception as exc:
                print(f"[seed] Aviso ao processar fixture {rname}: {exc}")

        # Caso contrário, reconstrói diagnóstico + radar diretamente das features do CSV
        replay_id = hashlib.sha256(f"catalog:{rname}".encode("utf-8")).hexdigest()
        team_features: dict[int, dict[str, float]] = {}
        teams_diag = []

        for _, row in group.iterrows():
            t_num = int(row.get("team", 0))
            feats = {f: float(row[f]) for f in detector.feature_names if f in row}
            team_features[t_num] = feats

            X_row = np.array([[feats.get(f, 0.0) for f in detector.feature_names]], dtype=float)
            pred_label = str(detector.predict(X_row)[0]) if detector.is_built else "unknown"
            proba_list = detector.predict_proba(X_row)
            proba_dict = proba_list[0] if proba_list else {}
            diag = diagnose_team(feats, detector, strip_html=True)

            teams_diag.append({
                "team": t_num,
                "name": TEAM_NAMES.get(t_num, f"Time {t_num}"),
                "score": diag["score"],
                "tier": diag["tier"],
                "prediction": {"label": pred_label, "proba": proba_dict},
                "summary": diag["summary"],
                "positioning_synthesis": diag.get("positioning_synthesis"),
                "findings": diag["findings"],
                "alerts": [],
            })

        radar_payload = build_radar_payload(
            team_features.get(0, {}),
            team_features.get(1, {}),
            detector,
            reference_df=reference_df,
        )

        synthetic_report = {
            "replay_id": replay_id,
            "replay_name": rname,
            "metadata": {
                "map_name": "dfh_stadium",
                "team_size": 3,
                "team_0_score": 0,
                "team_1_score": 0,
                "num_frames": 9000,
                "total_seconds": 300.0,
                "date": None,
                "replay_name": rname,
                "player_stats": [],
            },
            "goals": [],
            "demolitions": [],
            "features": {"0": team_features.get(0, {}), "1": team_features.get(1, {})},
            "diagnosis": {
                "replay_id": replay_id,
                "model": model_info,
                "is_training_sample": True,
                "teams": teams_diag,
            },
            "heatmap": empty_heatmap,
            "radar": radar_payload,
            "parquet_paths": {},
        }
        upsert_replay_report(
            session,
            synthetic_report,
            source="catalog",
            catalog_label=label,
            has_spatial_telemetry=False,
        )
        inserted += 1

    return inserted


def main() -> int:
    """CLI `rl-seed-catalog`: processa todos os replays brutos de RAW_DIR (quando montado)."""
    database.init_analytics_schema()
    detector = ModelDrivenErrorDetector.load(str(MODEL_PATH))
    ref_df = pd.read_csv(DATASET_CSV_PATH) if Path(DATASET_CSV_PATH).exists() else None

    raw_dir = os.getenv("RAW_DIR", str(RAW_DIR))
    discovered = discover_replays(raw_dir)
    session = database.SessionLocal()
    try:
        if not discovered:
            count = seed_catalog_from_csv(session, detector, ref_df) if ref_df is not None else 0
            print(f"[seed] Nenhum .replay em {raw_dir}; {count} partidas populadas via CSV.")
            return 0

        # Remove registros sintéticos sem Parquet para evitar duplicatas
        deleted_syn = session.execute(
            delete(ReplayAnalysisRecord).where(
                ReplayAnalysisRecord.has_spatial_telemetry == False  # noqa: E712
            )
        ).rowcount
        session.commit()
        print(f"[seed] Removidos {deleted_syn} registros sintéticos tabulares para substituição por 2D.")

        total_ok = 0
        for label, paths in discovered.items():
            if label == "unlabeled":
                continue
            for path in paths:
                try:
                    rep = analyze_replay_file(
                        path,
                        detector=detector,
                        reference_df=ref_df,
                        persist_parquet=True,
                        parquet_dir=PARQUET_DIR,
                        require_3v3=True,
                    )
                    upsert_replay_report(
                        session,
                        rep,
                        source="catalog",
                        catalog_label=label,
                        has_spatial_telemetry=True,
                    )
                    total_ok += 1
                    print(f"[seed] OK ({total_ok}) [{label}] {Path(path).name} -> {rep['replay_id'][:12]}", flush=True)
                except Exception as exc:
                    print(f"[seed] IGNORADO/ERRO em {Path(path).name}: {exc}", flush=True)
        print(f"[seed] Concluído: {total_ok} replays 3v3 com telemetria 2D processados no catálogo.", flush=True)
        return 0
    finally:
        session.close()


if __name__ == "__main__":
    sys.exit(main())

