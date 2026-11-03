"""
tasks.py — Tarefa assíncrona do worker RQ e persistência atômica no schema `analytics`.
"""

from __future__ import annotations

import datetime as dt
import os
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from rl_analytics.config import DATASET_CSV_PATH, MODEL_PATH, PARQUET_DIR
from rl_analytics.error_detection import ModelDrivenErrorDetector
from rl_analytics.facade import analyze_replay_file
from rl_analytics.parser import ReplayParseError
from rl_analytics_api import database
from rl_analytics_api.models import AnalysisJobRecord, ReplayAnalysisRecord

# Cache global de processo (pré-carregado no bootstrap do worker antes do fork)
_PRELOADED_DETECTOR: ModelDrivenErrorDetector | None = None
_PRELOADED_REFERENCE_DF: pd.DataFrame | None = None


def preload_worker_resources() -> tuple[ModelDrivenErrorDetector, pd.DataFrame | None]:
    """Carrega modelo v2 e CSV de referência na memória do processo pai do worker (CoW no fork)."""
    global _PRELOADED_DETECTOR, _PRELOADED_REFERENCE_DF
    if _PRELOADED_DETECTOR is None:
        _PRELOADED_DETECTOR = ModelDrivenErrorDetector.load(str(MODEL_PATH))
    if _PRELOADED_REFERENCE_DF is None and Path(DATASET_CSV_PATH).exists():
        _PRELOADED_REFERENCE_DF = pd.read_csv(DATASET_CSV_PATH)
    return _PRELOADED_DETECTOR, _PRELOADED_REFERENCE_DF


def _update_job_stage(
    session: Session,
    job_id: str,
    *,
    status: str,
    stage: str,
    progress: float,
    error_code: str | None = None,
    error_message: str | None = None,
) -> None:
    record = session.get(AnalysisJobRecord, job_id)
    if record is not None:
        record.status = status
        record.stage = stage
        record.progress = progress
        record.error_code = error_code
        record.error_message = error_message
        record.updated_at = dt.datetime.now(dt.timezone.utc)
        session.commit()

    # Se estiver dentro de um work-horse do RQ, sincroniza job.meta também
    try:
        from rq import get_current_job

        rq_job = get_current_job()
        if rq_job is not None:
            rq_job.meta["status"] = status
            rq_job.meta["stage"] = stage
            rq_job.meta["progress"] = progress
            if error_code:
                rq_job.meta["error"] = {"code": error_code, "message": error_message or ""}
            rq_job.save_meta()
    except Exception:
        pass


def upsert_replay_report(
    session: Session,
    report: dict[str, Any],
    *,
    source: str = "upload",
    catalog_label: str | None = None,
    has_spatial_telemetry: bool = True,
) -> ReplayAnalysisRecord:
    """Insere ou atualiza o relatório analítico de um replay em `analytics.replay_analysis`."""
    replay_id = str(report["replay_id"])
    meta = report.get("metadata", {})
    diag = report.get("diagnosis", {})
    model_info = diag.get("model", {})
    teams_diag = diag.get("teams", [])

    t0_diag = next((t for t in teams_diag if t.get("team") == 0), {})
    t1_diag = next((t for t in teams_diag if t.get("team") == 1), {})

    pq_paths = report.get("parquet_paths", {})
    pq_dir: str | None = None
    if pq_paths.get("positions"):
        pq_dir = str(Path(pq_paths["positions"]).parent)

    record = session.get(ReplayAnalysisRecord, replay_id)
    now = dt.datetime.now(dt.timezone.utc)
    if record is None:
        record = ReplayAnalysisRecord(replay_id=replay_id, created_at=now)
        session.add(record)

    record.replay_name = str(report.get("replay_name") or meta.get("replay_name") or replay_id[:16])
    record.source = source
    record.catalog_label = catalog_label
    record.is_training_sample = bool(diag.get("is_training_sample", False))
    record.model_version = str(model_info.get("version", "unknown"))
    record.pipeline_version = str(model_info.get("pipeline", "0.2.0"))

    record.map_name = str(meta.get("map_name", "unknown"))
    record.team_size = int(meta.get("team_size", 3) or 3)
    record.team_0_score = int(meta.get("team_0_score", 0) or 0)
    record.team_1_score = int(meta.get("team_1_score", 0) or 0)
    record.num_frames = int(meta.get("num_frames", 0) or 0)
    record.total_seconds = float(meta.get("total_seconds", 0.0) or 0.0)
    record.match_date = meta.get("date")

    record.team_0_tactical_score = float(t0_diag.get("score", 0.0))
    record.team_1_tactical_score = float(t1_diag.get("score", 0.0))
    record.team_0_pred = t0_diag.get("prediction", {}).get("label")
    record.team_1_pred = t1_diag.get("prediction", {}).get("label")

    record.has_spatial_telemetry = has_spatial_telemetry
    record.parquet_dir = pq_dir

    record.metadata_json = meta
    record.features_json = report.get("features", {})
    record.diagnosis_json = diag
    record.heatmap_default_json = report.get("heatmap", {})
    record.radar_json = report.get("radar", {})
    record.goals_json = report.get("goals", [])
    record.demolitions_json = report.get("demolitions", [])
    record.updated_at = now

    session.commit()
    session.refresh(record)
    return record


def process_replay_job(
    replay_id: str,
    replay_path: str,
    job_id: str,
    source: str = "upload",
    catalog_label: str | None = None,
) -> dict[str, Any]:
    """Função executada pelo worker para processar um arquivo .replay ponta a ponta."""
    detector, ref_df = preload_worker_resources()
    session = database.SessionLocal()

    try:
        _update_job_stage(
            session,
            job_id,
            status="running",
            stage="parsing",
            progress=0.25,
        )

        report = analyze_replay_file(
            replay_path,
            detector=detector,
            reference_df=ref_df,
            persist_parquet=True,
            parquet_dir=os.getenv("PARQUET_DIR", str(PARQUET_DIR)),
            require_3v3=True,
        )
        # Garantir replay_id consistente
        report["replay_id"] = replay_id
        report["diagnosis"]["replay_id"] = replay_id

        _update_job_stage(
            session,
            job_id,
            status="running",
            stage="persisting_report",
            progress=0.90,
        )

        upsert_replay_report(
            session,
            report,
            source=source,
            catalog_label=catalog_label,
            has_spatial_telemetry=True,
        )

        _update_job_stage(
            session,
            job_id,
            status="done",
            stage="done",
            progress=1.0,
        )
        return {"replay_id": replay_id, "status": "done"}

    except ReplayParseError as err:
        _update_job_stage(
            session,
            job_id,
            status="failed",
            stage="failed",
            progress=1.0,
            error_code=err.code,
            error_message=err.message,
        )
        raise
    except Exception as exc:
        _update_job_stage(
            session,
            job_id,
            status="failed",
            stage="failed",
            progress=1.0,
            error_code="INTERNAL_ERROR",
            error_message=str(exc)[:500],
        )
        raise
    finally:
        session.close()


def get_redis_connection():
    """Retorna uma conexão Redis a partir de REDIS_URL."""
    import redis

    redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
    return redis.from_url(redis_url)


def enqueue_replay_analysis(
    session: Session,
    *,
    replay_id: str,
    replay_path: str,
    source: str = "upload",
    catalog_label: str | None = None,
) -> AnalysisJobRecord:
    """Registra o job na tabela `analytics.analysis_jobs` e o despacha para a fila RQ (ou sync)."""
    job_id = uuid.uuid4().hex
    job_rec = AnalysisJobRecord(
        job_id=job_id,
        replay_id=replay_id,
        status="queued",
        stage="queued",
        progress=0.0,
    )
    session.add(job_rec)
    session.commit()
    session.refresh(job_rec)

    queue_mode = os.getenv("ANALYTICS_QUEUE_MODE", "rq").lower()
    if queue_mode == "sync":
        try:
            process_replay_job(
                replay_id=replay_id,
                replay_path=replay_path,
                job_id=job_id,
                source=source,
                catalog_label=catalog_label,
            )
        except ReplayParseError:
            pass
        session.expire_all()
        return session.get(AnalysisJobRecord, job_id) or job_rec

    from rq import Queue, Retry

    conn = get_redis_connection()
    q_name = os.getenv("ANALYTICS_QUEUE_NAME", "analytics")
    queue = Queue(name=q_name, connection=conn)
    queue.enqueue(
        process_replay_job,
        kwargs={
            "replay_id": replay_id,
            "replay_path": replay_path,
            "job_id": job_id,
            "source": source,
            "catalog_label": catalog_label,
        },
        job_id=job_id,
        job_timeout=int(os.getenv("ANALYTICS_JOB_TIMEOUT", "60")),
        result_ttl=86400,
        failure_ttl=86400,
        retry=Retry(max=1, interval=[2]),
    )
    return job_rec

