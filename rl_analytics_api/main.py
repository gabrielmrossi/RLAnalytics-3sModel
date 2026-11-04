"""
main.py — Aplicação FastAPI (`analytics-api`) para `/api/analytics/v1`.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import sklearn
from fastapi import Depends, FastAPI, File, Header, Query, Request, Response, UploadFile
from fastapi.responses import JSONResponse
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from rl_analytics import __version__ as PIPELINE_VERSION
from rl_analytics.config import DATA_DIR, DATASET_CSV_PATH, MODEL_PATH, PARQUET_DIR
from rl_analytics.error_detection import ModelDrivenErrorDetector
from rl_analytics.facade import build_heatmap_payload, compute_file_sha256
from rl_analytics.parser import MAX_REPLAY_BYTES, REPLAY_HEADER_SIGNATURE
from rl_analytics_api import database
from rl_analytics_api.models import AnalysisJobRecord, ReplayAnalysisRecord
from rl_analytics_api.schemas import (
    DiagnosisResponse,
    HealthResponse,
    HeatmapCellsResponse,
    HeatmapGridResponse,
    JobErrorDetail,
    JobInfo,
    JobStatusResponse,
    ModelInfoResponse,
    ModelVersionInfo,
    ProblemDetail,
    RadarResponse,
    ReplayCatalogItem,
    ReplayCatalogResponse,
    ReplayMetadataResponse,
    ReplayReportResponse,
    ReplayUploadResponse,
)
from rl_analytics_api.seed import seed_catalog_from_csv
from rl_analytics_api.tasks import enqueue_replay_analysis, get_redis_connection

API_PREFIX = "/api/analytics/v1"
PROBLEM_MEDIA_TYPE = "application/problem+json"
_PREFER_WAIT_RE = re.compile(r"wait\s*=\s*(\d+)", re.IGNORECASE)
_BINS_RE = re.compile(r"^(\d+)x(\d+)$", re.IGNORECASE)


class ProblemException(Exception):
    """Exceção de domínio convertida automaticamente em resposta RFC 9457 (problem+json)."""

    def __init__(self, status: int, code: str, title: str, detail: str):
        super().__init__(detail)
        self.status = status
        self.code = code
        self.title = title
        self.detail = detail


def _load_app_resources(app: FastAPI) -> None:
    app.state.model_ready = False
    app.state.detector = None
    app.state.reference_df = None
    app.state.model_version_info = ModelVersionInfo(
        version="unavailable",
        sklearn=sklearn.__version__,
        pipeline=PIPELINE_VERSION,
    )

    model_path = Path(os.getenv("MODEL_PATH", str(MODEL_PATH)))
    csv_path = Path(os.getenv("REFERENCE_CSV_PATH", str(DATASET_CSV_PATH)))

    if model_path.exists():
        detector = ModelDrivenErrorDetector.load(str(model_path))
        if detector.is_built:
            model_sha = compute_file_sha256(model_path)[:8]
            app.state.detector = detector
            app.state.model_ready = True
            app.state.model_version_info = ModelVersionInfo(
                version=f"rf-{model_sha}",
                sklearn=sklearn.__version__,
                pipeline=PIPELINE_VERSION,
            )

    if csv_path.exists():
        app.state.reference_df = pd.read_csv(csv_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _load_app_resources(app)
    except Exception as exc:
        print(f"[analytics-api] ERRO ao carregar modelo v2 no startup: {exc}")
        app.state.model_ready = False

    try:
        database.init_analytics_schema()
        auto_seed = os.getenv("AUTO_SEED_FROM_CSV", "true").lower() in ("1", "true", "yes")
        if auto_seed and app.state.model_ready and app.state.reference_df is not None:
            with database.SessionLocal() as session:
                seed_fixtures = os.getenv("AUTO_SEED_FIXTURE_REPLAYS", "true").lower() in (
                    "1",
                    "true",
                    "yes",
                )
                seeded = seed_catalog_from_csv(
                    session,
                    app.state.detector,
                    app.state.reference_df,
                    seed_fixture_replays=seed_fixtures,
                )
                if seeded:
                    print(f"[analytics-api] Catálogo inicial populado com {seeded} partidas.")
    except Exception as exc:
        print(f"[analytics-api] Aviso ao inicializar banco no startup: {exc}")

    yield


app = FastAPI(
    title="RL Analytics TCC API",
    version=PIPELINE_VERSION,
    description=(
        "Motor de análise de telemetria binária, inferência Random Forest e diagnóstico "
        "tático XAI (Z-Score × MDI) integrado ao ecossistema Datafield."
    ),
    docs_url="/api/analytics/docs",
    redoc_url="/api/analytics/redoc",
    openapi_url="/api/analytics/openapi.json",
    lifespan=lifespan,
)


@app.exception_handler(ProblemException)
async def problem_exception_handler(_request: Request, exc: ProblemException) -> JSONResponse:
    body = ProblemDetail(
        type=f"https://datafield.local/problems/{exc.code.lower().replace('_', '-')}",
        title=exc.title,
        status=exc.status,
        detail=exc.detail,
        code=exc.code,
    )
    return JSONResponse(
        status_code=exc.status,
        content=body.model_dump(),
        media_type=PROBLEM_MEDIA_TYPE,
    )


def _check_etag(request: Request, response: Response, replay_id: str, model_version: str) -> bool:
    etag_val = f'"{replay_id}:{model_version}"'
    response.headers["ETag"] = etag_val
    response.headers["Cache-Control"] = "private, max-age=60"
    inm = request.headers.get("if-none-match")
    if inm and (inm.strip() == etag_val or inm.strip() == "*"):
        return True
    return False


def _get_replay_or_404(session: Session, replay_id: str) -> ReplayAnalysisRecord:
    rec = session.get(ReplayAnalysisRecord, replay_id)
    if rec is None:
        # Permite também busca pelo replay_name (stem) para conveniência
        stmt = select(ReplayAnalysisRecord).where(ReplayAnalysisRecord.replay_name == replay_id)
        rec = session.scalars(stmt).first()
    if rec is None:
        raise ProblemException(
            status=404,
            code="NOT_FOUND",
            title="Replay não encontrado",
            detail=f"Nenhum replay processado encontrado para o identificador '{replay_id}'.",
        )
    return rec


def _record_to_metadata(rec: ReplayAnalysisRecord) -> ReplayMetadataResponse:
    meta = rec.metadata_json or {}
    return ReplayMetadataResponse(
        replay_id=rec.replay_id,
        replay_name=rec.replay_name,
        source=rec.source if rec.source in ("upload", "catalog") else "upload",
        catalog_label=rec.catalog_label,
        is_training_sample=bool(rec.is_training_sample),
        map_name=rec.map_name or meta.get("map_name", "unknown"),
        team_size=int(rec.team_size or meta.get("team_size", 3)),
        team_0_score=int(rec.team_0_score or meta.get("team_0_score", 0)),
        team_1_score=int(rec.team_1_score or meta.get("team_1_score", 0)),
        num_frames=int(rec.num_frames or meta.get("num_frames", 0)),
        total_seconds=float(rec.total_seconds or meta.get("total_seconds", 0.0)),
        date=rec.match_date or meta.get("date"),
        player_stats=meta.get("player_stats", []),
        has_spatial_telemetry=bool(rec.has_spatial_telemetry),
    )


# ── Healthchecks & Modelo ──────────────────────────────────────────────────


@app.get(f"{API_PREFIX}/health/live", response_model=HealthResponse, tags=["health"])
async def health_live(request: Request) -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=bool(getattr(request.app.state, "model_ready", False)),
        database="unchecked",
        redis="unchecked",
    )


@app.get(f"{API_PREFIX}/health/ready", response_model=HealthResponse, tags=["health"])
async def health_ready(
    request: Request,
    response: Response,
    db: Session = Depends(database.get_db),
) -> HealthResponse:
    model_ok = bool(getattr(request.app.state, "model_ready", False))
    db_status = "ok"
    try:
        db.execute(text("SELECT 1"))
    except Exception as exc:
        db_status = f"error: {exc}"

    queue_mode = os.getenv("ANALYTICS_QUEUE_MODE", "rq").lower()
    redis_status = "ok (sync-mode)"
    if queue_mode != "sync":
        try:
            conn = get_redis_connection()
            conn.ping()
            redis_status = "ok"
        except Exception as exc:
            redis_status = f"error: {exc}"

    is_ready = model_ok and db_status == "ok" and redis_status.startswith("ok")
    if not is_ready:
        response.status_code = 503

    return HealthResponse(
        status="ok" if is_ready else "degraded",
        model_loaded=model_ok,
        database=db_status,
        redis=redis_status,
        details={
            "model_version": getattr(
                getattr(request.app.state, "model_version_info", None), "version", "unavailable"
            )
        },
    )


@app.get(f"{API_PREFIX}/model", response_model=ModelInfoResponse, tags=["model"])
async def get_model_info(request: Request) -> ModelInfoResponse:
    if not getattr(request.app.state, "model_ready", False) or request.app.state.detector is None:
        raise ProblemException(
            status=503,
            code="MODEL_UNAVAILABLE",
            title="Modelo de IA indisponível",
            detail="O artefato model_detector.v2.joblib não pôde ser carregado em memória.",
        )
    det: ModelDrivenErrorDetector = request.app.state.detector
    return ModelInfoResponse(
        model=request.app.state.model_version_info,
        feature_names=det.feature_names,
        feature_importances=det.feature_importances,
        reference_mean=det.reference_mean,
        reference_std=det.reference_std,
        provenance=det.provenance,
    )


# ── (a) Upload, Jobs e Catálogo ────────────────────────────────────────────


@app.post(
    f"{API_PREFIX}/replays",
    response_model=ReplayUploadResponse,
    status_code=202,
    tags=["replays"],
)
async def upload_replay(
    request: Request,
    response: Response,
    file: UploadFile = File(..., description="Arquivo binário .replay de Rocket League 3v3"),
    prefer: str | None = Header(default=None, alias="Prefer"),
    db: Session = Depends(database.get_db),
) -> ReplayUploadResponse:
    if not getattr(request.app.state, "model_ready", False):
        raise ProblemException(
            status=503,
            code="MODEL_UNAVAILABLE",
            title="Modelo de IA indisponível",
            detail="Serviço ainda não está pronto para processar replays.",
        )

    filename = file.filename or "upload.replay"
    if not filename.lower().endswith(".replay"):
        raise ProblemException(
            status=422,
            code="INVALID_REPLAY",
            title="Arquivo inválido",
            detail="O arquivo enviado deve possuir a extensão '.replay'.",
        )

    uploads_dir = Path(os.getenv("DATA_DIR", str(DATA_DIR))) / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    tmp_upload_path = uploads_dir / f".upload-{uuid.uuid4().hex}.tmp"

    hasher = hashlib.sha256()
    total_bytes = 0
    header_buf = bytearray()

    try:
        with open(tmp_upload_path, "wb") as out_f:
            while True:
                chunk = await file.read(65536)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_REPLAY_BYTES:
                    raise ProblemException(
                        status=422,
                        code="INVALID_REPLAY",
                        title="Arquivo excede o tamanho máximo",
                        detail=f"O replay excede o limite de {MAX_REPLAY_BYTES // (1024 * 1024)} MiB.",
                    )
                if len(header_buf) < 8192:
                    header_buf.extend(chunk[: 8192 - len(header_buf)])
                hasher.update(chunk)
                out_f.write(chunk)

        if total_bytes == 0 or REPLAY_HEADER_SIGNATURE not in header_buf:
            raise ProblemException(
                status=422,
                code="INVALID_REPLAY",
                title="Assinatura de replay inválida",
                detail="Assinatura TAGame.Replay_Soccar_TA não encontrada no cabeçalho binário.",
            )

        replay_id = hasher.hexdigest()
        final_replay_path = uploads_dir / f"{replay_id}.replay"
        os.replace(tmp_upload_path, final_replay_path)
    finally:
        if tmp_upload_path.exists():
            try:
                os.remove(tmp_upload_path)
            except OSError:
                pass

    current_model_ver = request.app.state.model_version_info.version
    existing = db.get(ReplayAnalysisRecord, replay_id)
    if existing is not None and existing.model_version == current_model_ver and existing.has_spatial_telemetry:
        response.status_code = 200
        return ReplayUploadResponse(
            replay_id=replay_id,
            deduplicated=True,
            job=JobInfo(
                id=f"dedup-{replay_id[:16]}",
                status="done",
                poll_url=f"{API_PREFIX}/replays/{replay_id}/report",
            ),
            report_url=f"{API_PREFIX}/replays/{replay_id}/report",
        )

    job_rec = enqueue_replay_analysis(
        db,
        replay_id=replay_id,
        replay_path=str(final_replay_path),
        source="upload",
    )

    # Suporte a Prefer: wait=N (RFC 7240)
    wait_seconds = 0
    if prefer:
        m = _PREFER_WAIT_RE.search(prefer)
        if m:
            wait_seconds = min(10, max(0, int(m.group(1))))

    if wait_seconds > 0 and job_rec.status not in ("done", "failed"):
        deadline = time.monotonic() + wait_seconds
        while time.monotonic() < deadline:
            db.expire_all()
            refreshed = db.get(AnalysisJobRecord, job_rec.job_id)
            if refreshed and refreshed.status in ("done", "failed"):
                job_rec = refreshed
                break
            await asyncio.sleep(0.15)

    if job_rec.status == "failed":
        code = job_rec.error_code or "INVALID_REPLAY"
        status_http = 422 if code in ("INVALID_REPLAY", "UNSUPPORTED_PLAYLIST") else 500
        raise ProblemException(
            status=status_http,
            code=code,
            title="Falha ao processar replay",
            detail=job_rec.error_message or "Erro durante o processamento do replay.",
        )

    if job_rec.status == "done":
        response.status_code = 200
        return ReplayUploadResponse(
            replay_id=replay_id,
            deduplicated=False,
            job=JobInfo(
                id=job_rec.job_id,
                status="done",
                poll_url=f"{API_PREFIX}/jobs/{job_rec.job_id}",
            ),
            report_url=f"{API_PREFIX}/replays/{replay_id}/report",
        )

    response.status_code = 202
    return ReplayUploadResponse(
        replay_id=replay_id,
        deduplicated=False,
        job=JobInfo(
            id=job_rec.job_id,
            status=job_rec.status,  # type: ignore[arg-type]
            poll_url=f"{API_PREFIX}/jobs/{job_rec.job_id}",
        ),
    )


@app.get(f"{API_PREFIX}/jobs/{{job_id}}", response_model=JobStatusResponse, tags=["jobs"])
async def get_job_status(job_id: str, db: Session = Depends(database.get_db)) -> JobStatusResponse:
    rec = db.get(AnalysisJobRecord, job_id)
    if rec is None:
        raise ProblemException(
            status=404,
            code="NOT_FOUND",
            title="Job não encontrado",
            detail=f"Job '{job_id}' não existe.",
        )

    err = (
        JobErrorDetail(code=rec.error_code, message=rec.error_message or "")
        if rec.error_code
        else None
    )
    result_url = f"{API_PREFIX}/replays/{rec.replay_id}/report" if rec.status == "done" else None
    return JobStatusResponse(
        id=rec.job_id,
        replay_id=rec.replay_id,
        status=rec.status,  # type: ignore[arg-type]
        stage=rec.stage,
        progress=float(rec.progress),
        error=err,
        result_url=result_url,
    )


@app.get(f"{API_PREFIX}/replays", response_model=ReplayCatalogResponse, tags=["replays"])
async def list_replays(
    label: str | None = Query(default=None, description="Filtrar por catalog_label (higher, lower)"),
    source: str | None = Query(default=None, description="Filtrar por origem (catalog, upload)"),
    cursor: int = Query(default=0, ge=0, description="Offset/cursor de paginação"),
    limit: int = Query(default=50, ge=1, le=200, description="Quantidade máxima de itens"),
    db: Session = Depends(database.get_db),
) -> ReplayCatalogResponse:
    base_stmt = select(ReplayAnalysisRecord)
    count_stmt = select(func.count()).select_from(ReplayAnalysisRecord)

    if label:
        base_stmt = base_stmt.where(ReplayAnalysisRecord.catalog_label == label)
        count_stmt = count_stmt.where(ReplayAnalysisRecord.catalog_label == label)
    if source:
        base_stmt = base_stmt.where(ReplayAnalysisRecord.source == source)
        count_stmt = count_stmt.where(ReplayAnalysisRecord.source == source)

    total = db.scalar(count_stmt) or 0
    rows = db.scalars(
        base_stmt.order_by(
            ReplayAnalysisRecord.has_spatial_telemetry.desc(),
            ReplayAnalysisRecord.created_at.desc(),
            ReplayAnalysisRecord.replay_name.asc(),
        )
        .offset(cursor)
        .limit(limit)
    ).all()

    items = [
        ReplayCatalogItem(
            replay_id=r.replay_id,
            replay_name=r.replay_name,
            source=r.source if r.source in ("upload", "catalog") else "upload",
            catalog_label=r.catalog_label,
            is_training_sample=bool(r.is_training_sample),
            map_name=r.map_name,
            team_0_score=r.team_0_score,
            team_1_score=r.team_1_score,
            total_seconds=r.total_seconds,
            team_0_tactical_score=r.team_0_tactical_score,
            team_1_tactical_score=r.team_1_tactical_score,
            team_0_pred=r.team_0_pred,
            team_1_pred=r.team_1_pred,
            has_spatial_telemetry=bool(r.has_spatial_telemetry),
            created_at=r.created_at.isoformat() if r.created_at else None,
        )
        for r in rows
    ]

    next_cursor = str(cursor + limit) if (cursor + limit) < total else None
    return ReplayCatalogResponse(items=items, next_cursor=next_cursor, total=total)


@app.get(
    f"{API_PREFIX}/replays/{{replay_id}}",
    response_model=ReplayMetadataResponse,
    tags=["replays"],
)
async def get_replay_metadata(
    replay_id: str, db: Session = Depends(database.get_db)
) -> ReplayMetadataResponse:
    rec = _get_replay_or_404(db, replay_id)
    return _record_to_metadata(rec)


# ── (b) Diagnóstico XAI (Z-Score × MDI) ────────────────────────────────────


@app.get(
    f"{API_PREFIX}/replays/{{replay_id}}/diagnosis",
    response_model=DiagnosisResponse,
    tags=["analytics"],
)
async def get_replay_diagnosis(
    replay_id: str,
    request: Request,
    response: Response,
    db: Session = Depends(database.get_db),
) -> DiagnosisResponse | Response:
    rec = _get_replay_or_404(db, replay_id)
    if _check_etag(request, response, rec.replay_id, rec.model_version):
        return Response(status_code=304, headers=dict(response.headers))
    return DiagnosisResponse.model_validate(rec.diagnosis_json)


# ── (c) Heatmap 2D Espacial ────────────────────────────────────────────────


@app.get(
    f"{API_PREFIX}/replays/{{replay_id}}/heatmap",
    response_model=HeatmapCellsResponse | HeatmapGridResponse,
    tags=["analytics"],
)
async def get_replay_heatmap(
    replay_id: str,
    request: Request,
    response: Response,
    team: Literal["0", "1", "both"] = Query(default="both"),
    bins: str = Query(default="24x32", description="Resolução da grade Nx x Ny (máx 64x80)"),
    entity: Literal["team", "ball"] = Query(default="team"),
    format: Literal["cells", "grid"] = Query(default="cells"),
    db: Session = Depends(database.get_db),
) -> Any:
    rec = _get_replay_or_404(db, replay_id)
    m = _BINS_RE.match(bins.strip())
    if not m:
        raise ProblemException(
            status=422,
            code="INVALID_PARAMETERS",
            title="Formato de bins inválido",
            detail="O parâmetro 'bins' deve seguir o padrão '<nx>x<ny>' (ex: '24x32').",
        )
    nx, ny = int(m.group(1)), int(m.group(2))
    if nx < 4 or nx > 64 or ny < 4 or ny > 80:
        raise ProblemException(
            status=422,
            code="INVALID_PARAMETERS",
            title="Resolução de bins fora do limite",
            detail="A resolução máxima permitida é 64x80 (mínimo 4x4).",
        )

    etag_key = f"{rec.model_version}:{team}:{nx}x{ny}:{entity}:{format}"
    if _check_etag(request, response, rec.replay_id, etag_key):
        return Response(status_code=304, headers=dict(response.headers))

    # Se for o padrão (24x32, team, cells, both), usa o JSONB já pré-calculado
    if (nx, ny) == (24, 32) and entity == "team" and format == "cells" and team == "both":
        return HeatmapCellsResponse.model_validate(rec.heatmap_default_json)

    # Caso contrário, lê do positions.parquet sob demanda (< 5 ms)
    pq_file: Path | None = None
    if rec.parquet_dir:
        candidate = Path(rec.parquet_dir) / "positions.parquet"
        if candidate.exists():
            pq_file = candidate
    if pq_file is None:
        fallback_pq = Path(os.getenv("PARQUET_DIR", str(PARQUET_DIR))) / rec.replay_id / "positions.parquet"
        if fallback_pq.exists():
            pq_file = fallback_pq

    if pq_file is not None and pq_file.exists():
        positions_df = pd.read_parquet(pq_file)
    else:
        positions_df = pd.DataFrame()

    payload = build_heatmap_payload(positions_df, bins=(nx, ny), entity=entity, fmt=format)

    if team in ("0", "1") and format == "cells":
        other_key = "team_1" if team == "0" else "team_0"
        filtered_cells = []
        for c in payload["cells"]:
            c_copy = dict(c)
            c_copy[other_key] = 0
            if c_copy[f"team_{team}"] > 0:
                filtered_cells.append(c_copy)
        payload["cells"] = filtered_cells

    if format == "grid":
        return HeatmapGridResponse.model_validate(payload)
    return HeatmapCellsResponse.model_validate(payload)


# ── (d) Radar Tático (Array Plano Direction-Aware) ─────────────────────────


@app.get(
    f"{API_PREFIX}/replays/{{replay_id}}/radar",
    response_model=RadarResponse,
    tags=["analytics"],
)
async def get_replay_radar(
    replay_id: str,
    request: Request,
    response: Response,
    db: Session = Depends(database.get_db),
) -> RadarResponse | Response:
    rec = _get_replay_or_404(db, replay_id)
    if _check_etag(request, response, rec.replay_id, rec.model_version):
        return Response(status_code=304, headers=dict(response.headers))
    return RadarResponse.model_validate(rec.radar_json)


# ── Relatório Agregado (/report) ───────────────────────────────────────────


@app.get(
    f"{API_PREFIX}/replays/{{replay_id}}/report",
    response_model=ReplayReportResponse,
    tags=["analytics"],
)
async def get_replay_report(
    replay_id: str,
    request: Request,
    response: Response,
    db: Session = Depends(database.get_db),
) -> ReplayReportResponse | Response:
    rec = _get_replay_or_404(db, replay_id)
    if _check_etag(request, response, rec.replay_id, rec.model_version):
        return Response(status_code=304, headers=dict(response.headers))

    return ReplayReportResponse(
        replay_id=rec.replay_id,
        replay_name=rec.replay_name,
        metadata=_record_to_metadata(rec),
        diagnosis=DiagnosisResponse.model_validate(rec.diagnosis_json),
        heatmap=HeatmapCellsResponse.model_validate(rec.heatmap_default_json),
        radar=RadarResponse.model_validate(rec.radar_json),
        goals=rec.goals_json or [],
        demolitions=rec.demolitions_json or [],
    )


# ── (e) Timeline 2D Frame a Frame (/timeline) ──────────────────────────────

from rl_analytics_api.schemas import (  # noqa: E402
    ReplayTimelineResponse,
    TimelineEntitySample,
    TimelineFrameItem,
)


@app.get(
    f"{API_PREFIX}/replays/{{replay_id}}/timeline",
    response_model=ReplayTimelineResponse,
    tags=["analytics"],
)
async def get_replay_timeline(
    replay_id: str,
    request: Request,
    response: Response,
    db: Session = Depends(database.get_db),
) -> ReplayTimelineResponse | Response:
    rec = _get_replay_or_404(db, replay_id)
    if _check_etag(request, response, rec.replay_id, f"{rec.model_version}:timeline"):
        return Response(status_code=304, headers=dict(response.headers))

    pq_file: Path | None = None
    if rec.parquet_dir:
        candidate = Path(rec.parquet_dir) / "positions.parquet"
        if candidate.exists():
            pq_file = candidate
    if pq_file is None:
        fallback_pq = Path(os.getenv("PARQUET_DIR", str(PARQUET_DIR))) / rec.replay_id / "positions.parquet"
        if fallback_pq.exists():
            pq_file = fallback_pq

    if pq_file is None or not pq_file.exists():
        return ReplayTimelineResponse(
            replay_id=rec.replay_id,
            has_spatial_telemetry=False,
            start_time=0.0,
            frames=[],
            goals=rec.goals_json or [],
            demolitions=rec.demolitions_json or [],
        )

    positions_df = pd.read_parquet(pq_file)
    if positions_df.empty:
        return ReplayTimelineResponse(
            replay_id=rec.replay_id,
            has_spatial_telemetry=False,
            start_time=0.0,
            frames=[],
            goals=rec.goals_json or [],
            demolitions=rec.demolitions_json or [],
        )

    positions_df = positions_df.sort_values(["frame", "team", "entity"])
    frames_out: list[TimelineFrameItem] = []
    for frame_val, grp in positions_df.groupby("frame", sort=True):
        frame_time = float(grp.iloc[0]["time"])
        entities: list[TimelineEntitySample] = []
        for row in grp.itertuples(index=False):
            entities.append(
                TimelineEntitySample(
                    entity=str(getattr(row, "entity", "")),
                    team=int(getattr(row, "team", -1)),
                    x=round(float(getattr(row, "x", 0.0)), 1),
                    y=round(float(getattr(row, "y", 0.0)), 1),
                    z=round(float(getattr(row, "z", 0.0)), 1),
                    vx=round(float(getattr(row, "vx", 0.0)), 1),
                    vy=round(float(getattr(row, "vy", 0.0)), 1),
                )
            )
        frames_out.append(
            TimelineFrameItem(
                frame=int(frame_val),
                time=round(frame_time, 3),
                entities=entities,
            )
        )

    start_time = frames_out[0].time if frames_out else 0.0
    return ReplayTimelineResponse(
        replay_id=rec.replay_id,
        has_spatial_telemetry=True,
        start_time=start_time,
        frames=frames_out,
        goals=rec.goals_json or [],
        demolitions=rec.demolitions_json or [],
    )
