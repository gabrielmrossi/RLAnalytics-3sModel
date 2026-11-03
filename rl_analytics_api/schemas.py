"""
schemas.py — Contrato REST v1 (Pydantic v2) para `/api/analytics/v1`.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ProblemDetail(BaseModel):
    """Erro padronizado RFC 9457 (application/problem+json)."""

    type: str = Field(default="about:blank", description="URI identificando o tipo do problema")
    title: str = Field(..., description="Resumo legível do tipo do problema")
    status: int = Field(..., description="Código de status HTTP")
    detail: str = Field(..., description="Explicação detalhada do erro ocorrido")
    code: str = Field(
        ...,
        description="Código estável de erro (INVALID_REPLAY, UNSUPPORTED_PLAYLIST, PARSE_TIMEOUT, MODEL_UNAVAILABLE, NOT_FOUND)",
    )


class JobInfo(BaseModel):
    id: str = Field(..., description="Identificador único do job de processamento")
    status: Literal["queued", "running", "done", "failed"] = Field(
        ..., description="Estado atual na máquina de estados do job"
    )
    poll_url: str = Field(..., description="URL relativa para consultar o status do job")


class ReplayUploadResponse(BaseModel):
    replay_id: str = Field(..., description="SHA-256 do conteúdo do arquivo .replay")
    deduplicated: bool = Field(
        ..., description="True se o replay já estava analisado na versão atual do modelo"
    )
    job: JobInfo
    report_url: str | None = Field(
        default=None, description="Preenchido quando deduplicated=True ou concluído via Prefer: wait"
    )


class JobErrorDetail(BaseModel):
    code: str
    message: str


class JobStatusResponse(BaseModel):
    id: str
    replay_id: str
    status: Literal["queued", "running", "done", "failed"]
    stage: str = Field(
        ...,
        description="Etapa interna (queued, parsing, extracting, persisting_parquet, inferring, persisting_report, done, failed)",
    )
    progress: float = Field(..., ge=0.0, le=1.0, description="Progresso estimado de 0.0 a 1.0")
    error: JobErrorDetail | None = None
    result_url: str | None = None


class PlayerStatItem(BaseModel):
    name: str
    team: int
    score: int = 0
    goals: int = 0
    assists: int = 0
    saves: int = 0
    shots: int = 0


class ReplayMetadataResponse(BaseModel):
    replay_id: str
    replay_name: str
    source: Literal["upload", "catalog"]
    catalog_label: str | None = None
    is_training_sample: bool = False
    map_name: str = "unknown"
    team_size: int = 3
    team_0_score: int = 0
    team_1_score: int = 0
    num_frames: int = 0
    total_seconds: float = 0.0
    date: str | None = None
    player_stats: list[PlayerStatItem] = Field(default_factory=list)
    has_spatial_telemetry: bool = True


class ReplayCatalogItem(BaseModel):
    replay_id: str
    replay_name: str
    source: Literal["upload", "catalog"]
    catalog_label: str | None = None
    is_training_sample: bool = False
    map_name: str = "unknown"
    team_0_score: int = 0
    team_1_score: int = 0
    total_seconds: float = 0.0
    team_0_tactical_score: float = 0.0
    team_1_tactical_score: float = 0.0
    team_0_pred: str | None = None
    team_1_pred: str | None = None
    has_spatial_telemetry: bool = True
    created_at: str | None = None


class ReplayCatalogResponse(BaseModel):
    items: list[ReplayCatalogItem]
    next_cursor: str | None = None
    total: int


class ModelVersionInfo(BaseModel):
    version: str
    sklearn: str
    pipeline: str


class PredictionInfo(BaseModel):
    label: str
    proba: dict[str, float]


class FindingItem(BaseModel):
    feature: str
    label: str
    display_name: str | None = None
    value: float
    ref_mean: float
    ref_std: float
    z: float
    mdi: float
    direction: Literal["above", "below", "any"]
    tier: Literal["good", "neutral", "bad"]
    severity: Literal["alta", "media", "baixa", "ok"]
    partial_score: float
    weight: float
    text: str


class StagnationAlertItem(BaseModel):
    type: Literal["stagnation", "immobility"]
    player: str
    x: float
    y: float
    pct_time: float
    seconds: int
    text: str


class TeamDiagnosis(BaseModel):
    team: int
    name: str
    score: float = Field(..., ge=0.0, le=100.0)
    tier: Literal["excellent", "good", "average", "poor"]
    prediction: PredictionInfo
    summary: str
    findings: list[FindingItem]
    alerts: list[StagnationAlertItem] = Field(default_factory=list)


class DiagnosisResponse(BaseModel):
    replay_id: str
    model: ModelVersionInfo
    is_training_sample: bool
    teams: list[TeamDiagnosis]


class HeatmapGridMeta(BaseModel):
    nx: int
    ny: int
    x_range: list[float]
    y_range: list[float]
    cell_w: float
    cell_h: float
    sample_interval_s: float = 1.0
    layout: str | None = None


class HeatmapCell(BaseModel):
    ix: int
    iy: int
    x: float
    y: float
    team_0: int
    team_1: int


class HeatmapCellsResponse(BaseModel):
    grid: HeatmapGridMeta
    entity: Literal["team", "ball"]
    totals: dict[str, int]
    max: dict[str, int]
    cells: list[HeatmapCell]


class HeatmapTeamGrid(BaseModel):
    samples: int
    max: int
    counts: list[int]


class HeatmapGridResponse(BaseModel):
    grid: HeatmapGridMeta
    entity: Literal["team", "ball"]
    teams: dict[str, HeatmapTeamGrid]


class RadarScaleInfo(BaseModel):
    min: float = 0.0
    max: float = 100.0
    meaning: str


class RadarSeriesItem(BaseModel):
    key: str
    label: str
    color: str
    dashed: bool = False


class RadarDataPoint(BaseModel):
    feature: str
    axis: str
    direction: Literal["above", "below", "any"]
    team_0: float
    team_1: float
    ref_higher: float
    ref_lower: float
    raw_team_0: float
    raw_team_1: float
    raw_ref_higher: float | None = None
    raw_ref_lower: float | None = None


class RadarResponse(BaseModel):
    scale: RadarScaleInfo
    series: list[RadarSeriesItem]
    data: list[RadarDataPoint]


class ReplayReportResponse(BaseModel):
    replay_id: str
    replay_name: str
    metadata: ReplayMetadataResponse
    diagnosis: DiagnosisResponse
    heatmap: HeatmapCellsResponse
    radar: RadarResponse
    goals: list[dict[str, Any]] = Field(default_factory=list)
    demolitions: list[dict[str, Any]] = Field(default_factory=list)


class ModelInfoResponse(BaseModel):
    model: ModelVersionInfo
    feature_names: list[str]
    feature_importances: dict[str, float]
    reference_mean: dict[str, float]
    reference_std: dict[str, float]
    provenance: dict[str, Any]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    database: str
    redis: str
    details: dict[str, Any] = Field(default_factory=dict)

