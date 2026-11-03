"""
models.py — Modelos ORM SQLAlchemy no schema `analytics` do PostgreSQL (D2).
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import JSON, Boolean, Column, DateTime, Float, Integer, String, Text
from sqlalchemy.dialects import postgresql

from rl_analytics_api.database import ANALYTICS_SCHEMA, Base

JSONB_VARIANT = JSON().with_variant(postgresql.JSONB(astext_type=Text()), "postgresql")


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class ReplayAnalysisRecord(Base):
    """Registro persistido de análise tática, features, radar e heatmap de um replay."""

    __tablename__ = "replay_analysis"
    __table_args__ = {"schema": ANALYTICS_SCHEMA}

    replay_id = Column(String(64), primary_key=True, index=True)
    replay_name = Column(String(128), nullable=False, index=True)
    source = Column(String(16), nullable=False, default="upload", index=True)
    catalog_label = Column(String(16), nullable=True, index=True)
    is_training_sample = Column(Boolean, nullable=False, default=False)

    model_version = Column(String(64), nullable=False, index=True)
    pipeline_version = Column(String(32), nullable=False)

    map_name = Column(String(64), nullable=False, default="unknown")
    team_size = Column(Integer, nullable=False, default=3)
    team_0_score = Column(Integer, nullable=False, default=0)
    team_1_score = Column(Integer, nullable=False, default=0)
    num_frames = Column(Integer, nullable=False, default=0)
    total_seconds = Column(Float, nullable=False, default=0.0)
    match_date = Column(String(64), nullable=True)

    team_0_tactical_score = Column(Float, nullable=False, default=0.0)
    team_1_tactical_score = Column(Float, nullable=False, default=0.0)
    team_0_pred = Column(String(16), nullable=True)
    team_1_pred = Column(String(16), nullable=True)

    has_spatial_telemetry = Column(Boolean, nullable=False, default=True)
    parquet_dir = Column(String(512), nullable=True)

    metadata_json = Column(JSONB_VARIANT, nullable=False, default=dict)
    features_json = Column(JSONB_VARIANT, nullable=False, default=dict)
    diagnosis_json = Column(JSONB_VARIANT, nullable=False, default=dict)
    heatmap_default_json = Column(JSONB_VARIANT, nullable=False, default=dict)
    radar_json = Column(JSONB_VARIANT, nullable=False, default=dict)
    goals_json = Column(JSONB_VARIANT, nullable=False, default=list)
    demolitions_json = Column(JSONB_VARIANT, nullable=False, default=list)

    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow)


class AnalysisJobRecord(Base):
    """Estado persistido de um job de processamento de replay."""

    __tablename__ = "analysis_jobs"
    __table_args__ = {"schema": ANALYTICS_SCHEMA}

    job_id = Column(String(64), primary_key=True, index=True)
    replay_id = Column(String(64), nullable=False, index=True)
    status = Column(String(16), nullable=False, default="queued", index=True)
    stage = Column(String(32), nullable=False, default="queued")
    progress = Column(Float, nullable=False, default=0.0)
    error_code = Column(String(64), nullable=True)
    error_message = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow)

