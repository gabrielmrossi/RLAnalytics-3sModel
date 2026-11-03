"""
database.py — Configuração do SQLAlchemy e bootstrap idempotente do schema `analytics` (D2).
"""

from __future__ import annotations

import os
from collections.abc import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker
from sqlalchemy.pool import StaticPool

ANALYTICS_SCHEMA = os.getenv("ANALYTICS_DB_SCHEMA", "analytics")

Base = declarative_base()


def build_database_url() -> str:
    """Constrói a URL de conexão ao banco a partir de DATABASE_URL ou variáveis POSTGRES_*."""
    explicit_url = os.getenv("ANALYTICS_DATABASE_URL") or os.getenv("DATABASE_URL")
    if explicit_url:
        if explicit_url.startswith("postgresql://"):
            return explicit_url.replace("postgresql://", "postgresql+psycopg2://", 1)
        return explicit_url

    pg_user = os.getenv("POSTGRES_USER", "postgres")
    pg_pass = os.getenv("POSTGRES_PASSWORD", "postgres")
    pg_host = os.getenv("POSTGRES_HOST", "postgres")
    pg_port = os.getenv("POSTGRES_PORT", "5432")
    pg_db = os.getenv("POSTGRES_DB", "postgres")
    return f"postgresql+psycopg2://{pg_user}:{pg_pass}@{pg_host}:{pg_port}/{pg_db}"


def create_analytics_engine(db_url: str | None = None) -> Engine:
    url = db_url or build_database_url()
    if url.startswith("sqlite"):
        return create_engine(
            url,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
            execution_options={"schema_translate_map": {ANALYTICS_SCHEMA: None}},
        )

    return create_engine(url, pool_pre_ping=True)


engine: Engine = create_analytics_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def configure_database(db_url: str) -> Engine:
    """Reconfigura o engine e SessionLocal (útil para testes e overrides de ambiente)."""
    global engine, SessionLocal
    if engine is not None:
        engine.dispose()
    engine = create_analytics_engine(db_url)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return engine


def init_analytics_schema(target_engine: Engine | None = None) -> None:
    """Cria de forma idempotente o schema `analytics` e as tabelas do serviço."""
    import rl_analytics_api.models  # noqa: F401 — garante registro dos modelos no Base.metadata

    eng = target_engine or engine
    if eng.dialect.name == "postgresql":
        with eng.begin() as conn:
            conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {ANALYTICS_SCHEMA}"))
            role_name = os.getenv("ANALYTICS_DB_ROLE", "analytics_svc")
            conn.execute(
                text(
                    f"""
                    DO $$
                    BEGIN
                        IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{role_name}') THEN
                            CREATE ROLE {role_name} NOLOGIN;
                        END IF;
                    EXCEPTION WHEN insufficient_privilege THEN
                        NULL;
                    END
                    $$;
                    """
                )
            )
            conn.execute(
                text(
                    f"""
                    DO $$
                    BEGIN
                        EXECUTE 'GRANT USAGE, CREATE ON SCHEMA {ANALYTICS_SCHEMA} TO {role_name}';
                    EXCEPTION WHEN undefined_object OR insufficient_privilege THEN
                        NULL;
                    END
                    $$;
                    """
                )
            )
    Base.metadata.create_all(bind=eng)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
