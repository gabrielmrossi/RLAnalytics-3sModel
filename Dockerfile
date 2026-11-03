# syntax=docker/dockerfile:1.7
# Estágio 1: Build do ambiente virtual Python 3.13 com pins exatos (core + api)
FROM python:3.13-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml README.md ./
COPY rl_analytics/ ./rl_analytics/
COPY scripts/ ./scripts/

RUN pip install --upgrade pip setuptools wheel && \
    pip install ".[core,api]"

# Estágio 2: Runtime enxuto, determinístico e não-root
FROM python:3.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app" \
    RRROCKET_PATH="/usr/local/bin/rrrocket" \
    MODEL_PATH="/app/artifacts/models/model_detector.v2.joblib" \
    REFERENCE_CSV_PATH="/app/data/processed/features_dataset.csv" \
    DATASET_CSV_PATH="/app/data/processed/features_dataset.csv" \
    DATA_DIR="/data" \
    PARQUET_DIR="/data/parquet" \
    TMP_DIR="/tmp"

WORKDIR /app

# 1. Copiar e verificar binário Linux local (offline, determinístico, musl estático)
COPY --chmod=0755 bin/linux/rrrocket /usr/local/bin/rrrocket
COPY bin/linux/SHA256SUMS bin/linux/VERSION /tmp/rrrocket_meta/
RUN cd /tmp/rrrocket_meta && \
    cp /usr/local/bin/rrrocket ./rrrocket && \
    sha256sum -c SHA256SUMS && \
    rm -rf /tmp/rrrocket_meta && \
    /usr/local/bin/rrrocket --help >/dev/null

# 2. Copiar venv do builder e código de domínio + artefatos v2
COPY --from=builder /opt/venv /opt/venv
COPY rl_analytics/ ./rl_analytics/
COPY scripts/ ./scripts/
COPY artifacts/models/model_detector.v2.joblib ./artifacts/models/model_detector.v2.joblib
COPY data/processed/features_dataset.csv ./data/processed/features_dataset.csv
COPY tests/ ./tests/

# 3. Smoke test no build: valida unpickle v2, predição e ausência do Streamlit (G4)
RUN python -c "\
import numpy as np; \
from rl_analytics.error_detection import ModelDrivenErrorDetector; \
det = ModelDrivenErrorDetector.load(); \
assert det.is_built and len(det.feature_names) == 10; \
x = np.zeros((1, len(det.feature_names)), dtype=float); \
pred = det.predict(x); \
proba = det.predict_proba(x); \
assert len(pred) == 1 and len(proba) == 1; \
print('[smoke] Modelo v2 OK:', pred[0], proba[0])" && \
    python -c "\
try:\
    import streamlit;\
    raise SystemExit('ERRO: streamlit nao deveria estar instalado na imagem!')\
except ModuleNotFoundError:\
    print('[smoke] Desacoplamento Streamlit (G4) OK')"

# 4. Usuário não-root e diretórios de dados
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -m -s /usr/sbin/nologin appuser && \
    mkdir -p /data/uploads /data/parquet && \
    chown -R appuser:appgroup /app /data

USER appuser

ENTRYPOINT ["python", "-m", "rl_analytics.facade"]
CMD ["--help"]

