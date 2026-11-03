"""
worker.py — Entrypoint do container `analytics-worker` (RQ).

Realiza:
1. Faxina automática de arquivos `/tmp/rl-*.json` órfãos no startup (§5.3).
2. Pré-carregamento do modelo `.joblib` v2 e do CSV de referência na memória antes do fork (§5.4).
3. Inicialização idempotente do schema `analytics` no banco.
4. Execução do worker RQ (SimpleWorker no Windows / Worker com fork no Linux).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from rl_analytics.config import TMP_DIR
from rl_analytics_api.database import init_analytics_schema
from rl_analytics_api.tasks import get_redis_connection, preload_worker_resources


def cleanup_orphan_tmp_files() -> int:
    """Remove arquivos temporários `/tmp/rl-*.json` órfãos de execuções anteriores."""
    tmp_path = Path(os.getenv("TMP_DIR", str(TMP_DIR)))
    if not tmp_path.exists():
        return 0
    removed = 0
    for f in tmp_path.glob("rl-*.json"):
        try:
            f.unlink()
            removed += 1
        except OSError:
            pass
    return removed


def main() -> int:
    removed = cleanup_orphan_tmp_files()
    print(f"[worker] Faxina de startup em TMP_DIR concluída ({removed} órfãos removidos).")

    detector, ref_df = preload_worker_resources()
    print(
        f"[worker] Modelo v2 pré-carregado antes do fork: "
        f"features={len(detector.feature_names)}, ref_rows={len(ref_df) if ref_df is not None else 0}"
    )

    init_analytics_schema()
    print("[worker] Schema 'analytics' verificado com sucesso.")

    from rq import Queue, SimpleWorker, Worker

    conn = get_redis_connection()
    q_name = os.getenv("ANALYTICS_QUEUE_NAME", "analytics")
    queue = Queue(name=q_name, connection=conn)

    worker_cls = SimpleWorker if sys.platform.startswith("win") else Worker
    worker = worker_cls([queue], connection=conn)
    print(f"[worker] Escutando fila '{q_name}' via {worker_cls.__name__}...")
    worker.work(with_scheduler=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())

