"""
parser.py — Converte arquivos .replay em JSON usando rrrocket e gerencia extração segura.

Inclui:
- Validação de tamanho e assinatura de header (TAGame.Replay_Soccar_TA).
- Timeout explícito no subprocesso do rrrocket.
- Redirecionamento de stdout para arquivo temporário em TMP_DIR com deleção imediata em finally.
- Persistência atômica de telemetria extraída em Parquet (positions.parquet / boost.parquet).
"""

from __future__ import annotations

import json
import os
import subprocess
import uuid
from pathlib import Path
from typing import Any

from rl_analytics.analyzer import ReplayAnalyzer
from rl_analytics.config import (
    PARQUET_DIR as _PARQUET_DIR,
    PROCESSED_DIR as _PROCESSED_DIR,
    RAW_DIR as _RAW_DIR,
    RRROCKET_PATH as _RRROCKET_PATH,
    TMP_DIR as _TMP_DIR,
)

RRROCKET_PATH = str(_RRROCKET_PATH)
RAW_DIR = str(_RAW_DIR)
PROCESSED_DIR = str(_PROCESSED_DIR)
PARQUET_DIR = str(_PARQUET_DIR)
TMP_DIR = str(_TMP_DIR)

MAX_REPLAY_BYTES = 10 * 1024 * 1024  # 10 MiB
DEFAULT_PARSE_TIMEOUT_SEC = 20.0
REPLAY_HEADER_SIGNATURE = b"TAGame.Replay_Soccar_TA"

# Mapeamento de nome de pasta para label
_FOLDER_LABEL_MAP = {
    "higher_rank_replays": "higher",
    "lower_rank_replays": "lower",
    "higher": "higher",
    "lower": "lower",
}


class ReplayParseError(RuntimeError):
    """Erro estruturado durante validação ou decodificação de um arquivo .replay."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def validate_replay_file(replay_path: str | Path, max_bytes: int = MAX_REPLAY_BYTES) -> Path:
    """Valida existência, extensão, limite de tamanho e assinatura binária de um .replay."""
    path = Path(replay_path)
    if not path.exists() or not path.is_file():
        raise ReplayParseError("NOT_FOUND", f"Arquivo de replay não encontrado: {path}")

    if path.suffix.lower() != ".replay":
        raise ReplayParseError("INVALID_REPLAY", f"Extensão inválida (esperado .replay): {path.name}")

    size = path.stat().st_size
    if size <= 0 or size > max_bytes:
        raise ReplayParseError(
            "INVALID_REPLAY",
            f"Tamanho de arquivo inválido ({size} bytes; limite={max_bytes} bytes).",
        )

    with open(path, "rb") as f:
        header_chunk = f.read(8192)
    if REPLAY_HEADER_SIGNATURE not in header_chunk:
        raise ReplayParseError(
            "INVALID_REPLAY",
            "Assinatura TAGame.Replay_Soccar_TA ausente no cabeçalho do arquivo.",
        )

    return path


def parse_single_replay(
    replay_path: str,
    *,
    timeout_sec: float = DEFAULT_PARSE_TIMEOUT_SEC,
    rrrocket_bin: str | None = None,
    tmp_dir: str | None = None,
    strict: bool = False,
) -> dict[str, Any] | None:
    """Converte 1 arquivo .replay em dict Python via rrrocket com arquivo temporário efêmero.

    O stdout do rrrocket é gravado diretamente em um arquivo temporário em `tmp_dir`
    (evitando manter stdout + dict duplicados em RAM) e deletado imediatamente no bloco `finally`.
    """
    if not os.path.exists(replay_path):
        print(f"[parser] Arquivo não encontrado: {replay_path}")
        if strict:
            raise ReplayParseError("NOT_FOUND", f"Arquivo não encontrado: {replay_path}")
        return None

    try:
        validate_replay_file(replay_path)
    except ReplayParseError as err:
        print(f"[parser] Validação falhou ({err.code}) para {replay_path}: {err.message}")
        if strict:
            raise
        return None

    binary = rrrocket_bin or os.getenv("RRROCKET_PATH", RRROCKET_PATH)
    work_tmp_dir = Path(tmp_dir or os.getenv("TMP_DIR", TMP_DIR))
    work_tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_json_path = work_tmp_dir / f"rl-{uuid.uuid4().hex}.json"

    try:
        with open(tmp_json_path, "wb") as out_f:
            result = subprocess.run(
                [binary, "-n", str(replay_path)],
                stdout=out_f,
                stderr=subprocess.PIPE,
                timeout=timeout_sec,
                check=False,
            )

        if result.returncode != 0:
            stderr_msg = result.stderr.decode("utf-8", errors="replace")[:500]
            print(f"[parser] Erro no rrrocket para {replay_path}: {stderr_msg}")
            if strict:
                raise ReplayParseError("INVALID_REPLAY", f"Falha no rrrocket: {stderr_msg}")
            return None

        with open(tmp_json_path, "r", encoding="utf-8") as in_f:
            return json.load(in_f)

    except subprocess.TimeoutExpired as exc:
        print(f"[parser] Timeout ({timeout_sec}s) no rrrocket para {replay_path}")
        if strict:
            raise ReplayParseError(
                "PARSE_TIMEOUT", f"rrrocket excedeu o limite de {timeout_sec}s"
            ) from exc
        return None
    except ReplayParseError:
        raise
    except Exception as e:
        print(f"[parser] Exceção ao processar {replay_path}: {e}")
        if strict:
            raise ReplayParseError("INVALID_REPLAY", f"Erro ao decodificar replay: {e}") from e
        return None
    finally:
        # Deleção imediata garantida do JSON bruto efêmero
        if tmp_json_path.exists():
            try:
                os.remove(tmp_json_path)
            except OSError:
                pass


def parse_and_extract_With_lifecycle(
    replay_path: str | Path,
    *,
    replay_id: str | None = None,
    parquet_dir: str | Path | None = None,
    persist_parquet: bool = True,
    timeout_sec: float = DEFAULT_PARSE_TIMEOUT_SEC,
) -> dict[str, Any]:
    """Executa rrrocket -> ReplayAnalyzer -> deleção imediata do JSON bruto -> escrita atômica de Parquet.

    Garante que o JSON bruto viva apenas durante `parse` + `analyze()` (~0.5s) e seja removido
    imediatamente antes de qualquer etapa subsequente.
    """
    valid_path = validate_replay_file(replay_path)
    rid = replay_id or valid_path.stem
    binary = os.getenv("RRROCKET_PATH", RRROCKET_PATH)
    work_tmp_dir = Path(os.getenv("TMP_DIR", TMP_DIR))
    work_tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_json_path = work_tmp_dir / f"rl-{rid}-{uuid.uuid4().hex[:8]}.json"

    raw_json_bytes = 0
    analysis_result: dict[str, Any] | None = None

    try:
        try:
            with open(tmp_json_path, "wb") as out_f:
                proc = subprocess.run(
                    [binary, "-n", str(valid_path)],
                    stdout=out_f,
                    stderr=subprocess.PIPE,
                    timeout=timeout_sec,
                    check=False,
                )
        except subprocess.TimeoutExpired as exc:
            raise ReplayParseError(
                "PARSE_TIMEOUT", f"rrrocket excedeu o limite de {timeout_sec}s"
            ) from exc

        if proc.returncode != 0:
            stderr_msg = proc.stderr.decode("utf-8", errors="replace")[:500]
            raise ReplayParseError("INVALID_REPLAY", f"rrrocket falhou: {stderr_msg}")

        raw_json_bytes = tmp_json_path.stat().st_size
        with open(tmp_json_path, "r", encoding="utf-8") as in_f:
            raw_data = json.load(in_f)

        analyzer = ReplayAnalyzer(raw_data)
        analysis_result = analyzer.analyze()
        del raw_data
    finally:
        # Deleção imediata do JSON bruto logo após analyze()
        if tmp_json_path.exists():
            try:
                os.remove(tmp_json_path)
            except OSError:
                pass

    assert analysis_result is not None
    parquet_paths: dict[str, str] = {}
    if persist_parquet:
        base_pq_dir = Path(parquet_dir or os.getenv("PARQUET_DIR", PARQUET_DIR)) / rid
        base_pq_dir.mkdir(parents=True, exist_ok=True)

        for key in ("positions", "boost"):
            df = analysis_result[key]
            target_pq = base_pq_dir / f"{key}.parquet"
            tmp_pq = base_pq_dir / f".{key}.{uuid.uuid4().hex[:8]}.tmp.parquet"
            df.to_parquet(tmp_pq, index=False)
            os.replace(tmp_pq, target_pq)
            parquet_paths[key] = str(target_pq)

    analysis_result["raw_json_bytes"] = raw_json_bytes
    analysis_result["parquet_paths"] = parquet_paths
    return analysis_result


def ensure_processed(
    replay_path: str, processed_dir: str = PROCESSED_DIR
) -> tuple[dict[str, Any], str] | None:
    """Se o JSON já existe em processed/, carrega. Senão, parseia e salva compacto (sem indent=2)."""
    stem = Path(replay_path).stem
    target = Path(processed_dir) / f"{stem}.json"
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists():
        print(f"[parser] Cache encontrado: {target}")
        with open(target, "r", encoding="utf-8") as f:
            return json.load(f), str(target)

    print(f"[parser] Processando: {replay_path}")
    data = parse_single_replay(replay_path)
    if data is None:
        return None

    with open(target, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))

    print(f"[parser] JSON salvo: {target}")
    return data, str(target)


def discover_replays(raw_dir: str = RAW_DIR) -> dict[str, list[str]]:
    """Descobre todos os .replay organizados por categoria."""
    root = Path(raw_dir)
    if not root.exists():
        return {}

    result: dict[str, list[str]] = {}

    for subfolder in root.iterdir():
        if subfolder.is_dir():
            label = _FOLDER_LABEL_MAP.get(subfolder.name)
            if label:
                replays = sorted(str(p) for p in subfolder.glob("*.replay"))
                if replays:
                    result.setdefault(label, []).extend(replays)

    root_replays = sorted(str(p) for p in root.glob("*.replay"))
    if root_replays:
        result["unlabeled"] = root_replays

    return result


def process_all_replays(
    raw_dir: str = RAW_DIR, processed_dir: str = PROCESSED_DIR
) -> dict[str, list[tuple[dict[str, Any], str]]]:
    """Descobre e processa todos os replays."""
    discovered = discover_replays(raw_dir)

    if not discovered:
        print("[parser] Nenhum replay encontrado.")
        return {}

    total = sum(len(v) for v in discovered.values())
    print(
        f"[parser] {total} replays encontrados: "
        f"{', '.join(f'{k}={len(v)}' for k, v in discovered.items())}"
    )

    results: dict[str, list[tuple[dict[str, Any], str]]] = {}

    for label, paths in discovered.items():
        results[label] = []
        for path in paths:
            processed = ensure_processed(path, processed_dir)
            if processed:
                results[label].append(processed)
            else:
                print(f"[parser] FALHA ao processar: {path}")

    success = sum(len(v) for v in results.values())
    print(f"[parser] Concluído: {success}/{total} replays processados com sucesso.")
    return results
